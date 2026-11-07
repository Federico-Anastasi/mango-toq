// GELU on the CPU against the reference in double, on a dense grid of finite values and on large values:
//   ggml_gelu     0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))   (gelu_pytorch_tanh, the Qwen3-VL encoder)
//   ggml_gelu_erf 0.5 x (1 + erf(x / sqrt 2))
// A single non-finite value in the encoder's FFN becomes a whole row of NaN after the next product: with
// -ffp-model=fast the vectorized GELU gave inf for x just below 10 (layer 18 of the Qwen3.8 encoder, x = 9.978).

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int check(ggml_tensor * (*op)(ggml_context *, ggml_tensor *), double (*ref)(double), const char * name,
                 const std::vector<float> & xs) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        ((float *) x->data)[i] = xs[i];
    }
    ggml_tensor * y = op(ctx, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    int bad = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double r = ref(xs[i]);
        const float got = ((const float *) y->data)[i];
        // fp32 with approximated exp/erf: 1e-5 relative (or absolute below 1)
        const bool ok = std::isfinite(got) && std::fabs(got - r) <= 1e-5 * std::fmax(1.0, std::fabs(r));
        if (!ok && bad++ < 10) {
            printf("FAIL %s(%.9g) = %.9g, expected %.9g\n", name, xs[i], got, r);
        }
    }
    printf("%s %s: %zu values, %d wrong\n", bad ? "FAIL" : "ok  ", name, xs.size(), bad);
    ggml_free(ctx);
    return bad;
}

int main() {
    std::vector<float> xs;
    for (int i = -400000; i <= 400000; ++i) {
        xs.push_back(i * 1e-4f);                  // [-40, 40] in steps of 1e-4
    }
    for (int i = 0; i <= 100000; ++i) {
        xs.push_back(9.9f + i * 1e-6f);           // [9.9, 10] dense: the threshold of the vectorized GELU
    }
    for (float v : { 1e2f, 1e3f, 1e4f, 1e6f, 1e10f, 1e30f, 3e38f }) {
        xs.push_back(v);
        xs.push_back(-v);
    }
    const int bad =
        check(ggml_gelu, [](double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); },
              "gelu", xs) +
        check(ggml_gelu_erf, [](double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }, "gelu_erf", xs);
    return bad ? 1 : 0;
}
