// Product with TRL weights (ggml-trl.h) on every non-CPU backend, against a reference on the CPU.
//
// The backend computes the product the way llm_build_lora_mm builds it: ggml_mul_mat(W, x) with the vector d in
// src[2] and the Hadamard block / GDN reorder in op_params[1..4]. The reference uses only pieces the CPU
// already has: W decoded with to_float, then the same graph as the old path (llm_build_hadamard_rotate:
// reorder, MUL by d, HADAMARD) and an F32 product.
// The weights are random bytes with hand-written scales: every bit sequence is a valid TRL weight.
//
//   test-trl-backend [backend name]      without a name: all non-CPU backends

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct trl_case {
    ggml_type type;
    int64_t K, N, ncols;
    int blk, hd, nk, rep;   // blk = 0: no d, input already rotated
    int64_t row0, rows;     // weight rows used (view, like the sliced head); rows = 0: all
    int64_t d_col;          // column of d [K, d_col + 1] used (view, like the sliced head)
    int64_t x_pad;          // extra floats in front of and between the columns of x (unaligned view)
};

static std::mt19937 rng(1234);

static std::vector<uint8_t> random_trl(ggml_type type, int64_t K, int64_t N) {
    std::vector<uint8_t> w(ggml_row_size(type, K) * N);
    std::uniform_int_distribution<uint32_t> bits;
    for (size_t i = 0; i + 4 <= w.size(); i += 4) {
        const uint32_t v = bits(rng);
        memcpy(w.data() + i, &v, 4);
    }
    // each band of 64 rows starts with the 64 fp32 scales
    std::uniform_real_distribution<float> scale(-2.0f, 2.0f);
    const size_t band = ggml_row_size(type, K) * 64;
    for (int64_t b = 0; b < N / 64; ++b) {
        for (int r = 0; r < 64; ++r) {
            const float s = scale(rng);
            memcpy(w.data() + b * band + 4 * r, &s, 4);
        }
    }
    return w;
}

static std::vector<float> random_f32(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = u(rng);
    return v;
}

// y_ref = Wf · Hblk(d ⊙ perm(x)), on the CPU
static std::vector<float> reference(const trl_case & c, const std::vector<uint8_t> & w, const std::vector<float> & x,
                                    const std::vector<float> & d) {
    const int64_t K = c.K, rows = c.rows ? c.rows : c.N;
    const auto traits = ggml_internal_get_type_traits(c.type);
    const size_t band = ggml_row_size(c.type, K) * 64;
    std::vector<float> wf(K * rows);
    for (int64_t b = 0; b < rows / 64; ++b) {
        traits.to_float(w.data() + (c.row0 / 64 + b) * band, wf.data() + b * 64 * K, 64 * K);
    }

    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows);
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, c.ncols);
    memcpy(tw->data, wf.data(), ggml_nbytes(tw));
    for (int64_t j = 0; j < c.ncols; ++j) {
        memcpy((float *) tx->data + j * K, x.data() + c.x_pad + j * (K + c.x_pad), K * sizeof(float));
    }
    ggml_tensor * xr = tx;
    if (c.blk) {
        if (c.rep > 1) {
            xr = ggml_reshape_4d(ctx, xr, c.hd, c.nk, c.rep, c.ncols);
            xr = ggml_cont(ctx, ggml_permute(ctx, xr, 0, 2, 1, 3));
            xr = ggml_reshape_2d(ctx, xr, K, c.ncols);
        }
        ggml_tensor * td = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        memcpy(td->data, d.data() + c.d_col * K, K * sizeof(float));
        xr = ggml_mul(ctx, xr, td);
        xr = ggml_hadamard(ctx, ggml_reshape_2d(ctx, xr, c.blk, K / c.blk * c.ncols), c.blk);
        xr = ggml_reshape_2d(ctx, xr, K, c.ncols);
    }
    ggml_tensor * y = ggml_mul_mat(ctx, tw, xr);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    std::vector<float> out((float *) y->data, (float *) y->data + ggml_nelements(y));
    ggml_free(ctx);
    return out;
}

// 1: passed, 0: failed, -1: the backend does not support it
static int run_case(ggml_backend_t backend, const trl_case & c) {
    const int64_t K = c.K, rows = c.rows ? c.rows : c.N;
    const auto w = random_trl(c.type, K, c.N);
    const auto x = random_f32(c.x_pad + c.ncols * (K + c.x_pad), -1.0f, 1.0f);
    const auto d = random_f32(K * (c.d_col + 1), -1.5f, 1.5f);

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tw  = ggml_new_tensor_2d(ctx, c.type, K, c.N);
    ggml_tensor * txb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) x.size());
    ggml_tensor * tdb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, c.d_col + 1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(tw, w.data(), 0, w.size());
    ggml_backend_tensor_set(txb, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(tdb, d.data(), 0, d.size() * sizeof(float));

    ggml_tensor * wv = c.rows ? ggml_view_2d(ctx, tw, K, rows, tw->nb[1], c.row0 * tw->nb[1]) : tw;
    ggml_tensor * xv = ggml_view_2d(ctx, txb, K, c.ncols, (K + c.x_pad) * sizeof(float), c.x_pad * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, wv, xv);
    if (c.blk) {
        y->src[2] = ggml_view_1d(ctx, tdb, K, c.d_col * tdb->nb[1]);
        int32_t * op = (int32_t *) y->op_params;
        op[1] = c.blk;
        op[2] = c.hd;
        op[3] = c.nk;
        op[4] = c.rep;
    }

    int res = -1;
    if (ggml_backend_supports_op(backend, y)) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
        ggml_backend_graph_compute(backend, gf);
        std::vector<float> got(ggml_nelements(y));
        ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
        ggml_gallocr_free(galloc);

        const auto ref = reference(c, w, x, d);
        double err = 0, norm = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            err  += (got[i] - ref[i]) * (double)(got[i] - ref[i]);
            norm += ref[i] * (double) ref[i];
        }
        const double nmse = err / norm;
        res = std::isfinite(nmse) && nmse < 1e-6 ? 1 : 0;
        printf("  %s K=%lld rows=%lld col=%lld blk=%d rep=%d view=%lld+%lld d_col=%lld x_pad=%lld: nmse %.3g %s\n",
               ggml_type_name(c.type), (long long) K, (long long) rows, (long long) c.ncols, c.blk, c.rep,
               (long long) c.row0, (long long) rows, (long long) c.d_col, (long long) c.x_pad, nmse, res ? "OK" : "FAILED");
    } else {
        printf("  %s K=%lld rows=%lld col=%lld blk=%d: not supported\n", ggml_type_name(c.type), (long long) K,
               (long long) rows, (long long) c.ncols, c.blk);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // lines get out even if the backend crashes
    const char * filter = argc > 1 ? argv[1] : nullptr;
    std::vector<trl_case> cases;
    for (ggml_type t : { GGML_TYPE_TRL2_M, GGML_TYPE_TRL3_M }) {
        cases.push_back({ t,  256,  64, 1,  256,  1, 1, 1,   0,  0, 0, 0 });
        cases.push_back({ t,  512, 128, 3,  128, 32, 4, 4,   0,  0, 0, 0 });   // GDN reorder
        cases.push_back({ t, 1024, 192, 9, 1024,  1, 1, 1,   0,  0, 0, 0 });   // more column groups
        cases.push_back({ t, 2048,  64, 2, 2048,  1, 1, 1,   0,  0, 0, 0 });
        cases.push_back({ t,  512, 256, 5,  256,  1, 1, 1, 128, 64, 2, 3 });   // slice of rows, d and x as views
        cases.push_back({ t,  256,  64, 2,    0,  1, 1, 1,   0,  0, 0, 0 });   // input already rotated
    }

    int tried = 0, failed = 0;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char * name = ggml_backend_reg_get_name(i);
        if (filter ? strcmp(filter, name) != 0 : strcmp(name, "CPU") == 0) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_reg_init_backend(i, nullptr);
        if (backend == nullptr || ggml_backend_is_cpu(backend)) {
            continue;
        }
        printf("backend %s\n", ggml_backend_name(backend));
        ++tried;
        for (const auto & c : cases) {
            failed += run_case(backend, c) != 1;   // "not supported" is a failure too: the backend must do it
        }
        ggml_backend_free(backend);
    }
    if (tried == 0) {
        printf("no backend to test\n");
        return 1;
    }
    printf("%s\n", failed ? "FAILED" : "OK");
    return failed ? 1 : 0;
}
