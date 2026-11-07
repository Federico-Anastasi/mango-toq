// The operations of the DeltaNet layers of Qwen3.5/3.8 (TOQ) on every non-CPU backend, against the CPU:
// DELTA_NET, SSM_CONV, L2_NORM, SOFTPLUS, with the shapes and views that src/llama-delta-net.cpp builds.
// Same graph, same inputs on both backends; the output and the tensors written in place are compared
// (saved_steps, the states after every token for the MTP check).
//
//   test-deltanet-backend [backend name]      without a name: all non-CPU backends

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// An input: the tensor and how to fill it. An output: the tensor to compare.
struct graph_io {
    std::vector<std::pair<ggml_tensor *, std::function<float(std::mt19937 &)>>> inputs;
    std::vector<ggml_tensor *> outputs;
};

using build_fn = std::function<graph_io(ggml_context *)>;

static std::vector<std::vector<float>> run(ggml_backend_t backend, const build_fn & build, bool & supported) {
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    graph_io io = build(ctx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (auto * t : io.outputs) {
        ggml_build_forward_expand(gf, t);
    }
    supported = true;
    for (int i = 0; i < gf->n_nodes; ++i) {
        supported = supported && ggml_backend_supports_op(backend, gf->nodes[i]);
    }
    std::vector<std::vector<float>> res;
    if (!supported) {
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::mt19937 rng(42);
    for (auto & in : io.inputs) {
        if (in.first->type == GGML_TYPE_I32) {
            std::vector<int32_t> v(ggml_nelements(in.first), 0);
            ggml_backend_tensor_set(in.first, v.data(), 0, ggml_nbytes(in.first));
            continue;
        }
        std::vector<float> v(ggml_nelements(in.first));
        for (auto & x : v) x = in.second(rng);
        ggml_backend_tensor_set(in.first, v.data(), 0, ggml_nbytes(in.first));
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_graph_compute(backend, gf);
    for (auto * t : io.outputs) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        res.push_back(std::move(v));
    }
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static std::function<float(std::mt19937 &)> uniform(float lo, float hi) {
    return [lo, hi](std::mt19937 & rng) { return std::uniform_real_distribution<float>(lo, hi)(rng); };
}

// ggml_delta_net like build_fused_delta_net: v, g, beta permuted, q and k normalized before the operation
static build_fn delta_net_case(int64_t S, int64_t H_k, int64_t H_v, int64_t n_tok, int64_t n_seqs, int repeat_type, bool saved) {
    return [=](ggml_context * ctx) {
        graph_io io;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tok, H_k, n_seqs);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tok, H_k, n_seqs);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, n_tok, n_seqs);
        ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_v, n_tok, n_seqs);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H_v, 1, n_tok, n_seqs);
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S * H_v, 1, n_seqs);
        const int64_t state_size = S * S * H_v * n_seqs;
        ggml_tensor * steps = saved ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (n_tok - 1) * state_size) : nullptr;
        io.inputs = { { q, uniform(-1, 1) }, { k, uniform(-1, 1) }, { v, uniform(-1, 1) }, { g, uniform(-2, 0) },
                      { beta, uniform(-3, 3) }, { state, uniform(-0.5f, 0.5f) } };
        if (steps) io.inputs.push_back({ steps, uniform(0, 0) });
        ggml_tensor * qn = ggml_l2_norm(ctx, q, 1e-6f);
        ggml_tensor * kn = ggml_l2_norm(ctx, k, 1e-6f);
        ggml_tensor * r = ggml_delta_net(ctx, qn, kn, ggml_permute(ctx, v, 0, 2, 1, 3), ggml_permute(ctx, g, 2, 0, 3, 1),
                                         ggml_permute(ctx, beta, 2, 0, 1, 3), state, steps);
        r->op_params[0] = repeat_type;
        io.outputs = { r };
        if (steps) io.outputs.push_back(steps);
        return io;
    };
}

// ggml_ssm_conv with one sequence; x as a view of longer rows (qkv_mixed)
static build_fn ssm_conv_case(int64_t nc, int64_t nr, int64_t n_t, bool saved) {
    return [=](ggml_context * ctx) {
        graph_io io;
        ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc - 1, nr, 1);
        ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nr + 8, n_t);
        ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
        ggml_tensor * sq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_t);
        ggml_tensor * steps = saved ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (nc - 1) * nr * n_t) : nullptr;
        io.inputs = { { s, uniform(-1, 1) }, { xb, uniform(-1, 1) }, { c, uniform(-1, 1) }, { sq, nullptr } };
        if (steps) io.inputs.push_back({ steps, uniform(0, 0) });
        ggml_tensor * x = ggml_view_2d(ctx, xb, nr, n_t, xb->nb[1], 4 * sizeof(float));
        // as in the model: after the convolution the view of the output and the SiLU (the CPU fuses them if it can)
        ggml_tensor * conv = ggml_ssm_conv(ctx, s, x, c, sq, steps);
        ggml_tensor * out = ggml_silu(ctx, ggml_view_2d(ctx, conv, nr, n_t, nr * sizeof(float), 0));
        io.outputs = { conv, out };
        if (steps) io.outputs.push_back(steps);
        return io;
    };
}

// ggml_l2_norm on a view like q_conv: heads of a longer token row
static build_fn l2_norm_case(int64_t hd, int64_t heads, int64_t n_tok) {
    return [=](ggml_context * ctx) {
        graph_io io;
        const int64_t row = 3 * hd * heads + 16;
        ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, n_tok);
        io.inputs = { { xb, uniform(-2, 2) } };
        ggml_tensor * x = ggml_view_4d(ctx, xb, hd, heads, 1, n_tok, hd * sizeof(float), xb->nb[1], xb->nb[1], hd * heads * sizeof(float));
        io.outputs = { ggml_l2_norm(ctx, x, 1e-6f) };
        return io;
    };
}

static build_fn softplus_case(int64_t n0, int64_t n1) {
    return [=](ggml_context * ctx) {
        graph_io io;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        io.inputs = { { x, uniform(-30, 30) } };
        io.outputs = { ggml_softplus(ctx, x) };
        return io;
    };
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // lines get out even if the backend crashes
    const char * filter = argc > 1 ? argv[1] : nullptr;
    std::vector<std::pair<std::string, build_fn>> cases = {
        { "DELTA_NET S=128 Hk=2 Hv=4 tok=1 seq=1 rep=0",        delta_net_case(128, 2, 4, 1, 1, 0, false) },
        { "DELTA_NET S=128 Hk=2 Hv=4 tok=5 seq=1 rep=1",        delta_net_case(128, 2, 4, 5, 1, 1, false) },
        { "DELTA_NET S=128 Hk=2 Hv=4 tok=3 seq=1 rep=0 saved",  delta_net_case(128, 2, 4, 3, 1, 0, true) },
        { "DELTA_NET S=64 Hk=1 Hv=2 tok=4 seq=2 rep=0",         delta_net_case(64, 1, 2, 4, 2, 0, false) },
        { "SSM_CONV nc=4 nr=300 tok=1",                         ssm_conv_case(4, 300, 1, false) },
        { "SSM_CONV nc=4 nr=300 tok=7",                         ssm_conv_case(4, 300, 7, false) },
        { "SSM_CONV nc=4 nr=300 tok=3 saved",                   ssm_conv_case(4, 300, 3, true) },
        { "L2_NORM hd=128 heads=4 tok=3 (view)",                l2_norm_case(128, 4, 3) },
        { "SOFTPLUS 100x3",                                     softplus_case(100, 3) },
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
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
            bool sup_b = false, sup_c = false;
            const auto got = run(backend, c.second, sup_b);
            const auto ref = run(cpu, c.second, sup_c);
            GGML_ASSERT(sup_c);
            if (!sup_b) {
                printf("  %s: not supported\n", c.first.c_str());
                ++failed;   // the backend must do it
                continue;
            }
            double worst = 0;
            for (size_t o = 0; o < ref.size(); ++o) {
                double err = 0, norm = 0;
                for (size_t j = 0; j < ref[o].size(); ++j) {
                    err  += (got[o][j] - ref[o][j]) * (double)(got[o][j] - ref[o][j]);
                    norm += ref[o][j] * (double) ref[o][j];
                }
                const double nmse = err / norm;
                worst = std::isfinite(nmse) ? std::max(worst, nmse) : INFINITY;
            }
            const bool ok = worst < 1e-6;
            failed += !ok;
            printf("  %s: nmse %.3g %s\n", c.first.c_str(), worst, ok ? "OK" : "FAILED");
        }
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);
    if (tried == 0) {
        printf("no backend to test\n");
        return 1;
    }
    printf("%s\n", failed ? "FAILED" : "OK");
    return failed ? 1 : 0;
}
