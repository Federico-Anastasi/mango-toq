// TRL product on the CPU (ggml-trl.h) against the reference: the same weights decoded to fp32 by the type's
// to_float and the input rotation done with the graph operations (GDN reorder, ggml_mul by d, ggml_hadamard), as
// llm_build_hadamard_rotate does it for non-TRL weights. The TRL product receives d in src[2] and the block and the
// reorder in op_params, as llm_build_lora_mm does.
//
// The tile bytes are random (every bit sequence is a valid weight), the row scales and d are random values.
// Shapes: 2 and 3 bits, gaussian codebook and mul1, several bands, a view of rows (the sliced head), columns from 1 to 37
// (decode and prompt prefill), with and without rotation, with the V heads reorder, several threads.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct trl_case {
    ggml_type type;
    int64_t K, N;       // N weight rows, in bands of 64
    int64_t row0, rows; // view of rows [row0, row0 + rows)
    int64_t M;          // columns of the input
    int blk, hd, nk, rep; // blk = 0: no rotation (the input arrives already rotated)
    int n_threads;
};

static int bits_of(ggml_type type) {
    return type == GGML_TYPE_TRL3_G || type == GGML_TYPE_TRL3_M ? 3 : 2;
}

static void dump(const char * dir, const trl_case & tc, const char * what, const void * data, size_t n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", dir, ggml_type_name(tc.type), what);
    FILE * f = fopen(path, "wb");
    fwrite(data, 1, n, f);
    fclose(f);
}

// dump_dir: writes weights, d, x and the TRL product for comparison with a reference implementation
static bool run_case(const trl_case & tc, std::mt19937 & rng, const char * dump_dir = nullptr) {
    const int64_t band_bytes = 64 * (4 + tc.K * bits_of(tc.type) / 8);
    const int64_t nbands = tc.N / 64;

    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, tc.type, tc.K, tc.N);
    if ((int64_t) ggml_nbytes(w) != nbands * band_bytes) {
        printf("FAIL %s: ggml_nbytes %zu, expected %lld\n", ggml_type_name(tc.type), ggml_nbytes(w),
               (long long) (nbands * band_bytes));
        ggml_free(ctx);
        return false;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> unif(0.5f, 1.5f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    uint8_t * wb = (uint8_t *) w->data;
    for (int64_t i = 0; i < nbands * band_bytes; ++i) {
        wb[i] = (uint8_t) byte(rng);
    }
    for (int64_t ib = 0; ib < nbands; ++ib) {
        float * s = (float *) (wb + ib * band_bytes);
        for (int r = 0; r < 64; ++r) {
            s[r] = (r & 1 ? -1.0f : 1.0f) * unif(rng);
        }
    }

    ggml_tensor * wf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.N);
    const auto traits = ggml_internal_get_type_traits(tc.type);
    for (int64_t ib = 0; ib < nbands; ++ib) {
        traits.to_float(wb + ib * band_bytes, (float *) wf->data + ib * 64 * tc.K, 64 * tc.K);
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.K, tc.M);
    for (int64_t i = 0; i < tc.K * tc.M; ++i) {
        ((float *) x->data)[i] = gauss(rng);
    }
    ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tc.K);
    for (int64_t i = 0; i < tc.K; ++i) {
        ((float *) d->data)[i] = (i % 3 ? 1.0f : -1.0f) * unif(rng);
    }

    ggml_tensor * wv  = ggml_view_2d(ctx, w,  tc.K, tc.rows, w->nb[1],  tc.row0 * w->nb[1]);
    ggml_tensor * wfv = ggml_view_2d(ctx, wf, tc.K, tc.rows, wf->nb[1], tc.row0 * wf->nb[1]);

    ggml_tensor * xr = x;
    if (tc.blk > 0) {
        if (tc.rep > 1) {
            xr = ggml_reshape_4d(ctx, xr, tc.hd, tc.nk, tc.rep, tc.M);
            xr = ggml_cont(ctx, ggml_permute(ctx, xr, 0, 2, 1, 3));
            xr = ggml_reshape_2d(ctx, xr, tc.K, tc.M);
        }
        xr = ggml_mul(ctx, xr, d);
        xr = ggml_hadamard(ctx, ggml_reshape_2d(ctx, xr, tc.blk, tc.K * tc.M / tc.blk), tc.blk);
        xr = ggml_reshape_2d(ctx, xr, tc.K, tc.M);
    }
    ggml_tensor * ref = ggml_mul_mat(ctx, wfv, xr);

    ggml_tensor * out = ggml_mul_mat(ctx, wv, x);
    if (tc.blk > 0) {
        out->src[2] = d;
        int32_t * op = (int32_t *) out->op_params;
        op[1] = tc.blk;
        op[2] = tc.hd;
        op[3] = tc.nk;
        op[4] = tc.rep;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, tc.n_threads);

    double err2 = 0.0, ref2 = 0.0;
    bool finite = true;
    for (int64_t c = 0; c < tc.M; ++c) {
        for (int64_t r = 0; r < tc.rows; ++r) {
            const float a = ((const float *) ((const char *) out->data + c * out->nb[1]))[r];
            const float b = ((const float *) ((const char *) ref->data + c * ref->nb[1]))[r];
            finite = finite && std::isfinite(a);
            err2 += (double) (a - b) * (a - b);
            ref2 += (double) b * b;
        }
    }
    const double nmse = err2 / ref2;
    if (dump_dir != nullptr) {
        dump(dump_dir, tc, "blob", w->data, ggml_nbytes(w));
        dump(dump_dir, tc, "d", d->data, ggml_nbytes(d));
        dump(dump_dir, tc, "x", x->data, ggml_nbytes(x));
        dump(dump_dir, tc, "y", out->data, ggml_nbytes(out));
    }
    // fp32 against fp32 with a different summation order: the expected relative error is ~1e-7 (nmse ~1e-14)
    const bool ok = finite && nmse < 1e-10;
    printf("%s %-6s K=%5lld rows %3lld..%3lld M=%2lld blk=%4d rep=%d thr=%d  nmse %.3e\n", ok ? "ok  " : "FAIL",
           ggml_type_name(tc.type), (long long) tc.K, (long long) tc.row0, (long long) (tc.row0 + tc.rows),
           (long long) tc.M, tc.blk, tc.rep, tc.n_threads, nmse);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    std::mt19937 rng(1234);
    if (argc > 1) {
        bool ok = true;
        for (ggml_type t : { GGML_TYPE_TRL2_M, GGML_TYPE_TRL3_M, GGML_TYPE_TRL2_G, GGML_TYPE_TRL3_G }) {
            ok = run_case({ t, 512, 128, 0, 128, 2, 256, 1, 1, 1, 2 }, rng, argv[1]) && ok;
        }
        return ok ? 0 : 1;
    }
    const ggml_type types[] = { GGML_TYPE_TRL2_M, GGML_TYPE_TRL3_M, GGML_TYPE_TRL2_G, GGML_TYPE_TRL3_G };
    const int64_t cols[] = { 1, 2, 3, 4, 5, 8, 37 };
    std::vector<trl_case> cases;
    for (ggml_type t : types) {
        for (int64_t M : cols) {
            cases.push_back({ t, 512, 192, 0, 192, M, 256, 1, 1, 1, 3 });          // three bands, block 256
        }
        cases.push_back({ t, 768, 128, 0, 128, 2, 0, 1, 1, 1, 2 });                // input already rotated
        cases.push_back({ t, 768, 128, 0, 128, 9, 0, 1, 1, 1, 2 });
        cases.push_back({ t, 1024, 256, 64, 128, 1, 1024, 1, 1, 1, 4 });          // view of rows, whole block
        cases.push_back({ t, 1024, 256, 128, 128, 6, 1024, 1, 1, 1, 4 });
        cases.push_back({ t, 512, 128, 0, 128, 1, 128, 64, 2, 4, 3 });             // V heads reorder
        cases.push_back({ t, 512, 128, 0, 128, 17, 128, 64, 2, 4, 3 });
    }
    int failures = 0;
    for (const trl_case & tc : cases) {
        failures += run_case(tc, rng) ? 0 : 1;
    }
    printf("%s: %d cases, %d failed\n", failures ? "FAIL" : "OK", (int) cases.size(), failures);
    return failures ? 1 : 0;
}
