//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "dsa_attn.cuh"
#include "fattn-q8-gqa.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE 256

static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > CC_TURING;
}

// On GPUs without fp16 MMA (Pascal/sm_60), the MLA absorbed head sizes 576/512 are asymmetric, so the f16
// vector kernel (which requires K==V) rejects them and decode falls back to the CPU. Route MLA decode
// (batch <= 8) to the f32 vector kernel to keep it on the GPU. Single-sourced so the dispatch and the
// is_supported check cannot drift.
static inline bool is_pascal_mla_absorbed_decode(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    return Q->ne[1] <= 8 && K->ne[0] == 576 && V->ne[0] == 512;
}

// Pascal has no other kernel for head size 512 (the Gemma-4 global layers), in prefill or decode.
static inline bool is_pascal_hs512(const int cc, const ggml_tensor * K, const ggml_tensor * V) {
    return cc >= CC_PASCAL && cc < CC_VOLTA && K->ne[0] == 512 && V->ne[0] == 512;
}

static void ggml_cuda_flash_attn_ext_orig(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    if (dst->src[5]) {
        if (ggml_cuda_dsa_attn_ext(ctx, dst)) {
            return;
        }
    }

    ggml_tensor local_dst, Kl, Vl, Ml;
    if (n_swa > 0) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        local_dst = *dst;
        local_dst.op_params[4] = 0;
        if (first > 0) {
            local_dst = *dst;
            Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
            Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
            Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
            local_dst.src[1] = &Kl;
            local_dst.src[2] = &Vl;
            local_dst.src[3] = &Ml;
            local_dst.op_params[4] = 0;
            dst = &local_dst;
        }
        dst = &local_dst;
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (is_pascal_hs512(cc, K, V)) {
        ggml_cuda_flash_attn_ext_tile_new(ctx, dst);
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (is_pascal_mla_absorbed_decode(Q, K, V)) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            return;
        }
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8) {
                // P100 (sm_60/CC_PASCAL): the fp16 vec kernel accumulates the online-softmax
                // denominator and the P*V product in fp16, flipping ~3-4% of decode top-1 tokens
                // vs fp32 (llama.cpp#25593). Decode is bandwidth-bound on P100 so vec_f32 is free.
                if (cc == CC_PASCAL) {
                    ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                }
            } else if (Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            } else if (ctx.fabsum) {
                ggml_cuda_flash_attn_ext_tile_f16_fabsum(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    int gqa_ratio = Q->ne[2] / K->ne[2];

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (gqa_ratio == 12 || gqa_ratio == 6 || gqa_ratio == 10)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 && (gqa_ratio == 6 || gqa_ratio == 12)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 512 && V->ne[0] == 512) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];
    if (cc >= CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (is_pascal_hs512(cc, K, V)) {
        return ggml_cuda_fattn_tile_new_is_supported(ctx, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (is_pascal_mla_absorbed_decode(Q, K, V)) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        }
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8) {
                if (cc == CC_PASCAL) {
                    return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
                }
                return ggml_cuda_fattn_vec_f16_is_supported(ctx, dst);
            } else if (Q->ne[0] == 256) {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8) {
                return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (new_mma_available(cc) &&
            (Q->ne[0] == 576 || Q->ne[0] == 320 || Q->ne[0] == 512 || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (Q->ne[0] == 576 || Q->ne[0] == 512 || Q->ne[0] == 320) {
            int gqa_ratio = Q->ne[2]/K->ne[2];
            return (gqa_ratio % 4) == 0;
        }
        return true;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(ctx, dst);
}

// Decode with q8_0 cache and GQA: kernel that reads the cache once for all the heads of the group, without
// converting it to f16 (fattn-q8-gqa.cu). FA_Q8_GQA=0 turns it off; FA_Q8_CHECK=1 also computes the usual path
// and prints the relative difference (sentinel; turn off CUDA graphs, GGML_CUDA_DISABLE_GRAPHS=1).
#include "fattn-q8-prefill.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Prompt reading with q8_0 cache without converting it to f16 (FA_Q8_PREFILL=1): ~38 MB temporary instead of ~480 at
    // 120k (needed for the 3060's memory with MTP at long context), 25 times smaller error, but 0.7x the speed.
    static const bool q8_prefill = getenv("FA_Q8_PREFILL") != nullptr && atoi(getenv("FA_Q8_PREFILL")) != 0;
    if (q8_prefill && dst->src[0]->ne[1] > 4 && ggml_cuda_fattn_q8_prefill_supported(dst)) {
        ggml_cuda_fattn_q8_prefill(ctx, dst);
        return;
    }
    static const bool on    = getenv("FA_Q8_GQA") == nullptr || atoi(getenv("FA_Q8_GQA")) != 0;
    static const bool check = getenv("FA_Q8_CHECK") != nullptr;
    if (!on || !ggml_cuda_fattn_q8_gqa_supported(dst)) {
        ggml_cuda_flash_attn_ext_orig(ctx, dst);
        return;
    }
    // Diagnostics (FA_Q8_DUMP=<file>): saves Q, K, V, mask and parameters of the real calls (the first FA_Q8_DUMP_N
    // with at least FA_Q8_DUMP_FROM positions), to reproduce the kernel outside the engine on the model's data.
    static const char * dump = getenv("FA_Q8_DUMP");
    if (dump) {
        static int left = getenv("FA_Q8_DUMP_N") ? atoi(getenv("FA_Q8_DUMP_N")) : 32;
        static const int64_t from = getenv("FA_Q8_DUMP_FROM") ? atoll(getenv("FA_Q8_DUMP_FROM")) : 0;
        const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * M = dst->src[3];
        if (left > 0 && K->ne[1] >= from) {
            --left;
            FILE * f = fopen(dump, "ab");
            if (f) {
                float scale; memcpy(&scale, (const float *) dst->op_params, sizeof(float));
                const int64_t h[16] = { 0x46415138, K->ne[0], Q->ne[1], Q->ne[2], K->ne[2], K->ne[1],
                                        (int64_t) Q->nb[1], (int64_t) Q->nb[2], (int64_t) K->nb[1], (int64_t) K->nb[2],
                                        (int64_t) V->nb[1], (int64_t) V->nb[2], M ? (int64_t) M->nb[1] : 0,
                                        (int64_t) ggml_nbytes(Q), (int64_t) ggml_nbytes(K), M ? (int64_t) ggml_nbytes(M) : 0 };
                fwrite(h, sizeof(h), 1, f);
                fwrite(&scale, sizeof(float), 1, f);
                const ggml_tensor * ts[4] = { Q, K, V, M };
                const size_t ns[4] = { ggml_nbytes(Q), ggml_nbytes(K), ggml_nbytes(V), M ? ggml_nbytes(M) : 0 };
                std::vector<char> buf;
                for (int i = 0; i < 4; ++i) {
                    if (!ts[i]) continue;
                    buf.resize(ns[i]);
                    CUDA_CHECK(cudaMemcpyAsync(buf.data(), ts[i]->data, ns[i], cudaMemcpyDeviceToHost, ctx.stream()));
                    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                    fwrite(buf.data(), 1, ns[i], f);
                }
                fclose(f);
            }
        }
    }
    if (!check) {
        ggml_cuda_fattn_q8_gqa(ctx, dst);
        return;
    }
    const int64_t n = ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> mine(ctx.pool(), n);
    ggml_tensor tmp = *dst;
    tmp.data = mine.get();
    ggml_cuda_fattn_q8_gqa(ctx, &tmp);
    ggml_cuda_flash_attn_ext_orig(ctx, dst);
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaMemcpyAsync(a.data(), mine.get(), n * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(b.data(), dst->data, n * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += (double) b[i] * b[i]; }
    fprintf(stderr, "FA_Q8_CHECK n_kv %6lld heads %lld/%lld diff %.2e\n", (long long) dst->src[1]->ne[1],
            (long long) dst->src[0]->ne[2], (long long) dst->src[1]->ne[2], sqrt(num / (den + 1e-30)));
}
