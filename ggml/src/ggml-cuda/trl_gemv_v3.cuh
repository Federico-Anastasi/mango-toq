// trl_gemv_v3.cuh — trellis GEMV for 1..4 columns (tokens) on GPUs without tensor cores (Pascal sm_61, Turing TU116 sm_75).
// Replaces trl_gemv of ggml-cuda/trl.cu: same weight format, same int8 inputs in blocks of 32, same mul1 codebook,
// same result (exact int32 accumulation; differs from the current kernel only in the order of the float sums, ~1e-7).
//
// Interface: trl_args as in trl.cu (the ks field is ignored); the caller zeroes y first (trl_prep already does).
//   trl_gemv_v3<B, NC>(args, cc, nsm, stream) -> true if it launched the kernel, false if the GPU is Ampere (cc >= 800):
//   there the current kernel stays (not measured here).   Requirements: K % 256 == 0, N % 64 == 0, B = 2 or 3, NC = 1..4.
// Use inside trl.cu: define TRL_V3_HAVE_ARGS before the include (trl_args, trl_dp4a_us, TRL_QB already exist).
//
// What changes with respect to trl_gemv:
//  1. exact work distribution: the space (band of 64 rows) x (K chunk) is divided into equal CONTIGUOUS intervals, one per warp
//     (no K slices with ~1.5 items per warp: with whole items half of the warps stayed idle ~20% of the time); the output is always atomicAdd.
//  2. immediate-offset addresses: a chunk = 8 steps of one tile, the pointer advances once per chunk;
//     ring buffer of D steps (prefetch D-1 steps) instead of cur = nxt; previous word via shfl.
//  3. decode: the 14 high bits of (word << o) with a 1-operand rotation (SHF) when the window lies in one word,
//     2-operand funnel only for straddling windows; on Pascal native bfe. 4 instructions per weight (SHF, SHR, IMAD, IDP).
//  4. geometry per GPU and per NC (words per lane WPL, ring steps D, minimum blocks per SM MB) chosen on the bench.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "../ggml-trl.h"

#ifndef TRL_V3_HAVE_ARGS
#define TRL_QB 32
struct trl_args {
    const uint32_t * blob;
    const int      * aq;     // [NC][K/4] int8x4
    const float    * ad;     // [NC][K/32]
    const int      * asum;   // [NC][K/32]
    float          * y;      // [NC][N]  (to be zeroed first)
    int N, K, ks;
};
static __device__ __forceinline__ int trl_dp4a_us(uint32_t a, int b, int c) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define TRL_V3_LD "ld.global.nc.L1::no_allocate."
#else
#define TRL_V3_LD "ld.global.nc."
#endif

static __device__ __forceinline__ uint4 trl3_ld4(const uint32_t * p) {
    uint4 v;
    asm volatile(TRL_V3_LD "v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint2 trl3_ld2(const uint32_t * p) {
    uint2 v;
    asm volatile(TRL_V3_LD "v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint32_t trl3_ld1(const uint32_t * p) {
    return __ldcs(p);
}

// B bits per symbol; WPL uint32 words per lane (32*WPL/B symbols); D ring steps; NC columns; MB minimum blocks per SM.
template <int B, int WPL, int D, int NC, int MB>
static __global__ void __launch_bounds__(256, MB) trl_gemv_v3_kernel(trl_args a) {
    constexpr int RUN = 8;                      // 8 tiles = 32 columns = one activation block
    constexpr int SPL = 32 * WPL / B;           // symbols per lane
    constexpr int RPL = SPL / 4;                // rows per lane
    constexpr int LPT = 64 / RPL;               // lanes per tile
    constexpr int TPS = 32 / LPT;               // tiles per warp step
    constexpr int TW = 8 * B;                   // words per tile
    static_assert((32 * WPL) % B == 0 && SPL % 4 == 0 && LPT * WPL == 8 * B, "geometry");
    static_assert(RUN % D == 0, "D divide RUN");

    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nw = gridDim.x * (blockDim.x >> 5);
    const int g = lane / LPT, li = lane % LPT;

    const int ntiles = a.K >> 2;
    const int nrg = a.N >> 6;
    const size_t grpWords = (size_t)(64 + (size_t)ntiles * TW);
    constexpr int chunkTiles = TPS * RUN;
    const int nchunks = ntiles / chunkTiles;
    const int qstr = a.K >> 2, bstr = a.K / TRL_QB;

    struct Buf { uint32_t w[WPL]; int a4[NC]; };

    // contiguous interval of chunks of this warp in the (band, chunk) space
    long long lo = (long long) nrg * nchunks * gw / nw;
    const long long hi = (long long) nrg * nchunks * (gw + 1) / nw;

    while (lo < hi) {
        const int rg = (int)(lo / nchunks);
        const int c0 = (int)(lo - (long long) rg * nchunks);
        const int c1 = (int) min((long long) nchunks, (long long) c0 + (hi - lo));
        lo += c1 - c0;

        const uint32_t * gb = a.blob + (size_t)rg * grpWords;
        const uint32_t * tw = gb + 64;

        float acc[RPL][NC];
        int acci[RPL][NC];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                acc[r][c] = 0.f;
                acci[r][c] = 0;
            }
        }

        // pointers of the current chunk: step j is at a + j*TW words (immediate offset)
        const uint32_t * pc = tw + (size_t) c0 * chunkTiles * TW + (size_t) g * RUN * TW + li * WPL;
        const int * qc = a.aq + (size_t) c0 * chunkTiles + g * RUN;

        auto load = [&](Buf & bf, const uint32_t * p, const int * q, int j) {
            const uint32_t * pj = p + (size_t) j * TW;
            if constexpr (WPL % 4 == 0) {
#pragma unroll
                for (int k = 0; k < WPL; k += 4) {
                    const uint4 v = trl3_ld4(pj + k);
                    bf.w[k] = v.x; bf.w[k + 1] = v.y; bf.w[k + 2] = v.z; bf.w[k + 3] = v.w;
                }
            } else if constexpr (WPL % 2 == 0) {
#pragma unroll
                for (int k = 0; k < WPL; k += 2) {
                    const uint2 v = trl3_ld2(pj + k);
                    bf.w[k] = v.x; bf.w[k + 1] = v.y;
                }
            } else {
#pragma unroll
                for (int k = 0; k < WPL; ++k) bf.w[k] = trl3_ld1(pj + k);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                bf.a4[c] = __ldg(q + c * qstr + j);
            }
        };

        auto compute = [&](const Buf & bf) {
            uint32_t S[WPL + 1];
#pragma unroll
            for (int k = 0; k < WPL; ++k) S[k + 1] = bf.w[k];
            const uint32_t pv = __shfl_up_sync(0xffffffffu, S[WPL], 1, LPT);
            S[0] = (li == 0) ? 0u : pv;                  // the state restarts from 0 at every tile
            int rep[NC][4];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int a4 = bf.a4[c];
                rep[c][0] = __byte_perm(a4, 0, 0x0000);
                rep[c][1] = __byte_perm(a4, 0, 0x1111);
                rep[c][2] = __byte_perm(a4, 0, 0x2222);
                rep[c][3] = __byte_perm(a4, 0, 0x3333);
            }
#pragma unroll
            for (int t = 0; t < SPL; ++t) {
                const int E = 32 + B * (t + 1);          // end of the window, in bits from the start of S[0]
                const int st = E - TRL_L;
                const int wi = st >> 5, o = st & 31;
                uint32_t state;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
                if (o <= 18) {
                    asm("bfe.u32 %0, %1, %2, 14;" : "=r"(state) : "r"(S[wi]), "r"(18 - o));
                } else {
                    state = __funnelshift_l(S[wi + 1], S[wi], o) >> 18;
                }
#else
                if (o > 18) {
                    state = __funnelshift_l(S[wi + 1], S[wi], o) >> 18;      // window straddling two words
                } else {
                    state = __funnelshift_l(S[wi], S[wi], o) >> 18;          // rotation: the 14 high bits all come from S[wi]
                }
#endif
                const uint32_t wv = state * TRL_MUL1_K;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acci[t >> 2][c] = trl_dp4a_us(wv, rep[c][t & 3], acci[t >> 2][c]);
                }
            }
        };

        // ring of D buffers: step j of the chunk uses ring[j % D]; step j + D - 1 is prefetched (also in the next chunk)
        Buf ring[D];
#pragma unroll
        for (int j = 0; j < D - 1; ++j) load(ring[j], pc, qc, j);

        for (int ch = c0; ch < c1; ++ch) {
            const uint32_t * pn = pc + (size_t) chunkTiles * TW;
            const int * qn = qc + chunkTiles;
            const bool more = (ch + 1 < c1);
            const int blk = ch * chunkTiles / 8 + g;
            float dA[NC]; int sA[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                dA[c] = __ldg(a.ad + c * bstr + blk);
                sA[c] = __ldg(a.asum + c * bstr + blk);
            }
#pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int jj = j + D - 1;
                if (jj < RUN) {
                    load(ring[(j + D - 1) % D], pc, qc, jj);
                } else if (more) {
                    load(ring[(j + D - 1) % D], pn, qn, jj - RUN);
                }
                compute(ring[j % D]);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    acc[r][c] = fmaf(dA[c], (float)(acci[r][c] - 510 * sA[c]), acc[r][c]);
                    acci[r][c] = 0;
                }
            }
            pc = pn; qc = qn;
        }

#pragma unroll
        for (int off = LPT; off < 32; off <<= 1) {
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[r][c] += __shfl_xor_sync(0xffffffffu, acc[r][c], off);
                }
            }
        }
        if (lane < LPT) {
            const float * sc = reinterpret_cast<const float *>(gb);
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int row = rg * 64 + lane * RPL + r;
                const float scr = sc[lane * RPL + r] * (1.0f / TRL_MUL1_STD);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    atomicAdd(a.y + (size_t) c * a.N + row, acc[r][c] * scr);
                }
            }
        }
    }
}

// Geometries chosen on the bench (GB/s of weights): {WPL, D, MB} per (GPU, B, NC).
template <int B, int WPL, int D, int NC, int MB>
static bool trl3_run(const trl_args & a, int nsm, cudaStream_t stream) {
    static int occ_dev[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    int & occ = occ_dev[dev & 15];
    if (occ == 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, trl_gemv_v3_kernel<B, WPL, D, NC, MB>, 256, 0);
        occ = occ < 1 ? 1 : occ;
    }
    trl_gemv_v3_kernel<B, WPL, D, NC, MB><<<nsm * occ, 256, 0, stream>>>(a);
    return true;
}

template <int B, int NC>
static bool trl_gemv_v3(const trl_args & a, int cc, int nsm, cudaStream_t stream) {
    static_assert(B == 2 || B == 3, "B");
    if (cc >= 800) {
        return false;
    }
    if (cc >= 700) {          // Turing (TU116)
        if constexpr (B == 2) {
            if constexpr (NC == 1) return trl3_run<2, 2, 4, 1, 4>(a, nsm, stream);
            if constexpr (NC == 2) return trl3_run<2, 1, 2, 2, 4>(a, nsm, stream);
            if constexpr (NC == 3) return trl3_run<2, 1, 2, 3, 3>(a, nsm, stream);
            if constexpr (NC == 4) return trl3_run<2, 2, 2, 4, 2>(a, nsm, stream);
        } else {
            if constexpr (NC <= 2) return trl3_run<3, 3, 4, NC, 3>(a, nsm, stream);
            if constexpr (NC == 3) return trl3_run<3, 3, 4, NC, 2>(a, nsm, stream);
            if constexpr (NC == 4) return trl3_run<3, 3, 2, NC, 2>(a, nsm, stream);
        }
    } else {                  // Pascal
        if constexpr (B == 2) {
            if constexpr (NC == 1) return trl3_run<2, 1, 4, 1, 4>(a, nsm, stream);
            if constexpr (NC == 2) return trl3_run<2, 1, 4, 2, 3>(a, nsm, stream);
            if constexpr (NC >= 3) return trl3_run<2, 1, 2, NC, 2>(a, nsm, stream);
        } else {
            // 3 bit on Pascal: wins only with NC <= 2 (NC >= 3 equal or worse than the current kernel: that one stays)
            if constexpr (NC == 1) return trl3_run<3, 3, 4, 1, 3>(a, nsm, stream);
            if constexpr (NC == 2) return trl3_run<3, 3, 4, 2, 2>(a, nsm, stream);
        }
    }
    return false;
}
