// trl_mmq3: trellis GEMM (mul1 codebook) with the int8 tensor units for prompt reading on Ampere.
// Used by trl.cu.
//
// y[m][n] = s_n * 4/std * dA[m] * sum_k h[n][k] * a[m][k],  h = sat8((q + 2) >> 2), q = bytesum(state*K) - 510
// Int8 activations with ONE scale per token over the whole K (dA[m]): exact int32 accumulation to the end
// (|sum| <= K * 127 * 127 < 2^31 for K <= 133k), no conversion in the loop.
//
// Block = 128 rows (2 bands of 64) x BN = 32*WT tokens, NW = 2*WT warps (2 along the rows x WT along the tokens,
// each 64 rows x 32 tokens = 4 x 4 mma m16n8k32). Stage = 64 values of k.
//   - int8 activations and raw weight words in smem with cp.async, NS stages in flight;
//   - the weights of stage s+1 are decoded (int8, layout for ldmatrix) between the mma of stage s;
//   - one __syncthreads per stage; precomputed addresses, no branches in the loop.
// Raw in smem: each tile has 4 zero words in front (TS = 8B + 4): a state that starts before the tile reads the leading zero,
// one that ends beyond reads the leading zero of the next tile (or of the tail). The zeros are written
// once at the start; cp.async writes only the data.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace trl3 {

static __device__ __forceinline__ void cp16(uint32_t dst, const void * src, int sz) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz));
#endif
}
static __device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
static __device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
static __device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
}
static __device__ __forceinline__ void mma(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
// row of 64 bytes (4 pieces of 16): piece shuffled with (row >> 1) & 3, ldmatrix on 8 rows without conflicts
static __device__ __forceinline__ int swz(int r, int c) { return r * 16 + ((c ^ ((r >> 1) & 3)) << 2); }

template <int B, int WT> struct cfg {
    static constexpr int NW   = 2 * WT;                    // warps
    static constexpr int NT   = 32 * NW;                   // threads
    static constexpr int BN   = 32 * WT;                   // tokens per block
    static constexpr int TS   = 8 * B + 4;                 // words per tile in smem: 4 zeros + 8B data
    static constexpr int RAW  = 2 * 16 * TS + 4;           // raw words per stage (2 bands x 16 tiles) + tail
    static constexpr int RAWC = 2 * 16 * 2 * B;            // 16-byte pieces of raw data per stage
    static constexpr int ACT  = BN * 16;                   // activations per stage: BN tokens x 64 bytes
    static constexpr int SLOT = RAW + ACT;
};

template <int B, int WT, int NS>
static constexpr size_t smem_bytes() { return (size_t) (NS * cfg<B, WT>::SLOT + 2 * 128 * 16) * 4; }

struct yes { static constexpr bool value = true; };
struct no  { static constexpr bool value = false; };

// ABL (bench only): 0 full, 1 without decode, 2 without mma, 3 without cp.async (after the prologue)
template <int B, int WT, int NS, int MINB = 1, int ABL = 0>
static __global__ void __launch_bounds__(64 * WT, MINB) mmq(const uint32_t * __restrict__ blob, const int8_t * __restrict__ aq,
                                                             const float * __restrict__ ad, float * __restrict__ y,
                                                             int N, int K, int M) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = cfg<B, WT>;
    static_assert(NS >= 3, "at least 3 stages are needed");
    static_assert(C::RAWC <= C::NT, "at most one raw piece per thread");
    static_assert(WT == 4 || WT == 8, "8 or 16 warps");
    extern __shared__ __align__(16) uint32_t sm[];
    uint32_t * As = sm + NS * C::SLOT;
    const uint32_t sm0 = (uint32_t) __cvta_generic_to_shared(sm);
    const uint32_t as0 = (uint32_t) __cvta_generic_to_shared(As);

    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, tig = lane & 3;
    const int wm = w / WT, wt = w % WT;
    const int nb = N >> 6;
    const int rbk = blockIdx.y;                            // grid: x = blocks of tokens (adjacent: weights read once)
    const int b0 = min(2 * rbk, nb - 1), b1 = min(2 * rbk + 1, nb - 1);
    const size_t bandWords = 64 + (size_t) (K >> 2) * 8 * B;
    const int m0 = blockIdx.x * C::BN;
    const int nst = K >> 6;

    // leading and tail zeros of the tiles, in every slot (cp.async does not touch them)
    for (int i = tid; i < NS * 33; i += C::NT) {
        const int slot = i / 33, t = i % 33;
        *reinterpret_cast<uint4 *>(sm + slot * C::SLOT + t * C::TS) = make_uint4(0, 0, 0, 0);
    }

    // asynchronous copies: fixed addresses per thread, they advance one stage per call
    const bool raw_on = tid < C::RAWC;
    const uint32_t * raw_src;
    uint32_t raw_dst;
    {
        const int band = tid / (32 * B), rem = tid % (32 * B), tile = rem / (2 * B), ch = rem % (2 * B);
        raw_src = blob + (size_t) (band ? b1 : b0) * bandWords + 64 + (size_t) tile * (8 * B) + ch * 4;
        raw_dst = (uint32_t) ((band * 16 + tile) * C::TS + 4 + ch * 4) * 4;
    }
    const int8_t * act_src[2];
    uint32_t act_dst[2];
    int act_sz[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int i = tid + u * C::NT, tok = i >> 2, c = i & 3, gt = m0 + tok;
        const bool ok = gt < M;
        act_src[u] = aq + (size_t) (ok ? gt : 0) * K + c * 16;
        act_dst[u] = (uint32_t) (C::RAW + swz(tok, c)) * 4;
        act_sz[u] = ok ? 16 : 0;
    }
    auto load_stage = [&](int s, int slot) {
        const uint32_t base = sm0 + (uint32_t) (slot * C::SLOT) * 4;
        if (raw_on) cp16(base + raw_dst, raw_src + (size_t) s * (16 * 8 * B), 16);
#pragma unroll
        for (int u = 0; u < 2; ++u) cp16(base + act_dst[u], act_src[u] + s * 64, act_sz[u]);
    };

    // decode: thread = row rb = 8*(w % 8) + g of each band it handles, tile 4*piece + tig.
    // 8 warps: each warp does the same row in the 2 bands (8 quartets); 16 warps: one band (w / 8), 4 quartets.
    constexpr int RGI = 16 / C::NW;
    constexpr int NQ = 4 * RGI;
    const int rb = 8 * (w % 8) + g;
    const int start0 = (4 * rb + 1) * B - 14;
    const int i0 = start0 >> 5;                            // -1 if the state starts before the tile: it reads the zero
    const int sr = 64 - 14 - ((start0 - 32 * i0) + 3 * B);
    const int rd_off = tig * C::TS + 4 + i0;
    const int xr = (rb >> 1) & 3;
    const int band_w = w / 8;
    auto decode_q = [&](int slot, int abuf, int qd) {
        const int pz = qd & 3;
        const int band = RGI == 2 ? (qd >> 2) : band_w;
        const uint32_t * tp = sm + slot * C::SLOT + (band * 16 + 4 * pz) * C::TS + rd_off;
        const uint32_t hi = tp[0], lo = tp[1];
        const uint32_t u = (uint32_t) ((((uint64_t) hi << 32) | lo) >> sr);
        int h[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t wv = ((u >> ((3 - c) * B)) & 0x3FFFu) * 0x83DCD12Du;
            int bs;
            asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(bs) : "r"(wv), "r"(0x01010101), "r"(-508));
            h[c] = bs >> 2;
        }
        uint32_t hi2, v;
        asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;" : "=r"(hi2) : "r"(h[3]), "r"(h[2]), "r"(0));
        asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(h[1]), "r"(h[0]), "r"(hi2));
        As[abuf * 2048 + (band * 64 + rb) * 16 + ((pz ^ xr) << 2) + tig] = v;
    };

    int c[4][4][4];
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) c[f][j][i] = 0;

    const int lmA_r = (lane & 7) + (((lane >> 3) & 1) << 3), lmA_c = lane >> 4;
    const int lmB_t = (lane & 7) + ((lane >> 4) << 3), lmB_c = (lane >> 3) & 1;
    uint32_t offB[2][2], offA[2][4];                       // ldmatrix offsets in bytes, for ks = 0, 1
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
        for (int p = 0; p < 2; ++p) offB[ks][p] = (uint32_t) (C::RAW + swz(wt * 32 + 16 * p + lmB_t, 2 * ks + lmB_c)) * 4;
#pragma unroll
        for (int f = 0; f < 4; ++f) offA[ks][f] = (uint32_t) swz(wm * 64 + 16 * f + lmA_r, 2 * ks + lmA_c) * 4;
    }

#pragma unroll
    for (int s = 0; s < NS - 1; ++s) {
        if (s < nst) load_stage(s, s);
        commit();
    }
    wait<NS - 2>();
    __syncthreads();
#pragma unroll
    for (int qd = 0; qd < NQ; ++qd) decode_q(0, 0, qd);

    auto stage = [&](int s, auto dec) {
        wait<NS - 3>();
        __syncthreads();
        if (ABL != 3 && s + NS - 1 < nst) load_stage(s + NS - 1, (s + NS - 1) % NS);
        commit();
        const int slot = s % NS, abuf = s & 1;
        const uint32_t actb = sm0 + (uint32_t) (slot * C::SLOT) * 4;
        const uint32_t ab = as0 + (uint32_t) (abuf * 2048) * 4;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t bf[4][2];
#pragma unroll
            for (int p = 0; p < 2; ++p)
                ldsm4(actb + offB[ks][p], bf[2 * p][0], bf[2 * p][1], bf[2 * p + 1][0], bf[2 * p + 1][1]);
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                uint32_t a0, a1, a2, a3;
                ldsm4(ab + offA[ks][f], a0, a1, a2, a3);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if constexpr (ABL == 2) { c[f][j][0] += (int) (a0 ^ bf[j][0]); c[f][j][3] += (int) (a3 ^ bf[j][1]); }
                    else mma(c[f][j], a0, a1, a2, a3, bf[j][0], bf[j][1]);
                }
                // ALU while the mma work: the NQ quartets of the next stage, spread over the 8 groups (ks, f)
                if constexpr (decltype(dec)::value && ABL != 1) {
                    const int grp = 4 * ks + f;
                    if constexpr (NQ == 8) decode_q((s + 1) % NS, (s + 1) & 1, grp);
                    else if ((grp & 1) == 0) decode_q((s + 1) % NS, (s + 1) & 1, grp >> 1);
                }
            }
        }
    };
    for (int s = 0; s < nst - 1; ++s) stage(s, yes{});
    stage(nst - 1, no{});
    wait<0>();

    if (2 * rbk + wm >= nb) return;
    float dA[4][2];
    int tokc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        tokc[j] = m0 + wt * 32 + 8 * j + 2 * tig;
        dA[j][0] = tokc[j] < M ? __ldg(ad + tokc[j]) : 0.f;
        dA[j][1] = tokc[j] + 1 < M ? __ldg(ad + tokc[j] + 1) : 0.f;
    }
    const int band = wm ? b1 : b0;
    const float * sc = reinterpret_cast<const float *>(blob + (size_t) band * bandWords);
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        const int r0 = 16 * f + g, r1 = r0 + 8;
        const float s0 = sc[r0] * (4.0f / 147.80054127099805f), s1 = sc[r1] * (4.0f / 147.80054127099805f);
        const int row0 = band * 64 + r0, row1 = band * 64 + r1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n0 = tokc[j], n1 = n0 + 1;
            if (n0 < M) { y[(size_t) n0 * N + row0] = dA[j][0] * (float) c[f][j][0] * s0; y[(size_t) n0 * N + row1] = dA[j][0] * (float) c[f][j][2] * s1; }
            if (n1 < M) { y[(size_t) n1 * N + row0] = dA[j][1] * (float) c[f][j][1] * s0; y[(size_t) n1 * N + row1] = dA[j][1] * (float) c[f][j][3] * s1; }
        }
    }
#else
    (void) blob; (void) aq; (void) ad; (void) y; (void) N; (void) K; (void) M;
#endif
}

} // namespace trl3
