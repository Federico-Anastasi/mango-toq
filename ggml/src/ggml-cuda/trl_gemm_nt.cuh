// Many-column trellis product (prompt reading) for GPUs WITHOUT int8/bf16 tensor cores:
// GTX 1080 / Pascal (sm_61) and GTX 1660 (TU116, sm_75). Three FUSED kernels: the trellis weights (mul1 codebook) are decoded
// in shared memory and the weights never pass through global memory (no slice-wise reconstruction + cuBLAS).
//
//   TRL_NT_F32 : q as float, fp32 activations, FFMA.            Exact (error ~1e-6). The best on Pascal without int8.
//   TRL_NT_H2  : q as half2 (exact), fp16 activations scaled per column, HFMA2 with fp16 accumulation over windows
//                of 32 values of k and carry into fp32. Only with fast fp16 (Turing). Error ~9e-4 (< bf16: 2.3e-3).
//   TRL_NT_I8  : weights as u16 in pairs of k, int8 activations in blocks of ablk values with one scale, dp2a with
//                exact int32 accumulation. The activations are APPROXIMATE (same quantization as the GEMV): error
//                ~5e-3 (ablk 32) .. 7e-3 (ablk 128).
//
// The weight is W[n,k] = s_n * q / 147.80..., q = bytesum(state * 0x83DCD12D) - 510 in [-510, 510]; the row scale
// s_n is applied on output, the inner product is between q (exact) and the activation.
// Input x32: already rotated activations (trl_prep<2,float>), contiguous [M][K]. Output y[col * N + row].
// Requirements as ggml_cuda_trl_mm_supported: K % 256 == 0, N % 64 == 0. Any M (>= 1), any N multiple of 64.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#if __has_include("../ggml-trl.h")
#include "../ggml-trl.h"      // inside ik_llama.cpp/ggml/src/ggml-cuda
#else
#include "ggml-trl.h"         // bench: -I ggml/src
#endif

#define TRL_NT_F32 0
#define TRL_NT_H2  1
#define TRL_NT_I8  2

// ---------------------------------------------------------------------------------------------------------
// Decoding of the 4 symbols of a row in a tile (cf. trl_state in ggml-trl.h).
//   Symbol t = 4r + c has its state in the 14-bit window ending at bit (t+1)*B of the tile.
//   The window of the 4 symbols [(4r+1)*B - 14, (4r+4)*B) lies in two consecutive words (i0, i0+1); i0 can be -1
//   (zeros before the tile). A single 64-bit shift, then cuts to 14 bits with constant shifts.
// ---------------------------------------------------------------------------------------------------------
template <int B>
struct trl_nt_dec4 {
    int i0, sr;
    __device__ __forceinline__ void init(int r) {
        const int start0 = (4 * r + 1) * B - TRL_L;
        i0 = start0 >> 5;                                        // arithmetic shift: -1 if negative
        sr = 64 - TRL_L - ((start0 - 32 * i0) + 3 * B);           // >= 10
    }
    __device__ __forceinline__ void load(const uint32_t * tile, uint32_t & hi, uint32_t & lo) const {
        hi = i0 >= 0 ? __ldg(tile + i0) : 0u;
        lo = (i0 + 1 < 8 * B) ? __ldg(tile + i0 + 1) : 0u;
    }
    // byte sums S_t = bytesum(state_t * K) in [0, 1020], t = 0..3; the integer weight is q = S - 510
    __device__ __forceinline__ void sums(uint32_t hi, uint32_t lo, int S[4]) const {
        const uint32_t u = (uint32_t) ((((uint64_t) hi << 32) | lo) >> sr);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t st = (u >> ((3 - t) * B)) & TRL_MASK;
            S[t] = (int) __dp4a(st * TRL_MUL1_K, 0x01010101u, 0u);
        }
    }
};

static __device__ __forceinline__ float trl_nt_s2q(int S) {      // S - 510 as float without cvt (S < 2^22)
    return __int_as_float(0x4B400000 | S) - 12583422.0f;
}

// ---------------------------------------------------------------------------------------------------------
// F32: tile 128 rows x 128 columns, BK = 16, 256 threads (16x16 of 8x8 tiles), double buffer in smem.
// ---------------------------------------------------------------------------------------------------------
template <int B>
static __global__ void __launch_bounds__(256) trl_nt_gemm_f32(const uint32_t * __restrict__ blob, const float * __restrict__ x,
                                                              float * __restrict__ y, int N, int K, int M) {
    constexpr int BN = 128, BM = 128, BK = 16;
    __shared__ __align__(16) float Ws[2][BK][BN];
    __shared__ __align__(16) float Xs[2][BK][BM];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nbands = N >> 6;
    const size_t bandWords = 64 + (size_t) (K >> 2) * 8 * B;

    // decode: row drow, tile dj0 and dj0 + 2 of each step of 4 tiles (the band beyond N, if any, repeats the last one)
    const int drow = tid & 127, dj0 = tid >> 7;
    const uint32_t * bandp = blob + (size_t) min((n0 + drow) >> 6, nbands - 1) * bandWords;
    trl_nt_dec4<B> dec; dec.init(drow & 63);
    const uint32_t * tiles = bandp + 64;
    // activations: column xc (and xc + 64), quarter of k xq; the columns beyond M repeat the last one (the output discards them)
    const int xc = tid >> 2, xq = tid & 3;
    const float * xp0 = x + (size_t) min(m0 + xc, M - 1) * K + xq * 4;
    const float * xp1 = x + (size_t) min(m0 + xc + 64, M - 1) * K + xq * 4;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

    uint32_t wh[2], wl[2];
    float4 xv[2];
    auto gload = [&](int kt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) dec.load(tiles + (size_t) (kt * 4 + dj0 + 2 * h) * (8 * B), wh[h], wl[h]);
        xv[0] = __ldg(reinterpret_cast<const float4 *>(xp0 + kt * BK));
        xv[1] = __ldg(reinterpret_cast<const float4 *>(xp1 + kt * BK));
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int S[4];
            dec.sums(wh[h], wl[h], S);
            const int kb = (dj0 + 2 * h) * 4;
#pragma unroll
            for (int t = 0; t < 4; ++t) Ws[buf][kb + t][drow] = trl_nt_s2q(S[t]);
            const int c = xc + 64 * h;
            Xs[buf][xq * 4 + 0][c] = xv[h].x;
            Xs[buf][xq * 4 + 1][c] = xv[h].y;
            Xs[buf][xq * 4 + 2][c] = xv[h].z;
            Xs[buf][xq * 4 + 3][c] = xv[h].w;
        }
    };
    const int nk = K / BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int cur = kt & 1;
        if (kt + 1 < nk) gload(kt + 1);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4 *>(&Ws[cur][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&Ws[cur][kk][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&Xs[cur][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&Xs[cur][kk][64 + tx * 4]);
            const float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            const float b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (kt + 1 < nk) sstore(cur ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int rl = hi * 64 + ty * 4;
        if (n0 + rl >= N) continue;
        const float * sc = reinterpret_cast<const float *>(blob + (size_t) ((n0 + rl) >> 6) * bandWords);
        float s[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = sc[(rl & 63) + i] * (1.0f / TRL_MUL1_STD);
#pragma unroll
        for (int hj = 0; hj < 2; ++hj) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = m0 + hj * 64 + tx * 4 + j;
                if (col < M) {
                    *reinterpret_cast<float4 *>(y + (size_t) col * N + n0 + rl) =
                        make_float4(acc[hi * 4 + 0][hj * 4 + j] * s[0], acc[hi * 4 + 1][hj * 4 + j] * s[1],
                                    acc[hi * 4 + 2][hj * 4 + j] * s[2], acc[hi * 4 + 3][hj * 4 + j] * s[3]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// H2 (Turing: HFMA2 at the FFMA rate, 2 MACs per instruction). Tile 128x128, BK = 16, thread 8 rows x 8 columns.
//   Weights: Ws[k][row pair] = half2 {q(2p), q(2p+1)} -> HFMA2 with .H0_H0/.H1_H1 selector.
//   q exact in half: (0x6400 | S) is 1024 + S, minus 1534.
//   Activations: fp16 scaled per column to |x|max = 8, layout [k][column pair] (half2), direct copy into smem.
//   fp16 accumulation over WIN steps of 16 (WIN * 16 values of k), then carry into fp32.
// ---------------------------------------------------------------------------------------------------------
// one block per column: amax, then x * 8/amax into the [k][Mp] layout (half precisions); cs = amax/8 (scale to put back)
static __global__ void trl_nt_scale_cols_f16(const float * x, __half * xh, float * cs, int K, int Mp) {
    __shared__ float red[32];
    const float * xc = x + (size_t) blockIdx.x * K;
    float am = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) am = fmaxf(am, fabsf(xc[k]));
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x < 32) {
        am = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        if (threadIdx.x == 0) red[0] = am;
    }
    __syncthreads();
    am = red[0];
    const float sc = am > 0.f ? 8.0f / am : 1.f;
    if (threadIdx.x == 0) cs[blockIdx.x] = am > 0.f ? am / 8.0f : 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) xh[(size_t) k * Mp + blockIdx.x] = __float2half(xc[k] * sc);
}

template <int B, int WIN>
static __global__ void __launch_bounds__(256, 2) trl_nt_gemm_h2(const uint32_t * __restrict__ blob, const __half2 * __restrict__ xh,
                                                             const float * __restrict__ cs, float * __restrict__ y,
                                                             int N, int K, int M, int Mp) {
    constexpr int BN = 128, BM = 128, BK = 16;
    __shared__ __align__(16) __half2 Ws[2][BK][BN / 2];
    __shared__ __align__(16) __half2 Xs[2][BK][BM / 2];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nbands = N >> 6;
    const size_t bandWords = 64 + (size_t) (K >> 2) * 8 * B;

    // decode: row pair dp = tid & 63 (rows 2dp, 2dp+1), tile dj = tid >> 6 of each step of 4 tiles
    const int dp = tid & 63, dj = tid >> 6;
    const uint32_t * bandp = blob + (size_t) min((n0 + 2 * dp) >> 6, nbands - 1) * bandWords;
    trl_nt_dec4<B> dec0, dec1; dec0.init((2 * dp) & 63); dec1.init((2 * dp + 1) & 63);
    const uint32_t * tiles = bandp + 64;
    // activations: row of k xk = tid >> 4, 16-byte piece (4 column pairs) xq = tid & 15
    const int xk = tid >> 4, xq = tid & 15;
    const uint4 * xp = reinterpret_cast<const uint4 *>(xh) + ((size_t) xk * (Mp / 2) + (m0 >> 1) + xq * 4) / 4;

    float acc[8][8];
    __half2 hacc[8][4];
    const __half2 z2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) hacc[i][j] = z2;

    uint32_t h0, l0, h1, l1;
    uint4 xv;
    auto gload = [&](int kt) {
        const uint32_t * tp = tiles + (size_t) (kt * 4 + dj) * (8 * B);
        dec0.load(tp, h0, l0);
        dec1.load(tp, h1, l1);
        xv = __ldg(xp + (size_t) kt * BK * (Mp / 2) / 4);
    };
    auto sstore = [&](int buf) {
        int S0[4], S1[4];
        dec0.sums(h0, l0, S0);
        dec1.sums(h1, l1, S1);
        const __half2 off = __halves2half2(__ushort_as_half(0x65FE), __ushort_as_half(0x65FE));   // 1024 + 510
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t w = __byte_perm(S0[t], S1[t], 0x5410) | 0x64006400u;                    // {1024 + S0, 1024 + S1}
            Ws[buf][dj * 4 + t][dp] = __hsub2(*reinterpret_cast<const __half2 *>(&w), off);
        }
        *reinterpret_cast<uint4 *>(&Xs[buf][xk][xq * 4]) = xv;
    };
    const int nk = K / BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int cur = kt & 1;
        if (kt + 1 < nk) gload(kt + 1);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const uint4 a = *reinterpret_cast<const uint4 *>(&Ws[cur][kk][ty * 4]);
            const uint4 b = *reinterpret_cast<const uint4 *>(&Xs[cur][kk][tx * 4]);
            const __half2 aw[4] = { *reinterpret_cast<const __half2 *>(&a.x), *reinterpret_cast<const __half2 *>(&a.y),
                                    *reinterpret_cast<const __half2 *>(&a.z), *reinterpret_cast<const __half2 *>(&a.w) };
            const __half2 bb[4] = { *reinterpret_cast<const __half2 *>(&b.x), *reinterpret_cast<const __half2 *>(&b.y),
                                    *reinterpret_cast<const __half2 *>(&b.z), *reinterpret_cast<const __half2 *>(&b.w) };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __half2 wl = __low2half2(aw[i]), wh = __high2half2(aw[i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    hacc[2 * i][j]     = __hfma2(wl, bb[j], hacc[2 * i][j]);
                    hacc[2 * i + 1][j] = __hfma2(wh, bb[j], hacc[2 * i + 1][j]);
                }
            }
        }
        if ((kt + 1) % WIN == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(hacc[i][j]);
                    acc[i][2 * j] += f.x;
                    acc[i][2 * j + 1] += f.y;
                    hacc[i][j] = z2;
                }
        }
        if (kt + 1 < nk) sstore(cur ^ 1);
        __syncthreads();
    }
    // output: rows ty*8 + i, columns tx*8 + j; row and column scale
    const int rl = ty * 8;
    if (n0 + rl >= N) return;
    const float * sc = reinterpret_cast<const float *>(blob + (size_t) ((n0 + rl) >> 6) * bandWords);
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = sc[(rl & 63) + i] * (1.0f / TRL_MUL1_STD);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = m0 + tx * 8 + j;
        if (col >= M) continue;
        const float cc = cs[col];
        float4 o0 = make_float4(acc[0][j] * s[0] * cc, acc[1][j] * s[1] * cc, acc[2][j] * s[2] * cc, acc[3][j] * s[3] * cc);
        float4 o1 = make_float4(acc[4][j] * s[4] * cc, acc[5][j] * s[5] * cc, acc[6][j] * s[6] * cc, acc[7][j] * s[7] * cc);
        *reinterpret_cast<float4 *>(y + (size_t) col * N + n0 + rl) = o0;
        *reinterpret_cast<float4 *>(y + (size_t) col * N + n0 + rl + 4) = o1;
    }
}

// ---------------------------------------------------------------------------------------------------------
// I8: weights as u16 (S = bytesum in [0,1020], NOT centered) in pairs of k, int8 activations in blocks of ABLK values
//   with one scale, product with dp2a (2 MACs per instruction, exact int32 accumulation).
//   dp2a.lo(w01, a) = S0*a0 + S1*a1, dp2a.hi(w23, a) = S2*a2 + S3*a3, with a = 4 consecutive int8 activations in k.
//   Every ABLK values of k: facc += dA * (acc - 510 * sA), sA = sum of the block's activations.
//   Layout of the quantized activations (transposed for coalesced reads): aqt[(k/16 * Mp + m) * 16 + k%16];
//   dA, sA: [block][Mp].
// ---------------------------------------------------------------------------------------------------------
static __device__ __forceinline__ int trl_nt_dp2a_lo(uint32_t a, uint32_t b, int c) {
    int r; asm("dp2a.lo.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
static __device__ __forceinline__ int trl_nt_dp2a_hi(uint32_t a, uint32_t b, int c) {
    int r; asm("dp2a.hi.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}

// one warp per (column, block of ablk values)
static __global__ void trl_nt_quant_act(const float * x, int8_t * aqt, float * dA, int * sA, int K, int M, int Mp, int ablk) {
    const int lane = threadIdx.x & 31;
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nb = K / ablk;
    if (w >= (int64_t) M * nb) return;
    const int m = (int) (w / nb), blk = (int) (w % nb);
    const float * xc = x + (size_t) m * K + (size_t) blk * ablk;
    float am = 0.f;
    for (int i = lane; i < ablk; i += 32) am = fmaxf(am, fabsf(xc[i]));
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.f;
    int sum = 0;
    for (int i = lane; i < ablk; i += 32) {
        const int q = am == 0.f ? 0 : __float2int_rn(xc[i] / d);
        const int k = blk * ablk + i;
        aqt[((size_t) (k >> 4) * Mp + m) * 16 + (k & 15)] = (int8_t) q;
        sum += q;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { dA[(size_t) blk * Mp + m] = d; sA[(size_t) blk * Mp + m] = sum; }
}

template <int B, int ABLK>
static __global__ void __launch_bounds__(256, 1) trl_nt_gemm_i8(const uint32_t * __restrict__ blob, const int8_t * __restrict__ aqt,
                                                                const float * __restrict__ dA, const int * __restrict__ sA,
                                                                float * __restrict__ y, int N, int K, int M, int Mp) {
    constexpr int BN = 128, BM = 128, BK = 16;
    static_assert(ABLK % BK == 0, "ABLK");
    __shared__ __align__(16) uint32_t Ws[2][BK / 2][BN];     // pairs of k: word = S(even k) | S(odd k) << 16
    __shared__ __align__(16) uint32_t Xs[2][BK / 4][BM];     // 4 consecutive int8 activations in k
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int nbands = N >> 6;
    const size_t bandWords = 64 + (size_t) (K >> 2) * 8 * B;

    const int drow = tid & 127, dj0 = tid >> 7;
    const uint32_t * bandp = blob + (size_t) min((n0 + drow) >> 6, nbands - 1) * bandWords;
    trl_nt_dec4<B> dec; dec.init(drow & 63);
    const uint32_t * tiles = bandp + 64;
    const int xc = tid & 127, xh = tid >> 7;                  // column, half of the 16 bytes

    int acc[8][8];
    float facc[8][8];
    float fd[8]; int fsc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0; facc[i][j] = 0.f; }

    uint32_t wh[2], wl[2];
    uint2 xv;
    auto gload = [&](int kt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) dec.load(tiles + (size_t) (kt * 4 + dj0 + 2 * h) * (8 * B), wh[h], wl[h]);
        xv = __ldg(reinterpret_cast<const uint2 *>(aqt + ((size_t) kt * Mp + m0 + xc) * 16 + xh * 8));
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int S[4];
            dec.sums(wh[h], wl[h], S);
            const int jj = dj0 + 2 * h;
            Ws[buf][jj * 2][drow]     = __byte_perm(S[0], S[1], 0x5410);
            Ws[buf][jj * 2 + 1][drow] = __byte_perm(S[2], S[3], 0x5410);
        }
        Xs[buf][xh * 2][xc]     = xv.x;
        Xs[buf][xh * 2 + 1][xc] = xv.y;
    };
    const int nk = K / BK;
    gload(0);
    sstore(0);
    __syncthreads();
    for (int kt = 0; kt < nk; ++kt) {
        const int cur = kt & 1;
        if (kt + 1 < nk) gload(kt + 1);
        if ((kt * BK) % ABLK == 0) {          // rescaling parameters of the block, read at the start and used at the end
            const int blk = (kt * BK) / ABLK;
            const float4 d0 = __ldg(reinterpret_cast<const float4 *>(dA + (size_t) blk * Mp + m0 + tx * 4));
            const float4 d1 = __ldg(reinterpret_cast<const float4 *>(dA + (size_t) blk * Mp + m0 + 64 + tx * 4));
            const int4 s0 = __ldg(reinterpret_cast<const int4 *>(sA + (size_t) blk * Mp + m0 + tx * 4));
            const int4 s1 = __ldg(reinterpret_cast<const int4 *>(sA + (size_t) blk * Mp + m0 + 64 + tx * 4));
            fd[0] = d0.x; fd[1] = d0.y; fd[2] = d0.z; fd[3] = d0.w; fd[4] = d1.x; fd[5] = d1.y; fd[6] = d1.z; fd[7] = d1.w;
            fsc[0] = 510 * s0.x; fsc[1] = 510 * s0.y; fsc[2] = 510 * s0.z; fsc[3] = 510 * s0.w;
            fsc[4] = 510 * s1.x; fsc[5] = 510 * s1.y; fsc[6] = 510 * s1.z; fsc[7] = 510 * s1.w;
            if constexpr (ABLK <= 64) {
#pragma unroll
                for (int j = 0; j < 8; ++j) fsc[j] -= 0x4B400000;     // int->float with fixed exponent (|v| < 2^22)
            }
        }
#pragma unroll
        for (int kq = 0; kq < BK / 4; ++kq) {
            const uint4 l0 = *reinterpret_cast<const uint4 *>(&Ws[cur][2 * kq][ty * 4]);
            const uint4 l1 = *reinterpret_cast<const uint4 *>(&Ws[cur][2 * kq][64 + ty * 4]);
            const uint4 h0 = *reinterpret_cast<const uint4 *>(&Ws[cur][2 * kq + 1][ty * 4]);
            const uint4 h1 = *reinterpret_cast<const uint4 *>(&Ws[cur][2 * kq + 1][64 + ty * 4]);
            const uint4 b0 = *reinterpret_cast<const uint4 *>(&Xs[cur][kq][tx * 4]);
            const uint4 b1 = *reinterpret_cast<const uint4 *>(&Xs[cur][kq][64 + tx * 4]);
            const uint32_t wl8[8] = { l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w };
            const uint32_t wh8[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
            const uint32_t a8[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] = trl_nt_dp2a_lo(wl8[i], a8[j], acc[i][j]);
                    acc[i][j] = trl_nt_dp2a_hi(wh8[i], a8[j], acc[i][j]);
                }
        }
        if (((kt + 1) * BK) % ABLK == 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float fv;
                    if constexpr (ABLK <= 64) fv = __int_as_float(acc[i][j] - fsc[j]) - 12582912.0f;
                    else                      fv = (float) (acc[i][j] - fsc[j]);
                    facc[i][j] = fmaf(fd[j], fv, facc[i][j]);
                    acc[i][j] = 0;
                }
        }
        if (kt + 1 < nk) sstore(cur ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int rl = hi * 64 + ty * 4;
        if (n0 + rl >= N) continue;
        const float * sc = reinterpret_cast<const float *>(blob + (size_t) ((n0 + rl) >> 6) * bandWords);
        float s[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = sc[(rl & 63) + i] * (1.0f / TRL_MUL1_STD);
#pragma unroll
        for (int hj = 0; hj < 2; ++hj) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = m0 + hj * 64 + tx * 4 + j;
                if (col < M) {
                    *reinterpret_cast<float4 *>(y + (size_t) col * N + n0 + rl) =
                        make_float4(facc[hi * 4 + 0][hj * 4 + j] * s[0], facc[hi * 4 + 1][hj * 4 + j] * s[1],
                                    facc[hi * 4 + 2][hj * 4 + j] * s[2], facc[hi * 4 + 3][hj * 4 + j] * s[3]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Launch. mode: TRL_NT_F32 / H2 / I8; bits 2 or 3; blob = weights of band 0; x32 = rotated activations [M][K] fp32.
// scratch: trl_nt_scratch_bytes(mode, K, M, ablk) bytes (0 for F32), allocated by the caller.
// ablk (I8 only): 32 (same quantization as the GEMV) or 128 (faster, slightly larger error).
// ---------------------------------------------------------------------------------------------------------
static inline int trl_nt_mp(int M) { return (M + 127) & ~127; }

static inline size_t trl_nt_scratch_bytes(int mode, int K, int M, int ablk) {
    const size_t Mp = trl_nt_mp(M);
    if (mode == TRL_NT_H2) return ((size_t) K * Mp * 2 + 15) / 16 * 16 + Mp * 4;
    if (mode == TRL_NT_I8) return ((size_t) K * Mp + 15) / 16 * 16 + 2 * (size_t) (K / ablk) * Mp * 4;
    return 0;
}

static inline void trl_nt_run(int mode, int bits, const uint32_t * blob, const float * x32, void * scratch, float * y,
                              int N, int K, int M, int ablk, cudaStream_t st) {
    const int Mp = trl_nt_mp(M);
    const dim3 grid((N + 127) / 128, Mp / 128);
    if (mode == TRL_NT_F32) {
        if (bits == 2) trl_nt_gemm_f32<2><<<grid, 256, 0, st>>>(blob, x32, y, N, K, M);
        else           trl_nt_gemm_f32<3><<<grid, 256, 0, st>>>(blob, x32, y, N, K, M);
    } else if (mode == TRL_NT_H2) {
        __half * xh = (__half *) scratch;
        float * cs = (float *) ((char *) scratch + ((size_t) K * Mp * 2 + 15) / 16 * 16);
        if (Mp != M) {      // zero padding columns
            cudaMemsetAsync(xh, 0, (size_t) K * Mp * 2, st);
            cudaMemsetAsync(cs, 0, (size_t) Mp * 4, st);
        }
        trl_nt_scale_cols_f16<<<M, 256, 0, st>>>(x32, xh, cs, K, Mp);
        if (bits == 2) trl_nt_gemm_h2<2, 2><<<grid, 256, 0, st>>>(blob, (const __half2 *) xh, cs, y, N, K, M, Mp);
        else           trl_nt_gemm_h2<3, 2><<<grid, 256, 0, st>>>(blob, (const __half2 *) xh, cs, y, N, K, M, Mp);
    } else {
        int8_t * aqt = (int8_t *) scratch;
        float * dA = (float *) ((char *) scratch + ((size_t) K * Mp + 15) / 16 * 16);
        int * sA = (int *) (dA + (size_t) (K / ablk) * Mp);
        if (Mp != M) {
            cudaMemsetAsync(aqt, 0, (size_t) K * Mp, st);
            cudaMemsetAsync(dA, 0, 2 * (size_t) (K / ablk) * Mp * 4, st);
        }
        const int64_t nw = (int64_t) M * (K / ablk);
        trl_nt_quant_act<<<(unsigned) ((nw + 7) / 8), 256, 0, st>>>(x32, aqt, dA, sA, K, M, Mp, ablk);
        if (ablk == 32) {
            if (bits == 2) trl_nt_gemm_i8<2, 32><<<grid, 256, 0, st>>>(blob, aqt, dA, sA, y, N, K, M, Mp);
            else           trl_nt_gemm_i8<3, 32><<<grid, 256, 0, st>>>(blob, aqt, dA, sA, y, N, K, M, Mp);
        } else {
            if (bits == 2) trl_nt_gemm_i8<2, 128><<<grid, 256, 0, st>>>(blob, aqt, dA, sA, y, N, K, M, Mp);
            else           trl_nt_gemm_i8<3, 128><<<grid, 256, 0, st>>>(blob, aqt, dA, sA, y, N, K, M, Mp);
        }
    }
}
