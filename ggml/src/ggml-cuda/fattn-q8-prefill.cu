// Attention for prompt reading (many query tokens, Q->ne[1] > 4) with q8_0 cache and GQA, without converting the cache to f16.
//
// Why: for head size 256 and GQA 6 ik uses mma_f16, which converts the WHOLE q8_0 cache to f16 on every call (fattn-common,
// need_f16_K/V): at 120k tokens that is ~480 MB of temporaries per layer (and ~5 times the bytes read/written). Here K and V
// are read in q8_0 in tiles of 32 positions with cp.async and dequantized in registers; no buffer proportional to the context
// (only the partials of the context chunks: ~13 MB for 512 query tokens, independent of the context).
//
// Structure (FlashAttention style, all for ONE KV head and ONE context chunk):
//  * one block = 4 warps = 32 rows of Q (G heads x consecutive tokens, row = token * G + head); two blocks per SM, so
//    their phases (tensor core / integer) overlap;
//  * S = K.Q^T with int8 mma (m16n8k32): raw K from the cache, Q in int8 digits relative to max|Q| of the row (QD = 3 digits:
//    24 bit, exact; QD = 2: 16 bit); the q8_0 block scales in fp32. Warp = 8 rows x 32 positions, whole D;
//  * inline fp32 softmax on registers (the mask is the engine's, one tile per token row via cp.async); the
//    reference maximum is updated only if exceeded by FQ3_LAZY;
//  * O = P.V with f16 mma (fp32 accumulation): exact V in half (int8). PS = 1: the V block scale goes into V (like the current
//    f16 path) and P is a half; PS = 2 (exact): the scale goes into P (P' = P dV 2^FQ3_SSHIFT, fp32), split into high + low.
//    Warp = 2 q8_0 blocks of V (64 columns of D) for all 32 rows;
//  * context chunks (split-K) merged by fq3_combine.
// Only D = 256, 4 <= G <= 8, cc >= 8.0, K/V rows and mask aligned to 16 bytes.
#include "fattn-q8-prefill.cuh"
#include "common.cuh"

#include <cfloat>
#include <algorithm>

#ifndef FQ3_QD
#define FQ3_QD 2          // int8 digits of Q: 2 (16 bit) or 3 (24 bit, exact)
#endif
#ifndef FQ3_PS
#define FQ3_PS 1          // 1: V = half(dV v), P half (like the current f16 path); 2: exact (P' high + low)
#endif
#ifndef FQ3_SSHIFT
#define FQ3_SSHIFT 8      // PS = 2: P' is multiplied by 2^SSHIFT before the split (the low part does not fall into half subnormals)
#endif
#ifndef FQ3_LAZY
#define FQ3_LAZY 3.f      // the softmax reference maximum stays within LAZY of the true maximum: P <= e^LAZY
#endif

#define FQ3_THREADS 128
#define FQ3_WARPS   4
#define FQ3_MQ      32    // rows of Q per block
#define FQ3_NT      4     // groups of 8 rows
#define FQ3_N       32    // positions per tile
#define FQ3_MT      9     // distinct tokens at most in a block of rows (G >= 4)
#define FQ3_STG     2     // stages of K, V and mask (tile i+1 is loaded while tile i is computed)
#define FQ3_PF      40    // pitch (elements) of P in smem
#define FQ3_ROWB    272   // bytes of a q8_0 row with D = 256
#define FQ3_MASKB   576   // bytes of the mask tile per stage (9 tokens x 32 positions x 2)

struct fq3_params {
    const char * Q; const char * K; const char * V; const char * mask;
    float * tmp_acc; float2 * tmp_ms; float * dst;
    float scale;
    int n_kv, chunk, nq, G, nhead, nkvh, mask_ne0;
    size_t nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbm1;
};

template <int PS>
static constexpr size_t fq3_smem_bytes() {
    return (size_t) 2 * FQ3_STG * FQ3_N * FQ3_ROWB + (size_t) FQ3_STG * FQ3_MASKB + (size_t) 2 * FQ3_MQ * FQ3_PF * (PS == 2 ? 4 : 2) +
           (size_t) 3 * FQ3_MQ * 4 + 16;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800   // Ampere only: elsewhere the kernel is empty and never launched
static __device__ __forceinline__ uint32_t fq3_lds16(const char * p) { return *reinterpret_cast<const uint16_t *>(p); }
static __device__ __forceinline__ uint32_t fq3_ld4(const char * p) { return fq3_lds16(p) | (fq3_lds16(p + 2) << 16); }
static __device__ __forceinline__ uint32_t fq3_prmt(uint32_t a, uint32_t b, uint32_t sel) { return __byte_perm(a, b, sel); }
static __device__ __forceinline__ float fq3_h2f(uint32_t u) { return __half2float(__ushort_as_half((unsigned short) u)); }
static __device__ __forceinline__ uint32_t fq3_pack2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}
// half2 {0x6400|u0, 0x6400|u1} - 1152 = {v0, v1} exact (u = byte with the sign bit flipped)
static __device__ __forceinline__ uint32_t fq3_dqx(uint32_t h) {
    const half2 c = __hsub2(*reinterpret_cast<const half2 *>(&h), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *reinterpret_cast<const uint32_t *>(&c);
}
static __device__ __forceinline__ uint32_t fq3_hmul(uint32_t a, uint32_t b) {
    const half2 r = __hmul2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);
}
static __device__ __forceinline__ void fq3_split(float a, float b, uint32_t & hi, uint32_t & lo) {
    const half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    hi = *reinterpret_cast<const uint32_t *>(&h);
    lo = fq3_pack2(a - f.x, b - f.y);
}
static __device__ __forceinline__ void fq3_mma(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void fq3_mma_s8(int * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void fq3_mma_s8u8(int * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void fq3_cp_async16(void * dst, const void * src) {
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
static __device__ __forceinline__ void fq3_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
static __device__ __forceinline__ void fq3_cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
static __device__ __forceinline__ uint32_t fq3_pk4(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}
#endif

// 1D grid: (chunk, KV head, row tile) with the row tile varying fastest: the blocks in flight together read
// the same cache and serve it from the L2.
// Software pipeline: iteration i does S and softmax of tile i (warp = 8 rows of Q x 32 positions, in registers) and, interleaved,
// the product O += P V of tile i-1 (warp = 2 q8_0 blocks of V = 64 columns of D, all 32 rows). One barrier per tile.
template <int QD, int PS>
static __global__ void __launch_bounds__(FQ3_THREADS, 2) fq3_partial(const fq3_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NB = 8, SP = FQ3_ROWB, NCH = 17, NQP = FQ3_MQ, NT = FQ3_NT;
    constexpr int PBYTES = NQP * FQ3_PF * (PS == 2 ? 4 : 2);                  // one buffer of P
    extern __shared__ __align__(16) char smem[];
    char * Kring = smem;                                                      // FQ3_STG x [32 rows]
    char * Vring = Kring + FQ3_STG * FQ3_N * SP;                              // FQ3_STG x [32 rows]
    char * Mring = Vring + FQ3_STG * FQ3_N * SP;                              // FQ3_STG x FQ3_MASKB
    char * Pbuf  = Mring + FQ3_STG * FQ3_MASKB;                               // 2 x PBYTES
    float * corrS = reinterpret_cast<float *>(Pbuf + 2 * PBYTES);            // 2 x [32]
    float * qamax = corrS + 2 * NQP;                                          // [32]
    int   * flags = reinterpret_cast<int *>(qamax + NQP);                     // [3]

    const int G = p.G, R = p.nq * G;
    const int nQ = (R + FQ3_MQ - 1) / FQ3_MQ;
    const int id = blockIdx.x;
    const int qt = id % nQ, rest = id / nQ, kvh = rest % p.nkvh, ch = rest / p.nkvh;
    const int row0 = qt * FQ3_MQ, nvalid = min(FQ3_MQ, R - row0);
    const int tk_lo = row0 / G;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane >> 2, m = lane & 3;
    const int t0 = ch * p.chunk, t1 = min(p.n_kv, t0 + p.chunk);
    const int ntiles = (t1 - t0 + FQ3_N - 1) / FQ3_N;

    for (int i = tid; i < 2 * PBYTES / 4; i += FQ3_THREADS) reinterpret_cast<float *>(Pbuf)[i] = 0.f;
    for (int i = tid; i < 2 * NQP; i += FQ3_THREADS) corrS[i] = 1.f;
    if (tid < 3) flags[tid] = 0;

    // ---- Q: max |Q| per row, then the int8 digits in registers (B operand of S: column n = r is row 8 w + r)
    for (int ii = 0; ii < FQ3_MQ / FQ3_WARPS; ++ii) {
        const int row = w + FQ3_WARPS * ii;
        float mx = 0.f;
        if (row < nvalid) {
            const int gr = row0 + row, tk = gr / G, h = gr - tk * G;
            const float * qr = reinterpret_cast<const float *>(p.Q + (size_t) tk * p.nbq1 + (size_t) (kvh * G + h) * p.nbq2);
#pragma unroll
            for (int i = 0; i < 8; ++i) mx = fmaxf(mx, fabsf(qr[lane + 32 * i]));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        }
        if (lane == 0) qamax[row] = mx;
    }
    __syncthreads();
    constexpr float QMAX = QD == 3 ? 8388607.f : 32767.f;
    float dq[2];                                              // dequantization scale of the columns of C (rows 8 w + 2 m + c)
    dq[0] = qamax[8 * w + 2 * m] * (1.f / QMAX);
    dq[1] = qamax[8 * w + 2 * m + 1] * (1.f / QMAX);
    uint32_t qp[NB][QD][2];
    {
        const int n = 8 * w + r;
        const float mxn = qamax[n];
        const float inv = mxn > 0.f ? QMAX / mxn : 0.f;
#pragma unroll
        for (int blk = 0; blk < NB; ++blk) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                float f[4] = {0.f, 0.f, 0.f, 0.f};
                if (n < nvalid) {
                    const int gr = row0 + n, tk = gr / G, h = gr - tk * G;
                    const float * qr = reinterpret_cast<const float *>(p.Q + (size_t) tk * p.nbq1 + (size_t) (kvh * G + h) * p.nbq2)
                                       + 32 * blk + 16 * hf + 4 * m;
                    f[0] = qr[0]; f[1] = qr[1]; f[2] = qr[2]; f[3] = qr[3];
                }
                int d0[4], d1[4], d2[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (QD == 3) {
                        const int qi = max(-8388607, min(8388607, __float2int_rn(f[e] * inv)));
                        d0[e] = qi >> 16; d1[e] = (qi >> 8) & 255; d2[e] = qi & 255;
                    } else {
                        const int qi = max(-32767, min(32767, __float2int_rn(f[e] * inv)));
                        d0[e] = qi >> 8; d1[e] = qi & 255; d2[e] = 0;
                    }
                }
                qp[blk][0][hf] = fq3_pk4(d0[0], d0[1], d0[2], d0[3]);
                qp[blk][1][hf] = fq3_pk4(d1[0], d1[1], d1[2], d1[3]);
                if (QD == 3) qp[blk][QD - 1][hf] = fq3_pk4(d2[0], d2[1], d2[2], d2[3]);
            }
        }
    }

    // softmax state of the thread's two rows (8 w + 2 m + c) and O^T accumulators
    // (d = 32 (2 w + b) + 4 r + 2 t (+1 if c >= 2), row = 8 nt + 2 m + (c & 1))
    float mq[2] = {-FLT_MAX, -FLT_MAX}, ps[2] = {0.f, 0.f};
    const int tkl0 = (row0 + 8 * w + 2 * m) / G - tk_lo, tkl1 = (row0 + 8 * w + 2 * m + 1) / G - tk_lo;
    float acc[2][2][NT][4];                                   // [b][t][nt][c]
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[b][t][nt][c] = 0.f;

    // ---- prefetch of a tile (K, V: 32 rows of 272 bytes; mask: the block's tokens x 32 positions)
    const char * kbase = p.K + (size_t) kvh * p.nbk2 + lane * 16, * vbase = p.V + (size_t) kvh * p.nbv2 + lane * 16;
    const int tk_hi = min(p.nq - 1, (row0 + nvalid - 1) / G);
    auto issue = [&](int tile) {
        char * kb = Kring + (size_t) (tile % FQ3_STG) * FQ3_N * SP, * vb = Vring + (size_t) (tile % FQ3_STG) * FQ3_N * SP;
        const int tb = t0 + tile * FQ3_N;
        if (lane < NCH) {
#pragma unroll
            for (int i = 0; i < FQ3_N / FQ3_WARPS; ++i) {
                const int row = w + FQ3_WARPS * i, t = min(tb + row, t1 - 1);
                fq3_cp_async16(kb + (size_t) row * SP + lane * 16, kbase + (size_t) t * p.nbk1);
                fq3_cp_async16(vb + (size_t) row * SP + lane * 16, vbase + (size_t) t * p.nbv1);
            }
        }
        if (p.mask && tid < FQ3_MT * 4) {
            const int j = tid >> 2, c = tid & 3, tk = tk_lo + j, pos = tb + 8 * c;
            if (tk <= tk_hi && pos < p.mask_ne0)
                fq3_cp_async16(Mring + (size_t) (tile % FQ3_STG) * FQ3_MASKB + j * 64 + c * 16, p.mask + (size_t) tk * p.nbm1 + (size_t) pos * 2);
        }
    };

    float sacc[2][4];                                            // [mt][e]: pos = 16 mt + r (+8 if e >= 2), row 8 w + 2 m + (e & 1)

    // ---- S^T[pos, row] = dq[row] sum_blk dK[pos, blk] (K_int . Q_int): warp = 32 positions (2 tiles of 16) x 8 rows, whole D;
    // one q8_0 block of K per call (so it interleaves with O of the previous tile)
    auto s_blk = [&](const char * Ks, int blk) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const char * rowA = Ks + (size_t) (16 * mt + r) * SP;
            const char * rowB = rowA + 8 * SP;
            const int off = 34 * blk + 2 + 4 * m;                      // 4 aligned bytes if blk is odd
            uint32_t a0, a1, a2, a3;
            if (blk & 1) {
                a0 = *reinterpret_cast<const uint32_t *>(rowA + off); a2 = *reinterpret_cast<const uint32_t *>(rowA + off + 16);
                a1 = *reinterpret_cast<const uint32_t *>(rowB + off); a3 = *reinterpret_cast<const uint32_t *>(rowB + off + 16);
            } else {
                a0 = fq3_ld4(rowA + off); a2 = fq3_ld4(rowA + off + 16);
                a1 = fq3_ld4(rowB + off); a3 = fq3_ld4(rowB + off + 16);
            }
            const float dA = fq3_h2f(fq3_lds16(rowA + 34 * blk)), dB = fq3_h2f(fq3_lds16(rowB + 34 * blk));
            int c1[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0}, c3[4] = {0, 0, 0, 0};
            fq3_mma_s8(c1, a0, a1, a2, a3, qp[blk][0][0], qp[blk][0][1]);
            fq3_mma_s8u8(c2, a0, a1, a2, a3, qp[blk][1][0], qp[blk][1][1]);
            if (QD == 3) fq3_mma_s8u8(c3, a0, a1, a2, a3, qp[blk][2][0], qp[blk][2][1]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float sb = QD == 3 ? fmaf((float) c1[e], 65536.f, (float) (c2[e] * 256 + c3[e]))
                                         : (float) (c1[e] * 256 + c2[e]);
                sacc[mt][e] = fmaf(sb, e < 2 ? dA : dB, sacc[mt][e]);
            }
        }
    };

    // ---- inline softmax on registers: x = S dq scale + mask; reference maximum updated only if exceeded by FQ3_LAZY
    auto softmax_part = [&](int i, const char * Ms) {
        float x[2][4];
        bool need = false;
        const int pbase = t0 + i * FQ3_N;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int pl = 16 * mt + r + 8 * (e >> 1);                  // position in the tile
                const int c = e & 1;
                const float mk = pbase + pl >= t1 ? -INFINITY
                    : (p.mask ? __half2float(*reinterpret_cast<const half *>(Ms + ((c ? tkl1 : tkl0) * 32 + pl) * 2)) : 0.f);
                x[mt][e] = fmaf(sacc[mt][e] * dq[c], p.scale, mk);
                need |= x[mt][e] > mq[c] + FQ3_LAZY;
            }
        float corr[2] = {1.f, 1.f};
        if (__any_sync(0xffffffffu, need)) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                float mx = fmaxf(fmaxf(x[0][c], x[0][c + 2]), fmaxf(x[1][c], x[1][c + 2]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
                const float mn = fmaxf(mq[c], mx);
                corr[c] = __expf(mq[c] - mn);
                mq[c] = mn;
            }
            if (lane == 0) flags[i % 3] = 1;
        }
        if (r == 0) *reinterpret_cast<float2 *>(corrS + (i & 1) * NQP + 8 * w + 2 * m) = make_float2(corr[0], corr[1]);
        char * Pb = Pbuf + (i & 1) * PBYTES;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float sum = 0.f;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float pv = __expf(x[mt][c + 2 * h] - mq[c]);
                    sum += pv;
                    const int q = 8 * w + 2 * m + c, pl = 16 * mt + r + 8 * h;
                    if (PS == 2) reinterpret_cast<float *>(Pb)[q * FQ3_PF + pl] = pv;
                    else         reinterpret_cast<half *>(Pb)[q * FQ3_PF + pl] = __float2half(pv);
                }
            ps[c] = fmaf(ps[c], corr[c], sum);
        }
        if (tid == 0) flags[(i + 1) % 3] = 0;
    };

    // ---- O^T[d, row] += V^T[d, pos] P^T[pos, row] of tile j: warp = q8_0 blocks 2 w and 2 w + 1 of V, all 32 rows
    auto rescale_part = [&](int j) {
        if (*reinterpret_cast<volatile int *>(flags + (j + 3) % 3)) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float2 c = *reinterpret_cast<const float2 *>(corrS + (j & 1) * NQP + 8 * nt + 2 * m);
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        acc[b][t][nt][0] *= c.x; acc[b][t][nt][1] *= c.y; acc[b][t][nt][2] *= c.x; acc[b][t][nt][3] *= c.y;
                    }
            }
        }
    };
    uint32_t av[2][2][2][4];                                 // A operand of O: V of the previous tile, [b][ks2][t][4]
    float dv[2][2][4];                                       // PS = 2: scales (x 2^SSHIFT) of the 4 positions of each step
    auto pv_prep = [&](int j) {
        const char * Vs = Vring + (size_t) (j % FQ3_STG) * FQ3_N * SP;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int blk = 2 * w + b;                           // b is constant: the 4-byte alignment is known
            const int off = 34 * blk + 2 + 4 * r;
#pragma unroll
            for (int ks2 = 0; ks2 < 2; ++ks2) {
                const char * rA = Vs + (size_t) (16 * ks2 + 2 * m) * SP;       // positions pA, pA+1, pA+8, pA+9
                const char * rB = rA + SP, * rC = rA + 8 * SP, * rD = rA + 9 * SP;
                uint32_t wA, wB, wC, wD;
                if (b & 1) {
                    wA = *reinterpret_cast<const uint32_t *>(rA + off); wB = *reinterpret_cast<const uint32_t *>(rB + off);
                    wC = *reinterpret_cast<const uint32_t *>(rC + off); wD = *reinterpret_cast<const uint32_t *>(rD + off);
                } else {
                    wA = fq3_ld4(rA + off); wB = fq3_ld4(rB + off); wC = fq3_ld4(rC + off); wD = fq3_ld4(rD + off);
                }
                wA ^= 0x80808080u; wB ^= 0x80808080u; wC ^= 0x80808080u; wD ^= 0x80808080u;
                const uint32_t sA = fq3_lds16(rA + 34 * blk), sB = fq3_lds16(rB + 34 * blk), sC = fq3_lds16(rC + 34 * blk), sD = fq3_lds16(rD + 34 * blk);
                // 4x4 byte transpose: X = {A0,B0,A1,B1}, Y = {A2,B2,A3,B3} (columns d, d+1, d+2, d+3)
                const uint32_t X1 = fq3_prmt(wA, wB, 0x5140), Y1 = fq3_prmt(wA, wB, 0x7362);
                const uint32_t X2 = fq3_prmt(wC, wD, 0x5140), Y2 = fq3_prmt(wC, wD, 0x7362);
                uint32_t sAB = 0, sCD = 0;
                if (PS == 2) {
                    const float ss = (float) (1 << FQ3_SSHIFT);
                    dv[b][ks2][0] = ss * fq3_h2f(sA); dv[b][ks2][1] = ss * fq3_h2f(sB); dv[b][ks2][2] = ss * fq3_h2f(sC); dv[b][ks2][3] = ss * fq3_h2f(sD);
                } else {   // PS = 1: the V block scale goes into V (like the current f16 path: V = half(dV v))
                    sAB = sA | (sB << 16); sCD = sC | (sD << 16);
                }
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const uint32_t u1 = t ? Y1 : X1, u2 = t ? Y2 : X2;
                    uint32_t v0 = fq3_dqx(fq3_prmt(u1, 0x64646464u, 0x4140));   // row d,   low k
                    uint32_t v1 = fq3_dqx(fq3_prmt(u1, 0x64646464u, 0x4342));   // row d+1, low k
                    uint32_t v2 = fq3_dqx(fq3_prmt(u2, 0x64646464u, 0x4140));   // row d,   high k
                    uint32_t v3 = fq3_dqx(fq3_prmt(u2, 0x64646464u, 0x4342));   // row d+1, high k
                    if (PS == 1) { v0 = fq3_hmul(v0, sAB); v1 = fq3_hmul(v1, sAB); v2 = fq3_hmul(v2, sCD); v3 = fq3_hmul(v3, sCD); }
                    av[b][ks2][t][0] = v0; av[b][ks2][t][1] = v1; av[b][ks2][t][2] = v2; av[b][ks2][t][3] = v3;
                }
            }
        }
    };
    // a group of 8 rows and one block of V per call: fresh tile accumulator (tensor core accumulation truncates: on an
    // accumulator that lives for thousands of tiles the defect adds up), then rounded fp32 sum
    auto pv_nt = [&](int j, int nt, int b) {
        const char * Pb = Pbuf + (j & 1) * PBYTES;
        float tl[2][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) { tl[t][0] = tl[t][1] = tl[t][2] = tl[t][3] = 0.f; }
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ++ks2) {
            uint32_t bh0, bl0 = 0, bh1, bl1 = 0;
            if (PS == 2) {
                const float * Pf = reinterpret_cast<const float *>(Pb);
                const float2 u = *reinterpret_cast<const float2 *>(Pf + (8 * nt + r) * FQ3_PF + 16 * ks2 + 2 * m);
                const float2 v = *reinterpret_cast<const float2 *>(Pf + (8 * nt + r) * FQ3_PF + 16 * ks2 + 2 * m + 8);
                fq3_split(u.x * dv[b][ks2][0], u.y * dv[b][ks2][1], bh0, bl0);
                fq3_split(v.x * dv[b][ks2][2], v.y * dv[b][ks2][3], bh1, bl1);
            } else {
                const half * Ph = reinterpret_cast<const half *>(Pb);
                bh0 = *reinterpret_cast<const uint32_t *>(Ph + (8 * nt + r) * FQ3_PF + 16 * ks2 + 2 * m);
                bh1 = *reinterpret_cast<const uint32_t *>(Ph + (8 * nt + r) * FQ3_PF + 16 * ks2 + 2 * m + 8);
            }
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                fq3_mma(tl[t], av[b][ks2][t][0], av[b][ks2][t][1], av[b][ks2][t][2], av[b][ks2][t][3], bh0, bh1);
                if (PS == 2) fq3_mma(tl[t], av[b][ks2][t][0], av[b][ks2][t][1], av[b][ks2][t][2], av[b][ks2][t][3], bl0, bl1);
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[b][t][nt][c] += tl[t][c];
    };

    // ---- pipeline: iteration i does S/softmax of tile i and O of tile i-1 (the dequantization of V of tile i is done in the shadow
    // of the mma instructions of S, before the softmax). For i = 0 the tile "-1" has P = 0 and V = 0 in registers: null contribution.
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ++ks2) {
            dv[b][ks2][0] = dv[b][ks2][1] = dv[b][ks2][2] = dv[b][ks2][3] = 0.f;
#pragma unroll
            for (int t = 0; t < 2; ++t) av[b][ks2][t][0] = av[b][ks2][t][1] = av[b][ks2][t][2] = av[b][ks2][t][3] = 0u;
        }
    issue(0); fq3_cp_commit();
    for (int i = 0; i < ntiles; ++i) {
        fq3_cp_wait0();
        __syncthreads();                                         // tile i arrived; P and corrections of tile i-1 in smem
        if (i + 1 < ntiles) issue(i + 1);
        fq3_cp_commit();
        rescale_part(i - 1);
        const char * Ks = Kring + (size_t) (i % FQ3_STG) * FQ3_N * SP;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { sacc[mt][0] = sacc[mt][1] = sacc[mt][2] = sacc[mt][3] = 0.f; }
#pragma unroll
        for (int k = 0; k < 8; ++k) { s_blk(Ks, k); pv_nt(i - 1, k >> 1, k & 1); }
        pv_prep(i);                                              // V of tile i in registers for the next iteration
        softmax_part(i, Mring + (size_t) (i % FQ3_STG) * FQ3_MASKB);
    }
    fq3_cp_wait0();
    __syncthreads();
    rescale_part(ntiles - 1);
#pragma unroll
    for (int k = 0; k < 8; ++k) pv_nt(ntiles - 1, k >> 1, k & 1);

    // ---- chunk output: O not normalized (relative to the chunk maximum) and (m, s) per row
    const int nch = (p.n_kv + p.chunk - 1) / p.chunk;
    const size_t base = (size_t) kvh * R + row0;
    const float osc = PS == 2 ? 1.f / (float) (1 << FQ3_SSHIFT) : 1.f;
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int q = 8 * nt + 2 * m + (c & 1);
                    const int d = 32 * (2 * w + b) + 4 * r + 2 * t + (c >> 1);
                    if (q < nvalid) p.tmp_acc[((base + q) * nch + ch) * 256 + d] = acc[b][t][nt][c] * osc;
                }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        float s = ps[c];
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 8);
        s += __shfl_xor_sync(0xffffffffu, s, 16);
        const int q = 8 * w + 2 * m + c;
        if (r == 0 && q < nvalid) p.tmp_ms[(base + q) * nch + ch] = make_float2(mq[c], s);
    }
#endif
}

// one row (kvh, token, head) per block, 256 threads = the 256 columns of D
static __global__ void fq3_combine(const float * __restrict__ tmp_acc, const float2 * __restrict__ tmp_ms,
                                   float * __restrict__ dst, int nch, int R, int G, int nhead) {
    const size_t qi = blockIdx.x;                                  // kvh * R + row
    const int kvh = (int) (qi / R), row = (int) (qi - (size_t) kvh * R), tk = row / G, h = row - tk * G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, tmp_ms[qi * nch + c].x);
    float S = 0.f, A = 0.f;
    for (int c = 0; c < nch; ++c) {
        const float2 ms = tmp_ms[qi * nch + c];
        const float f = __expf(ms.x - M);
        S += ms.y * f;
        A += tmp_acc[(qi * nch + c) * 256 + d] * f;
    }
    dst[((size_t) tk * nhead + kvh * G + h) * 256 + d] = S > 0.f ? A / S : 0.f;
}

// Context chunks: (base blocks) x nch well distributed over the SMs (two blocks per SM), chunks of at least 1024 positions
static inline void fq3_plan(int n_kv, int nblk_base, int nsm, size_t rows_total, int * chunk, int * nch) {
    int best = 1;
    double best_cost = 1e30;
    const int nmax = std::max(1, std::min(8, n_kv / 1024));
    const int slots = 2 * nsm;
    for (int n = 1; n <= nmax; ++n) {
        if ((double) rows_total * n * 256 * 4 > 40e6) break;                    // temporaries <= 40 MB
        const double waves = (double) ((nblk_base * n + slots - 1) / slots);
        const double cost = waves / n * (1.0 + 0.004 * n);                      // relative time + small merge cost
        if (cost < best_cost - 1e-9) { best_cost = cost; best = n; }
    }
    int c = (n_kv + best - 1) / best;
    c = (c + FQ3_N - 1) / FQ3_N * FQ3_N;
    *chunk = c;
    *nch = (n_kv + c - 1) / c;
}

template <int QD, int PS>
static void fq3_run_t(const fq3_params & p, int nch, cudaStream_t stream) {
    const int R = p.nq * p.G, nQ = (R + FQ3_MQ - 1) / FQ3_MQ;
    constexpr size_t smem = fq3_smem_bytes<PS>();
    cudaFuncSetAttribute(fq3_partial<QD, PS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    fq3_partial<QD, PS><<<nQ * p.nkvh * nch, FQ3_THREADS, smem, stream>>>(p);
    fq3_combine<<<(unsigned) ((size_t) p.nkvh * R), 256, 0, stream>>>(p.tmp_acc, p.tmp_ms, p.dst, nch, R, p.G, p.nhead);
}
static void fq3_run(const fq3_params & p, int nch, cudaStream_t stream) { fq3_run_t<FQ3_QD, FQ3_PS>(p, nch, stream); }

#ifndef FQ_STANDALONE

bool ggml_cuda_fattn_q8_prefill_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    static const bool off = getenv("FA_Q8_PREFILL") != nullptr && atoi(getenv("FA_Q8_PREFILL")) == 0;   // sentinel: the usual path
    if (off) return false;
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc < CC_AMPERE) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || dst->type != GGML_TYPE_F32) return false;
    if (Q->ne[1] <= 4 || Q->ne[3] != 1 || K->ne[3] != 1) return false;
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) return false;
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] < 4 || Q->ne[2] / K->ne[2] > 8 || V->ne[2] != K->ne[2]) return false;
    if (dst->src[4] != nullptr) return false;                                       // sink
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] * mask->ne[3] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1]) return false;
    if (mask->ne[0] % 8 != 0 || mask->nb[1] % 16 != 0 || (uintptr_t) mask->data % 16 != 0) return false;
    if ((uintptr_t) K->data % 16 != 0 || (uintptr_t) V->data % 16 != 0 || K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 ||
        V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0) return false;
    float max_bias = 0.f, softcap = 0.f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.f || softcap != 0.f || dst->op_params[4] != 0) return false;   // ALiBi, softcap, SWA
    return ggml_is_contiguous(dst) && Q->nb[0] == sizeof(float);
}

void ggml_cuda_fattn_q8_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    const int nkvh = (int) K->ne[2];
    fq3_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.n_kv = (int) K->ne[1]; p.nq = (int) Q->ne[1]; p.nhead = (int) Q->ne[2]; p.nkvh = nkvh; p.G = p.nhead / nkvh;
    p.mask_ne0 = (int) mask->ne[0];
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2];
    p.nbm1 = mask->nb[1];
    p.dst = (float *) dst->data;
    const size_t R = (size_t) p.nq * p.G;
    int nch;
    fq3_plan(p.n_kv, nkvh * (int) ((R + FQ3_MQ - 1) / FQ3_MQ), ggml_cuda_info().devices[ggml_cuda_get_device()].nsm, nkvh * R, &p.chunk, &nch);
    ggml_cuda_pool_alloc<float>  acc(ctx.pool(), (size_t) nkvh * R * nch * 256);
    ggml_cuda_pool_alloc<float2> ms(ctx.pool(), (size_t) nkvh * R * nch);
    p.tmp_acc = acc.get(); p.tmp_ms = ms.get();
    fq3_run(p, nch, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

#endif  // FQ_STANDALONE
