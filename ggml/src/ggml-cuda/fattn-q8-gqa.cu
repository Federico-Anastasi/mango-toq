// Decode attention (1..4 query tokens) with q8_0 cache and GQA, without converting the cache to f16.
//
// Why: for head size 256 and GQA 6, ik uses mma_new, which converts the WHOLE q8_0 cache to f16 on every call
// (fattn-common, need_f16_K/V): it reads q8, writes f16 and rereads f16, ~5 times the bytes needed, plus a
// temporary buffer that at 90k tokens of context is ~370 MB per layer (28/10: out of memory on the 3060 Ti).
// Here one block serves one KV head and one chunk of context for ALL the G query heads that share it and for
// all the query tokens (NQ <= 4): every row of K and V is read only once, in q8_0. Inline softmax, then a
// second kernel merges the chunks. Only D = 128/256, G <= 8, no ALiBi/softcap/sink: the rest goes through the
// usual path.
//
// Two kernels:
//  * fq2_* (Ampere, sm_80+): multi-stage cp.async (coalesced 16-byte reads) + tensor cores. A tile = 32
//    positions, 4 warps. Numerically aligned with the fp32 kernel (deviation ~5e-6 from double precision; the previous fp32 kernel ~1-2e-6):
//    - S = K.Q^T in int8 x int8 -> int32 (mma m16n8k32): raw K from the cache, Q as three int8 digits, signed relative to
//      max|Q| of the query (error max|Q| 2^-21); the scales of the q8_0 blocks and of Q are applied in fp32;
//    - inline softmax in fp32 (warp = query, lane = position), reference maximum updated only if exceeded by FQ2_LAZY;
//    - O = P.V: exact V in half (int8), the V block scale goes into P (P' = P dV, fp32), P' split into two halves
//      (high + low), two f16 mma (fp32 accumulation) per product, the tile is accumulated in a fresh accumulator.
//    Warp: GEMM1 = 16 positions x half of D (the two halves are summed in smem), GEMM2 = 64 columns of D.
//  * fq_* (original, 1 token, any GPU): stays for GPUs without cp.async/mma (Pascal, Turing).
#include "fattn-q8-gqa.cuh"
#include "common.cuh"

#include <cfloat>

// ============================ original kernel (1 token, any GPU; unchanged) ============================

#define FQ_WARPS 4

// One lane covers 8 consecutive values of the head (D/8 lanes per row: 32 for D = 256, 16 for D = 128).
template <int D>
static __device__ __forceinline__ void fq_load_row(const char * __restrict__ row, int lane, float v[8]) {
    constexpr int LPR = D / 8;
    if (lane >= LPR) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.f;
        return;
    }
    const int b = lane >> 2, j = lane & 3;                          // q8_0 block of 32 values, quarter j
    const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row) + b;
    const float d = __half2float(blk->d);
    const uint16_t * q = reinterpret_cast<const uint16_t *>(blk->qs + 8 * j);   // 2-byte aligned
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint16_t w = q[i];
        v[2 * i]     = d * (float) (int8_t) (w & 0xff);
        v[2 * i + 1] = d * (float) (int8_t) (w >> 8);
    }
}

template <int D, int G>
static __global__ void __launch_bounds__(FQ_WARPS * 32) fq_partial(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, float * __restrict__ tmp_acc, float2 * __restrict__ tmp_ms,
        float scale, int n_kv, int chunk, size_t nbq2, size_t nbk1, size_t nbk2, size_t nbv1, size_t nbv2) {
    constexpr int LPR = D / 8;
    const int kvh = blockIdx.x, ch = blockIdx.y, nch = gridDim.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int t0 = ch * chunk, t1 = min(n_kv, t0 + chunk);

    // Q of the G heads of the group, already scaled: 8 values per lane
    float q[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float * qh = reinterpret_cast<const float *>(Q + (size_t) (kvh * G + h) * nbq2);
#pragma unroll
        for (int i = 0; i < 8; ++i) q[h][i] = lane < LPR ? scale * qh[8 * lane + i] : 0.f;
    }
    float m[G], s[G], acc[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        m[h] = -FLT_MAX; s[h] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[h][i] = 0.f;
    }
    // FQ_U positions per iteration: the reads of K, V and mask all start before the computations (the kernel is
    // limited by memory latency, not by computation)
    constexpr int FQ_U = 2;
    for (int tb = t0 + w; tb < t1; tb += FQ_U * FQ_WARPS) {
        float k[FQ_U][8], v[FQ_U][8], mk[FQ_U];
#pragma unroll
        for (int u = 0; u < FQ_U; ++u) {
            const int t = tb + u * FQ_WARPS;
            mk[u] = t >= t1 ? -INFINITY : (mask ? __half2float(mask[t]) : 0.f);
            const int tt = t < t1 ? t : t0;                          // valid position for the read
            fq_load_row<D>(K + (size_t) tt * nbk1 + (size_t) kvh * nbk2, lane, k[u]);
            fq_load_row<D>(V + (size_t) tt * nbv1 + (size_t) kvh * nbv2, lane, v[u]);
        }
#pragma unroll
        for (int u = 0; u < FQ_U; ++u) {
            if (mk[u] == -INFINITY) continue;                        // uniform across the warp
            float sc[G];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                float p = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) p = fmaf(q[h][i], k[u][i], p);
                sc[h] = p;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                for (int h = 0; h < G; ++h) sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], off);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float x = sc[h] + mk[u];
                const float mn = fmaxf(m[h], x);
                const float corr = __expf(m[h] - mn), p = __expf(x - mn);
                m[h] = mn;
                s[h] = s[h] * corr + p;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[h][i] = fmaf(acc[h][i], corr, p * v[u][i]);
            }
        }
    }
    // merge of the block's warps in shared memory
    __shared__ float sm_m[FQ_WARPS][G], sm_s[FQ_WARPS][G];
    __shared__ float sm_acc[FQ_WARPS][G][D];
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) { sm_m[w][h] = m[h]; sm_s[w][h] = s[h]; }
    }
    if (lane < LPR) {
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int i = 0; i < 8; ++i) sm_acc[w][h][8 * lane + i] = acc[h][i];
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D; idx += blockDim.x) {
        const int h = idx / D, d = idx - h * D;
        float M = -FLT_MAX;
#pragma unroll
        for (int ww = 0; ww < FQ_WARPS; ++ww) M = fmaxf(M, sm_m[ww][h]);
        float S = 0.f, A = 0.f;
#pragma unroll
        for (int ww = 0; ww < FQ_WARPS; ++ww) {
            const float f = __expf(sm_m[ww][h] - M);
            S += sm_s[ww][h] * f;
            A += sm_acc[ww][h][d] * f;
        }
        const int qh = kvh * G + h;
        tmp_acc[((size_t) qh * nch + ch) * D + d] = A;
        if (d == 0) tmp_ms[(size_t) qh * nch + ch] = make_float2(M, S);
    }
}

template <int D>
static __global__ void fq_combine(const float * __restrict__ tmp_acc, const float2 * __restrict__ tmp_ms,
                                  float * __restrict__ dst, int nch) {
    const int qh = blockIdx.x, d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, tmp_ms[(size_t) qh * nch + c].x);
    float S = 0.f, A = 0.f;
    for (int c = 0; c < nch; ++c) {
        const float2 ms = tmp_ms[(size_t) qh * nch + c];
        const float f = __expf(ms.x - M);
        S += ms.y * f;
        A += tmp_acc[((size_t) qh * nch + c) * D + d] * f;
    }
    dst[(size_t) qh * D + d] = S > 0.f ? A / S : 0.f;
}

// ============================ Ampere kernel (fq2) ============================

// P' = P dV is multiplied by 2^FQ2_SSHIFT before the hi/lo split (and the result is divided at the end): so the low
// part does not fall into half subnormals (at long context the small weights carry the mass, and with SSHIFT = 5 the
// deviation on real data was 2.2e-6 instead of 1.6e-6). The softmax reference maximum stays within FQ2_LAZY of the true maximum =>
// P' <= e^FQ2_LAZY dV 2^FQ2_SSHIFT < 65504 if dV (V block scale, max|V|/127) < ~12.
#ifndef FQ2_SSHIFT
#define FQ2_SSHIFT  8
#endif
#ifndef FQ2_LAZY
#define FQ2_LAZY    3.f
#endif
#define FQ2_THREADS 128
#define FQ2_WARPS   4
#define FQ2_TILE    32     // positions per tile
#define FQ2_SPP     36     // pitch (float) of Sp: 36 -> writes without bank conflicts
#define FQ2_PF      40     // pitch (float) of P: float2 reads without conflicts

struct fq2_params {
    const char * Q; const char * K; const char * V; const char * mask;   // mask may be null
    float * tmp_acc; float2 * tmp_ms; float * dst;
    float scale;
    int n_kv, chunk, nq, G, nhead;
    size_t nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbm1;                      // bytes
};

template <int D> struct fq2_geom {
    static constexpr int NB   = D / 32;                       // q8_0 blocks per row
    static constexpr int ROWB = NB * 34;                      // 272 (D = 256), 136 (D = 128)
    static constexpr int CS   = (ROWB % 16 == 0) ? 16 : 8;    // bytes per cp.async
    static constexpr int NCH  = ROWB / CS;
    static constexpr int SP   = (ROWB % 16 == 0) ? ROWB : ROWB + 8;   // row pitch in smem (odd-ish words: no conflicts)
    static constexpr int SB   = 2 * FQ2_TILE * SP;            // bytes per stage (K + V)
};

template <int D, int NT, int STAGES>
static constexpr size_t fq2_smem_bytes() {
    return (size_t) STAGES * fq2_geom<D>::SB + (size_t) 2 * 8 * NT * FQ2_SPP * 4 + (size_t) 8 * NT * FQ2_PF * 4 + (size_t) 2 * 8 * NT * 4 + 16;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800   // the rest is for Ampere only: on other GPUs the kernel is empty and never launched
static __device__ __forceinline__ uint32_t fq2_lds16(const char * p) { return *reinterpret_cast<const uint16_t *>(p); }
// 4 bytes at an even address (not necessarily a multiple of 4: the q8_0 blocks are 34 bytes)
static __device__ __forceinline__ uint32_t fq2_ld4(const char * p) { return fq2_lds16(p) | (fq2_lds16(p + 2) << 16); }
static __device__ __forceinline__ uint32_t fq2_prmt(uint32_t a, uint32_t b, uint32_t sel) { return __byte_perm(a, b, sel); }

// half2 {0x6400|u0, 0x6400|u1} = 1024 + u (u = unsigned byte, with the sign bit already flipped => v + 128):
// (h - 1152) is exactly v (integers in [-128, 127] are exact in half). The block scales are applied in fp32.
static __device__ __forceinline__ uint32_t fq2_dqx(uint32_t h) {
    const half2 c = __hsub2(*reinterpret_cast<const half2 *>(&h), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *reinterpret_cast<const uint32_t *>(&c);
}
static __device__ __forceinline__ uint32_t fq2_pack2(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// (a, b) -> two half2: high part and low part (a = hi + lo with ~2^-22 relative error)
static __device__ __forceinline__ void fq2_split(float a, float b, uint32_t & hi, uint32_t & lo) {
    // high part = a rounded to half, low part = remainder (exact in fp32, ~12 bits: half holds 11)
    const half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    hi = *reinterpret_cast<const uint32_t *>(&h);
    lo = fq2_pack2(a - f.x, b - f.y);
}

// A = K (signed int8), B = digits of Q: signed (s8) for the high digit, unsigned (u8) for the two low ones
static __device__ __forceinline__ void fq2_mma_s8(int * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void fq2_mma_s8u8(int * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Q (fp32) -> 24-bit signed integer relative to max|Q| of the query (inv = (2^23 - 1) / max), split into three bytes:
// q = p1 2^16 + p2 2^8 + p3 with p1 signed and p2, p3 unsigned (two's complement): quantization error max|Q| 2^-24.
static __device__ __forceinline__ void fq2_q3(float x, float inv, int & p1, int & p2, int & p3) {
    const int qi = max(-8388607, min(8388607, __float2int_rn(x * inv)));   // rounding x * inv can give 2^23
    p1 = qi >> 16;
    p2 = (qi >> 8) & 255;
    p3 = qi & 255;
}
static __device__ __forceinline__ uint32_t fq2_pk4(int a, int b, int c, int d) {
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}

static __device__ __forceinline__ void fq2_mma(float * c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int N>
static __device__ __forceinline__ void fq2_cp_async(void * dst, const void * src) {
    const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
    if (N == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    else         asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"  :: "r"(d), "l"(src));
}
static __device__ __forceinline__ void fq2_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void fq2_cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
#endif

// Grid (n. KV heads, n. context chunks); one block = 4 warps = one KV head, one chunk, all the G x NQ queries.
template <int D, int NT, int STAGES>
static __global__ void __launch_bounds__(FQ2_THREADS, 2) fq2_partial(const fq2_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using geo = fq2_geom<D>;
    constexpr int NB = geo::NB, CS = geo::CS, NCH = geo::NCH, SP = geo::SP, SB = geo::SB;
    constexpr int BPW = D / 128;       // q8_0 blocks (32 columns of D) of GEMM2 per warp: 2 (D = 256) or 1
    constexpr int NQP = 8 * NT;        // queries after padding to multiples of 8

    extern __shared__ __align__(16) char smem[];
    float * Sp    = reinterpret_cast<float *>(smem + (size_t) STAGES * SB);           // [2][NQP][FQ2_SPP]
    float * Pf    = Sp + 2 * NQP * FQ2_SPP;                                            // [NQP][FQ2_PF]
    float * corrS = Pf + NQP * FQ2_PF;                                                 // [NQP]
    float * qamax = corrS + NQP;                                                       // [NQP]: max |Q| per query
    int   * flags = reinterpret_cast<int *>(qamax + NQP);                              // [2]: does O need rescaling in this tile?

    const int kvh = blockIdx.x, ch = blockIdx.y, nch = gridDim.y;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane >> 2, m = lane & 3;
    const int mt = w & 1, dh = w >> 1;
    const int G = p.G, Nq = p.nq * G;
    const int t0 = ch * p.chunk, t1 = min(p.n_kv, t0 + p.chunk);
    const int ntiles = (t1 - t0 + FQ2_TILE - 1) / FQ2_TILE;

    for (int i = tid; i < NQP * FQ2_PF; i += FQ2_THREADS) Pf[i] = 0.f;
    for (int i = tid; i < NQP; i += FQ2_THREADS) corrS[i] = 1.f;
    if (tid < 2) flags[tid] = 0;
    int tkq[2 * NT];
#pragma unroll
    for (int ii = 0; ii < 2 * NT; ++ii) tkq[ii] = (w + 4 * ii) / G;

    // Q: B operand of GEMM1 in int8 (k = d, n = query): a single scale per query (max |Q| over the whole head), three int8 digits.
    // Thread m of a q8_0 block holds d 4m..4m+3 (b0) and 16+4m..16+4m+3 (b1), like the A operand (raw K).
    for (int ii = 0; ii < 2 * NT; ++ii) {                                   // maximum per query: warp = query
        const int q = w + 4 * ii;
        float mx = 0.f;
        if (q < Nq) {
            const int tk = q / G, h = q - tk * G;
            const float * qr = reinterpret_cast<const float *>(p.Q + (size_t) tk * p.nbq1 + (size_t) (kvh * G + h) * p.nbq2);
            for (int i = lane; i < D; i += 32) mx = fmaxf(mx, fabsf(qr[i]));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        }
        if (lane == 0 && q < NQP) qamax[q] = mx;
    }
    __syncthreads();
    float dq[NT][2];                                                         // dequantization scale of the columns of C
    uint32_t qp[NB / 2][NT][3][2];                                           // int8 digits of Q for the warp's blocks
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        dq[nt][0] = qamax[8 * nt + 2 * m]     * (1.f / 8388607.f);
        dq[nt][1] = qamax[8 * nt + 2 * m + 1] * (1.f / 8388607.f);
        const int n = 8 * nt + r;
        const float mxn = qamax[n];
        const float inv = mxn > 0.f ? 8388607.f / mxn : 0.f;
#pragma unroll
        for (int bi = 0; bi < NB / 2; ++bi) {
            const int blk = dh * (NB / 2) + bi;
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                float f[4] = {0.f, 0.f, 0.f, 0.f};
                if (n < Nq) {
                    const int tk = n / G, h = n - tk * G;
                    const float * qr = reinterpret_cast<const float *>(p.Q + (size_t) tk * p.nbq1 + (size_t) (kvh * G + h) * p.nbq2)
                                       + 32 * blk + 16 * hf + 4 * m;
                    f[0] = qr[0]; f[1] = qr[1]; f[2] = qr[2]; f[3] = qr[3];
                }
                int a[4], b2[4], c[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) fq2_q3(f[e], inv, a[e], b2[e], c[e]);
                qp[bi][nt][0][hf] = fq2_pk4(a[0], a[1], a[2], a[3]);
                qp[bi][nt][1][hf] = fq2_pk4(b2[0], b2[1], b2[2], b2[3]);
                qp[bi][nt][2][hf] = fq2_pk4(c[0], c[1], c[2], c[3]);
            }
        }
    }

    float mq[2 * NT], ps[2 * NT];            // softmax state of the warp's queries (q = w + 4 ii)
#pragma unroll
    for (int i = 0; i < 2 * NT; ++i) { mq[i] = -FLT_MAX; ps[i] = 0.f; }
    float acc[BPW][2][NT][4];                // O^T: d = 32 (w BPW + b) + 4r + 2t (+1 for c >= 2), q = 8 nt + 2m + (c & 1)
#pragma unroll
    for (int b = 0; b < BPW; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[b][t][nt][c] = 0.f;

    // prefetch of a tile (K and V, 32 positions) into stage tile % STAGES.
    // Dense cache (contiguous rows, D = 256: row in smem = row in memory): a tile is a linear copy of 32 rows,
    // affine addresses per thread. Otherwise (D = 128, spaced rows, last partial tile): row by row, with the
    // rows beyond the chunk replaced by the last valid one (P = 0). No per-thread tables: the registers are needed elsewhere.
    constexpr bool LINEAR = (SP == geo::ROWB) && CS == 16;
    const bool dense = LINEAR && p.nbk1 == (size_t) geo::ROWB && p.nbv1 == (size_t) geo::ROWB;
    const char * kbase = p.K + (size_t) kvh * p.nbk2 + lane * CS, * vbase = p.V + (size_t) kvh * p.nbv2 + lane * CS;
    auto issue = [&](int tile) {
        char * sb = smem + (size_t) (tile % STAGES) * SB;
        const int tb = t0 + tile * FQ2_TILE;
        if (LINEAR && dense && tb + FQ2_TILE <= t1) {
            const char * ksrc = p.K + (size_t) kvh * p.nbk2 + (size_t) tb * geo::ROWB;
            const char * vsrc = p.V + (size_t) kvh * p.nbv2 + (size_t) tb * geo::ROWB;
#pragma unroll
            for (int j = 0; j < (FQ2_TILE * NCH + FQ2_THREADS - 1) / FQ2_THREADS; ++j) {
                const int c = tid + j * FQ2_THREADS;
                if (c < FQ2_TILE * NCH) {
                    fq2_cp_async<CS>(sb + c * CS, ksrc + c * CS);
                    fq2_cp_async<CS>(sb + FQ2_TILE * SP + c * CS, vsrc + c * CS);
                }
            }
        } else if (lane < NCH) {   // row by row: warp w copies rows w, w+4, ... (the NCH <= 17 lanes cover one row)
            char * dst = sb + lane * CS;
            if (tb + FQ2_TILE <= t1) {
                const char * kp = kbase + (size_t) (tb + w) * p.nbk1, * vp = vbase + (size_t) (tb + w) * p.nbv1;
                const size_t sk = (size_t) FQ2_WARPS * p.nbk1, sv = (size_t) FQ2_WARPS * p.nbv1;
#pragma unroll
                for (int i = 0; i < FQ2_TILE / FQ2_WARPS; ++i) {
                    fq2_cp_async<CS>(dst + (size_t) (w + FQ2_WARPS * i) * SP, kp + i * sk);
                    fq2_cp_async<CS>(dst + (size_t) (FQ2_TILE + w + FQ2_WARPS * i) * SP, vp + i * sv);
                }
            } else {   // last partial tile: the rows beyond the chunk become the last valid one (P = 0)
                for (int i = 0; i < FQ2_TILE / FQ2_WARPS; ++i) {
                    const int row = w + FQ2_WARPS * i, t = min(tb + row, t1 - 1);
                    fq2_cp_async<CS>(dst + (size_t) row * SP, kbase + (size_t) t * p.nbk1);
                    fq2_cp_async<CS>(dst + (size_t) (FQ2_TILE + row) * SP, vbase + (size_t) t * p.nbv1);
                }
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < ntiles) issue(s); fq2_cp_commit(); }

    for (int i = 0; i < ntiles; ++i) {
        fq2_cp_wait<STAGES - 2>();
        __syncthreads();                                         // A: tile i arrived; the stage to fill is free
        if (i + STAGES - 1 < ntiles) issue(i + STAGES - 1);
        fq2_cp_commit();
        int * rescale = flags + (i & 1), * rescale_next = flags + ((i + 1) & 1);
        const char * Ks = smem + (size_t) (i % STAGES) * SB;
        const char * Vs = Ks + FQ2_TILE * SP;

        // mask (per query token) for position = lane
        float mk[4];
        {
            const int pos = t0 + i * FQ2_TILE + lane;
#pragma unroll
            for (int tk = 0; tk < 4; ++tk) {
                mk[tk] = pos >= t1 ? -INFINITY
                       : (p.mask && tk < p.nq ? __half2float(*reinterpret_cast<const half *>(p.mask + (size_t) tk * p.nbm1 + (size_t) pos * 2)) : 0.f);
            }
        }

        // ---- GEMM1: S^T[pos, q] = dQ[q] sum_blocks dK[pos, blk] (K_int[pos, d] . Q_int[q, d]), warp = 16 positions x half of D.
        // Exact int8 x int8 -> int32 product (three digits of Q), raw K from the cache; scales in fp32.
        float sacc[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f; }
        {
            const char * rowA = Ks + (size_t) (16 * mt + r) * SP;
            const char * rowB = rowA + 8 * SP;
#pragma unroll
            for (int bi = 0; bi < NB / 2; ++bi) {
                const int blk = dh * (NB / 2) + bi;
                const int off = 34 * blk + 2 + 4 * m;                      // 4 aligned bytes if blk is odd
                uint32_t a0, a1, a2, a3;
                if (bi & 1) {
                    a0 = *reinterpret_cast<const uint32_t *>(rowA + off); a2 = *reinterpret_cast<const uint32_t *>(rowA + off + 16);
                    a1 = *reinterpret_cast<const uint32_t *>(rowB + off); a3 = *reinterpret_cast<const uint32_t *>(rowB + off + 16);
                } else {
                    a0 = fq2_ld4(rowA + off); a2 = fq2_ld4(rowA + off + 16);
                    a1 = fq2_ld4(rowB + off); a3 = fq2_ld4(rowB + off + 16);
                }
                const float dA = __half2float(__ushort_as_half((unsigned short) fq2_lds16(rowA + 34 * blk)));
                const float dB = __half2float(__ushort_as_half((unsigned short) fq2_lds16(rowB + 34 * blk)));
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    int c1[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0}, c3[4] = {0, 0, 0, 0};
                    fq2_mma_s8(c1, a0, a1, a2, a3, qp[bi][nt][0][0], qp[bi][nt][0][1]);
                    fq2_mma_s8u8(c2, a0, a1, a2, a3, qp[bi][nt][1][0], qp[bi][nt][1][1]);
                    fq2_mma_s8u8(c3, a0, a1, a2, a3, qp[bi][nt][2][0], qp[bi][nt][2][1]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float sb = fmaf((float) c1[e], 65536.f, (float) (c2[e] * 256 + c3[e]));
                        sacc[nt][e] = fmaf(sb, e < 2 ? dA : dB, sacc[nt][e]);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                sacc[nt][0] *= dq[nt][0]; sacc[nt][1] *= dq[nt][1]; sacc[nt][2] *= dq[nt][0]; sacc[nt][3] *= dq[nt][1];
            }
        }
        {
            float * spw = Sp + dh * NQP * FQ2_SPP;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int q0 = 8 * nt + 2 * m, pos = 16 * mt + r;
                spw[q0 * FQ2_SPP + pos]           = sacc[nt][0];
                spw[(q0 + 1) * FQ2_SPP + pos]     = sacc[nt][1];
                spw[q0 * FQ2_SPP + pos + 8]       = sacc[nt][2];
                spw[(q0 + 1) * FQ2_SPP + pos + 8] = sacc[nt][3];
            }
        }
        __syncthreads();                                         // B: partial S in smem

        // ---- inline softmax: warp = query (q = w + 4 ii), lane = position. The reference maximum is updated
        // only if some position exceeds it by more than FQ2_LAZY: normally no reductions.
#pragma unroll
        for (int ii = 0; ii < 2 * NT; ++ii) {
            const int q = w + 4 * ii;
            if (q >= Nq) continue;                               // uniform across the warp
            const float s = Sp[q * FQ2_SPP + lane] + Sp[(NQP + q) * FQ2_SPP + lane];
            const int tk = tkq[ii];
            const float mkv = tk == 0 ? mk[0] : tk == 1 ? mk[1] : tk == 2 ? mk[2] : mk[3];
            const float x = fmaf(s, p.scale, mkv);
            float corr = 1.f;
            if (__any_sync(0xffffffffu, x > mq[ii] + FQ2_LAZY)) {
                float mx = x;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                const float mn = fmaxf(mq[ii], mx);
                corr = __expf(mq[ii] - mn);
                mq[ii] = mn;
                if (lane == 0) *rescale = 1;
            }
            const float pv = __expf(x - mq[ii]);
            ps[ii] = fmaf(ps[ii], corr, pv);
            Pf[q * FQ2_PF + lane] = pv;
            if (lane == 0) corrS[q] = corr;
        }
        if (tid == 0) rescale_next[0] = 0;
        __syncthreads();                                         // C: P and corrections in smem

        // ---- GEMM2: O^T[d, q] += V^T[d, pos] P^T[pos, q], warp = 64 columns of D (BPW q8_0 blocks)
        if (*rescale) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float2 c = *reinterpret_cast<const float2 *>(corrS + 8 * nt + 2 * m);
#pragma unroll
                for (int b = 0; b < BPW; ++b)
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        acc[b][t][nt][0] *= c.x; acc[b][t][nt][1] *= c.y;
                        acc[b][t][nt][2] *= c.x; acc[b][t][nt][3] *= c.y;
                    }
            }
        }
        // The tile is accumulated in a fresh accumulator and added in fp32 (rounded) to O: tensor core accumulation truncates,
        // and on an accumulator that lives for hundreds of tiles the defect adds up (1e-5 at 128k tokens).
#pragma unroll
        for (int b = 0; b < BPW; ++b) {
            const int blk = w * BPW + b;
            float tl[2][NT][4];
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) { tl[t][nt][0] = tl[t][nt][1] = tl[t][nt][2] = tl[t][nt][3] = 0.f; }
#pragma unroll
            for (int ks2 = 0; ks2 < 2; ++ks2) {
                float pf[NT][4];                                          // P[q = 8 nt + r][pos 16 ks2 + 2m, +1, +8, +9]
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const float2 u = *reinterpret_cast<const float2 *>(Pf + (8 * nt + r) * FQ2_PF + 16 * ks2 + 2 * m);
                    const float2 v = *reinterpret_cast<const float2 *>(Pf + (8 * nt + r) * FQ2_PF + 16 * ks2 + 2 * m + 8);
                    pf[nt][0] = u.x; pf[nt][1] = u.y; pf[nt][2] = v.x; pf[nt][3] = v.y;
                }
                const char * rA = Vs + (size_t) (16 * ks2 + 2 * m) * SP;     // positions pA, pA+1, pA+8, pA+9
                const char * rB = rA + SP, * rC = rA + 8 * SP, * rD = rA + 9 * SP;
                const int off = 34 * blk + 2 + 4 * r;
                const uint32_t wA = fq2_ld4(rA + off) ^ 0x80808080u, wB = fq2_ld4(rB + off) ^ 0x80808080u;
                const uint32_t wC = fq2_ld4(rC + off) ^ 0x80808080u, wD = fq2_ld4(rD + off) ^ 0x80808080u;
                // the V scale depends on the position (the contraction axis): it goes into P,
                // P'[q, pos] = P[q, pos] dV[pos, blk] in fp32, then hi + lo
                const float dA = (float) (1 << FQ2_SSHIFT) * __half2float(__ushort_as_half((unsigned short) fq2_lds16(rA + 34 * blk)));
                const float dB = (float) (1 << FQ2_SSHIFT) * __half2float(__ushort_as_half((unsigned short) fq2_lds16(rB + 34 * blk)));
                const float dC = (float) (1 << FQ2_SSHIFT) * __half2float(__ushort_as_half((unsigned short) fq2_lds16(rC + 34 * blk)));
                const float dD = (float) (1 << FQ2_SSHIFT) * __half2float(__ushort_as_half((unsigned short) fq2_lds16(rD + 34 * blk)));
                // 4x4 byte transpose: X = {A0,B0,A1,B1}, Y = {A2,B2,A3,B3} (columns d, d+1, d+2, d+3)
                const uint32_t X1 = fq2_prmt(wA, wB, 0x5140), Y1 = fq2_prmt(wA, wB, 0x7362);
                const uint32_t X2 = fq2_prmt(wC, wD, 0x5140), Y2 = fq2_prmt(wC, wD, 0x7362);
                uint32_t av[2][4];
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const uint32_t u1 = t ? Y1 : X1, u2 = t ? Y2 : X2;
                    av[t][0] = fq2_dqx(fq2_prmt(u1, 0x64646464u, 0x4140));   // row r,   low k
                    av[t][1] = fq2_dqx(fq2_prmt(u1, 0x64646464u, 0x4342));   // row r+8, low k
                    av[t][2] = fq2_dqx(fq2_prmt(u2, 0x64646464u, 0x4140));   // row r,   high k
                    av[t][3] = fq2_dqx(fq2_prmt(u2, 0x64646464u, 0x4342));   // row r+8, high k
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint32_t bh0, bl0, bh1, bl1;
                    fq2_split(pf[nt][0] * dA, pf[nt][1] * dB, bh0, bl0);
                    fq2_split(pf[nt][2] * dC, pf[nt][3] * dD, bh1, bl1);
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        fq2_mma(tl[t][nt], av[t][0], av[t][1], av[t][2], av[t][3], bh0, bh1);
                        fq2_mma(tl[t][nt], av[t][0], av[t][1], av[t][2], av[t][3], bl0, bl1);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[b][t][nt][c] += tl[t][nt][c];
        }
    }
    fq2_cp_wait<0>();

    // ---- chunk output: O not normalized (relative to the chunk maximum) and (m, s) per query
    const size_t nqt = (size_t) p.nq * G;
#pragma unroll
    for (int b = 0; b < BPW; ++b)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int q = 8 * nt + 2 * m + (c & 1);
                    const int d = 32 * (w * BPW + b) + 4 * r + 2 * t + (c >> 1);
                    if (q < Nq) p.tmp_acc[(((size_t) kvh * nqt + q) * nch + ch) * D + d] = acc[b][t][nt][c] * (1.f / (float) (1 << FQ2_SSHIFT));
                }
#pragma unroll
    for (int ii = 0; ii < 2 * NT; ++ii) {
        const int q = w + 4 * ii;
        if (q >= Nq) continue;
        float s = ps[ii];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) p.tmp_ms[((size_t) kvh * nqt + q) * nch + ch] = make_float2(mq[ii], s);
    }
#endif
}

template <int D>
static __global__ void fq2_combine(const float * __restrict__ tmp_acc, const float2 * __restrict__ tmp_ms,
                                   float * __restrict__ dst, int nch, int nqt, int G, int nhead) {
    const int qi = blockIdx.x, d = threadIdx.x;
    const int kvh = qi / nqt, n = qi - kvh * nqt, tk = n / G, h = n - tk * G;
    float M = -FLT_MAX;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, tmp_ms[(size_t) qi * nch + c].x);
    float S = 0.f, A = 0.f;
    for (int c = 0; c < nch; ++c) {
        const float2 ms = tmp_ms[(size_t) qi * nch + c];
        const float f = __expf(ms.x - M);
        S += ms.y * f;
        A += tmp_acc[((size_t) qi * nch + c) * D + d] * f;
    }
    dst[((size_t) tk * nhead + kvh * G + h) * D + d] = S > 0.f ? A / S : 0.f;
}

// Context chunks: ~2 blocks per SM (two blocks of <= 48 KB of smem per SM), multiples of a tile.
static inline void fq2_plan(int n_kv, int nkvh, int nsm, int * chunk, int * nch) {
    const int target = 2 * nsm;
    int n = (target + nkvh - 1) / nkvh;
    const int nmax = (n_kv + 255) / 256;                  // chunks of at least 256 positions
    n = n < nmax ? n : nmax;
    n = n < 1 ? 1 : n;
    int c = (n_kv + n - 1) / n;
    c = (c + FQ2_TILE - 1) / FQ2_TILE * FQ2_TILE;
    *chunk = c;
    *nch = (n_kv + c - 1) / c;
}

template <int D, int NT, int STAGES>
static void fq2_launch_nt(const fq2_params & p, int nkvh, int nch, cudaStream_t stream) {
    constexpr size_t smem = fq2_smem_bytes<D, NT, STAGES>();
    if (smem > 48 * 1024) cudaFuncSetAttribute(fq2_partial<D, NT, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    fq2_partial<D, NT, STAGES><<<dim3(nkvh, nch), FQ2_THREADS, smem, stream>>>(p);
    fq2_combine<D><<<nkvh * p.nq * p.G, D, 0, stream>>>(p.tmp_acc, p.tmp_ms, p.dst, nch, p.nq * p.G, p.G, p.nhead);
}

#ifndef FQ2_STAGES
#define FQ2_STAGES 2
#endif

// Launch with raw pointers (used by the engine and by the bench). tmp_acc: nkvh*nq*G*nch*D float; tmp_ms: nkvh*nq*G*nch float2.
template <int D>
static void fq2_run(const fq2_params & p, int nkvh, int nch, cudaStream_t stream) {
    const int nqt = p.nq * p.G;
    if      (nqt <= 8)  fq2_launch_nt<D, 1, FQ2_STAGES>(p, nkvh, nch, stream);
    else if (nqt <= 16) fq2_launch_nt<D, 2, FQ2_STAGES>(p, nkvh, nch, stream);
    else if (nqt <= 24) fq2_launch_nt<D, 3, FQ2_STAGES>(p, nkvh, nch, stream);
    else                fq2_launch_nt<D, 4, FQ2_STAGES>(p, nkvh, nch, stream);
}

// ============================ lato host (ggml) ============================
#ifndef FQ_STANDALONE

// The Ampere kernel wants K/V rows aligned to the cp.async size (16 bytes for D = 256, 8 for D = 128)
static bool fq2_ok(const ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1], * V = dst->src[2];
    static const bool old_kernel = getenv("FA_Q8_OLD") != nullptr;   // sentinel: the previous fp32 kernel, 1 token
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc < CC_AMPERE || old_kernel) return false;
    const size_t cs = K->ne[0] == 256 ? 16 : 8;
    return (uintptr_t) K->data % cs == 0 && (uintptr_t) V->data % cs == 0 &&
           K->nb[1] % cs == 0 && K->nb[2] % cs == 0 && V->nb[1] % cs == 0 && V->nb[2] % cs == 0;
}

bool ggml_cuda_fattn_q8_gqa_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || dst->type != GGML_TYPE_F32) return false;
    if (Q->ne[1] < 1 || Q->ne[1] > 4 || Q->ne[3] != 1 || K->ne[3] != 1) return false;
    if (Q->ne[1] != 1 && !fq2_ok(dst)) return false;                  // 2..4 query tokens: Ampere kernel only
    if (K->ne[0] != V->ne[0] || Q->ne[0] != K->ne[0] || (K->ne[0] != 256 && K->ne[0] != 128)) return false;
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > 8 || V->ne[2] != K->ne[2]) return false;
    if (dst->src[4] != nullptr) return false;                         // sink
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] * mask->ne[3] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) return false;
    float max_bias = 0.f, softcap = 0.f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.f || softcap != 0.f || dst->op_params[4] != 0) return false;   // ALiBi, softcap, SWA
    return ggml_is_contiguous(dst) && Q->nb[0] == sizeof(float);
}

template <int D, int G>
static void fq_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int n_kv = (int) K->ne[1], nkvh = (int) K->ne[2];
    // chunks of 128 positions (16 per warp): many blocks in flight to hide the read latency
    const int chunk = 128;
    const int nch = (n_kv + chunk - 1) / chunk;
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    ggml_cuda_pool_alloc<float>  acc(ctx.pool(), (size_t) nkvh * G * nch * D);
    ggml_cuda_pool_alloc<float2> ms(ctx.pool(), (size_t) nkvh * G * nch);
    fq_partial<D, G><<<dim3(nkvh, nch), FQ_WARPS * 32, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const half *) mask->data : nullptr, acc.get(), ms.get(), scale, n_kv, chunk,
        Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2]);
    fq_combine<D><<<nkvh * G, D, 0, stream>>>(acc.get(), ms.get(), (float *) dst->data, nch);
    CUDA_CHECK(cudaGetLastError());
}

static void fq2_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3];
    const int D = (int) K->ne[0], nkvh = (int) K->ne[2];
    fq2_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.n_kv = (int) K->ne[1]; p.nq = (int) Q->ne[1]; p.nhead = (int) Q->ne[2]; p.G = p.nhead / nkvh;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2];
    p.nbm1 = mask ? mask->nb[1] : 0;
    p.dst = (float *) dst->data;
    int nch;
    fq2_plan(p.n_kv, nkvh, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm, &p.chunk, &nch);
    const size_t nqt = (size_t) p.nq * p.G;
    ggml_cuda_pool_alloc<float>  acc(ctx.pool(), (size_t) nkvh * nqt * nch * D);
    ggml_cuda_pool_alloc<float2> ms(ctx.pool(), (size_t) nkvh * nqt * nch);
    p.tmp_acc = acc.get(); p.tmp_ms = ms.get();
    if (D == 256) fq2_run<256>(p, nkvh, nch, ctx.stream());
    else          fq2_run<128>(p, nkvh, nch, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fattn_q8_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1];
    if (fq2_ok(dst)) { fq2_launch(ctx, dst); return; }
    const int G = (int) (Q->ne[2] / K->ne[2]);
    const int D = (int) K->ne[0];
#define FQ_CASE(DD, GG) if (D == DD && G == GG) { fq_launch<DD, GG>(ctx, dst); return; }
    FQ_CASE(256, 1) FQ_CASE(256, 2) FQ_CASE(256, 3) FQ_CASE(256, 4) FQ_CASE(256, 5) FQ_CASE(256, 6) FQ_CASE(256, 7) FQ_CASE(256, 8)
    FQ_CASE(128, 1) FQ_CASE(128, 2) FQ_CASE(128, 3) FQ_CASE(128, 4) FQ_CASE(128, 5) FQ_CASE(128, 6) FQ_CASE(128, 7) FQ_CASE(128, 8)
#undef FQ_CASE
    GGML_ABORT("fattn-q8-gqa: unexpected case");
}

#endif  // FQ_STANDALONE
