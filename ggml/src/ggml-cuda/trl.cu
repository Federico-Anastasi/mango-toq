// TRL: GEMV for one token at a time on trellis weights with the mul1 codebook. See trl.cuh.
//
// Weight layout: ggml/src/ggml-trl.h (bands of 64 rows: 64 fp32 scales + K/4 tiles of 32*b bytes).
// The value of a weight is s_n * (bytesum(state * 0x83DCD12D) - 510) / 147.80: the integer part
// (bytesum - 510) multiplies the int8 activation with ONE dp4a, the rest (row scale, activation
// block scale, 1/147.80) is applied in floating point.
//
// The kernel is persistent: each warp works on its own on "items" = (band of 64 rows, slice of K),
// with no synchronization in the loop; final sum across lanes with shfl and across slices with atomicAdd on y.
#include "trl.cuh"
#include "convert.cuh"
#include "../ggml-trl.h"
#include "trl_gemm_nt.cuh"   // fused GEMMs for GPUs without tensor cores (Pascal, Turing TU116)
#include "trl_mmq3.cuh"      // int8 GEMM with asynchronous stages for Ampere (prompt reading)

#define TRL_QB 32   // activations: blocks of 32 int8 values with one scale

static __device__ __forceinline__ int trl_dp4a_us(uint32_t a, int b, int c) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define TRL_USE_BFE 1     // Pascal: native bfe +5-7% (measured on the 1080)
#endif

static __device__ __forceinline__ uint32_t trl_bfe14(uint32_t w, int lsb) {
    uint32_t r;
    asm("bfe.u32 %0, %1, %2, 14;" : "=r"(r) : "r"(w), "r"(lsb));
    return r;
}

struct trl_args {
    const uint32_t * blob;
    const int      * aq;     // [NC][K/4] int8x4
    const float    * ad;     // [NC][K/32]
    const int      * asum;   // [NC][K/32]
    float          * y;      // [NC][N]
    int N, K, ks;
};

// NC columns (tokens) together: each weight is read and decoded only once and serves NC dp4a.
template <int B, int WPL, int U, int NC>
static __global__ void __launch_bounds__(256) trl_gemv(trl_args a) {
    constexpr int RUN = 8;                     // 8 tiles = 32 columns = one activation block
    constexpr int SPL = 32 * WPL / B;          // symbols per lane
    constexpr int RPL = SPL / 4;               // rows per lane
    constexpr int LPT = 64 / RPL;              // lanes per tile
    constexpr int TPS = 32 / LPT;              // tiles per warp step
    static_assert((32 * WPL) % B == 0 && SPL % 4 == 0 && LPT * WPL == 8 * B, "geometry");

    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nw = gridDim.x * (blockDim.x >> 5);
    const int g = lane / LPT, li = lane % LPT;

    const int ntiles = a.K >> 2;
    const int nrg = a.N >> 6;
    const size_t grpWords = (size_t)(64 + (size_t)ntiles * 8 * B);
    constexpr int chunkTiles = TPS * RUN;
    const int nchunks = ntiles / chunkTiles;
    const int ks = a.ks;
    const int qstr = a.K >> 2, bstr = a.K / TRL_QB;

    for (int item = gw; item < nrg * ks; item += nw) {
        const int rg = item / ks, s = item - (item / ks) * ks;
        const int c0 = (int)((long long)s * nchunks / ks), c1 = (int)((long long)(s + 1) * nchunks / ks);
        const int nsteps = (c1 - c0) * RUN;
        if (nsteps <= 0) {
            continue;
        }
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

        struct Buf { uint32_t w[U][WPL]; int a4[U][NC]; };
        Buf cur, nxt;

        auto load = [&](Buf & bf, int n0) {
#pragma unroll
            for (int j = 0; j < U; ++j) {
                const int n = n0 + j;
                if (n < nsteps) {
                    const int chunk = c0 + n / RUN, r = n % RUN;
                    const int tile = chunk * chunkTiles + g * RUN + r;
                    const uint32_t * p = tw + (size_t)tile * (8 * B) + li * WPL;
                    if constexpr (WPL % 4 == 0) {
#pragma unroll
                        for (int q = 0; q < WPL; q += 4) {
                            uint4 v = __ldcs(reinterpret_cast<const uint4 *>(p + q));
                            bf.w[j][q] = v.x; bf.w[j][q + 1] = v.y; bf.w[j][q + 2] = v.z; bf.w[j][q + 3] = v.w;
                        }
                    } else if constexpr (WPL % 2 == 0) {
#pragma unroll
                        for (int q = 0; q < WPL; q += 2) {
                            uint2 v = __ldcs(reinterpret_cast<const uint2 *>(p + q));
                            bf.w[j][q] = v.x; bf.w[j][q + 1] = v.y;
                        }
                    } else {
#pragma unroll
                        for (int q = 0; q < WPL; ++q) {
                            bf.w[j][q] = __ldcs(p + q);
                        }
                    }
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        bf.a4[j][c] = __ldg(a.aq + c * qstr + tile);
                    }
                }
            }
        };

        auto compute = [&](const Buf & bf, int n0) {
#pragma unroll
            for (int j = 0; j < U; ++j) {
                const int n = n0 + j;
                if (n < nsteps) {
                    uint32_t S[WPL + 1];
#pragma unroll
                    for (int q = 0; q < WPL; ++q) {
                        S[q + 1] = bf.w[j][q];
                    }
                    const uint32_t pv = __shfl_up_sync(0xffffffffu, S[WPL], 1, LPT);
                    S[0] = (li == 0) ? 0u : pv;          // the state restarts from 0 at every tile
                    int rep[NC][4];
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const int a4 = bf.a4[j][c];
                        rep[c][0] = __byte_perm(a4, 0, 0x0000);
                        rep[c][1] = __byte_perm(a4, 0, 0x1111);
                        rep[c][2] = __byte_perm(a4, 0, 0x2222);
                        rep[c][3] = __byte_perm(a4, 0, 0x3333);
                    }
#pragma unroll
                    for (int t = 0; t < SPL; ++t) {
                        const int E = 32 + B * (t + 1);
                        const int st = E - TRL_L;
                        const int wi = st >> 5, o = st & 31;
#ifdef TRL_USE_BFE
                        uint32_t state;
                        if (o <= 18) {
                            state = trl_bfe14(S[wi], 32 - TRL_L - o);
                        } else {
                            state = __funnelshift_l(S[wi + 1], S[wi], o) >> 18;
                        }
#else
                        const uint32_t nx = (wi < WPL) ? S[wi + 1] : 0u;
                        const uint32_t state = __funnelshift_l(nx, S[wi], o) >> 18;
#endif
                        const uint32_t wv = state * TRL_MUL1_K;
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            acci[t >> 2][c] = trl_dp4a_us(wv, rep[c][t & 3], acci[t >> 2][c]);
                        }
                    }
                    if (((n + 1) % RUN) == 0) {
                        const int chunk = c0 + n / RUN;
                        const int blk = chunk * chunkTiles / 8 + g;
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            const float dA = __ldg(a.ad + c * bstr + blk);
                            const int sA = __ldg(a.asum + c * bstr + blk);
#pragma unroll
                            for (int r = 0; r < RPL; ++r) {
                                acc[r][c] = fmaf(dA, (float)(acci[r][c] - 510 * sA), acc[r][c]);
                                acci[r][c] = 0;
                            }
                        }
                    }
                }
            }
        };

        load(cur, 0);
        for (int n0 = 0; n0 < nsteps; n0 += U) {
            if (n0 + U < nsteps) {
                load(nxt, n0 + U);
            }
            compute(cur, n0);
            cur = nxt;
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
                    const float val = acc[r][c] * scr;
                    if (ks == 1) {
                        a.y[(size_t) c * a.N + row] = val;
                    } else {
                        atomicAdd(a.y + (size_t) c * a.N + row, val);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// A few tokens together (2..8 columns) on the int8 tensor units, Ampere+, 2-bit weights.
// The GEMV above pays ~3 integer decode operations per weight (state extraction, product by K)
// and one dp4a PER COLUMN: with 4 columns 7 operations per weight, limited by integer arithmetic (measured 30/10: 44 ms
// per step against 19 at one column). Here the decoded word wv = state*K, whose 4 unsigned bytes summed give
// the weight + 510, is directly an A register of mma.m16n8k32.u8.s8: k = the 4 bytes of the weight; B = the int8
// activation repeated 4 times (the same trick as dp4a, in matrix form). One mma covers 16 rows x 8 weights x 8 tokens, and
// the cost per weight stays the decode alone, from 1 to 8 columns. Integer sums identical to the GEMV (u8 x s8 products
// exact in int32): same result up to the order of the float sums.
// Geometry: one warp per (band of 64 rows, slice of K). Lane = (g = lane/4, q = lane%4): g decodes rows
// 8g..8g+7 of every tile (2 words of the tile + the last word of the previous group, taken with a shuffle), q the
// k = q of the tile. mma p (0..3) of the block of 8 tiles: quad q <- tile 2p, quad q+4 <- tile 2p+1; mma per row
// pair rgp (0..3): A row g <- 8g+2rgp, A row g+8 <- 8g+2rgp+1. B: column n = token g. C: rows as A,
// columns 2q, 2q+1. TRL_TC=0 turns it off; TRL_TC_CHECK=1 compares every call with the usual GEMV.
// ---------------------------------------------------------------------------------------------------------
static __device__ __forceinline__ void trl_mma_u8s8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= CC_AMPERE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a0); GGML_UNUSED(a1); GGML_UNUSED(a2); GGML_UNUSED(a3); GGML_UNUSED(b0); GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif
}

// state (14 bit) of the symbol (row 8g+I, k = q) of the tile, from words w0 (2g-1, or 0 for g = 0), w1 (2g), w2 (2g+1):
// the window starts at bit o = 8I + 20 + 2q counted from the top bit of w0 (bit 0 = MSB, as in the GEMV)
template <int I>
static __device__ __forceinline__ uint32_t trl_tc_state(uint32_t w0, uint32_t w1, uint32_t w2, int q2) {
    constexpr int O = 8 * I + 20;          // o = O + q2, q2 in {0, 2, 4, 6}
    if constexpr (O + 6 < 32) {
        return __funnelshift_l(w1, w0, O + q2) >> 18;
    } else if constexpr (O >= 32 && O + 6 < 64) {
        return __funnelshift_l(w2, w1, O - 32 + q2) >> 18;
    } else if constexpr (O >= 64) {
        return (w2 << (O - 64 + q2)) >> 18;
    } else if constexpr (O < 32) {
        const int o = O + q2;
        return o < 32 ? __funnelshift_l(w1, w0, o) >> 18 : __funnelshift_l(w2, w1, o - 32) >> 18;
    } else {
        const int o = O + q2;
        return o < 64 ? __funnelshift_l(w2, w1, o - 32) >> 18 : (w2 << (o - 64)) >> 18;
    }
}

// the mma of a row pair (8g+2RGP, 8g+2RGP+1) for a tile pair (quad q <- tile a, quad q+4 <- tile b)
template <int RGP>
static __device__ __forceinline__ void trl_tc_mma(int C[4], uint32_t w0a, uint2 wa, uint32_t w0b, uint2 wb,
                                                  uint32_t repa, uint32_t repb, int q2) {
    const uint32_t a0 = trl_tc_state<2 * RGP    >(w0a, wa.x, wa.y, q2) * TRL_MUL1_K;
    const uint32_t a1 = trl_tc_state<2 * RGP + 1>(w0a, wa.x, wa.y, q2) * TRL_MUL1_K;
    const uint32_t a2 = trl_tc_state<2 * RGP    >(w0b, wb.x, wb.y, q2) * TRL_MUL1_K;
    const uint32_t a3 = trl_tc_state<2 * RGP + 1>(w0b, wb.x, wb.y, q2) * TRL_MUL1_K;
    trl_mma_u8s8(C, a0, a1, a2, a3, repa, repb);
}

struct trl_tc_pair { uint2 wa, wb; int aa, ab; };   // words 2g, 2g+1 of two tiles and their activations (token g)

static __global__ void __launch_bounds__(256, 2) trl_gemv_tc(trl_args a, int ncols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= CC_AMPERE
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nw = gridDim.x * (blockDim.x >> 5);
    const int g = lane >> 2, q = lane & 3, q2 = 2 * q;
    const uint32_t sel = 0x1111u * (uint32_t) q;           // byte_perm: byte q repeated 4 times

    const int ntiles = a.K >> 2;
    const int nrg = a.N >> 6;
    const size_t grpWords = 64 + (size_t) ntiles * 16;      // B = 2: 16 words per tile
    const int nblk = ntiles / 8;                            // activation blocks (32 k)
    const int ks = a.ks;
    const int qstr = a.K >> 2, bstr = a.K / TRL_QB;
    const bool tok = g < ncols;                             // column B = token g
    const int ca = 2 * q, cb = 2 * q + 1;                   // C columns of this thread
    const bool va = ca < ncols, vb = cb < ncols;
    const int * aqg = a.aq + (size_t) (tok ? g : 0) * qstr;

    for (int item = gw; item < nrg * ks; item += nw) {
        const int rg = item / ks, s = item - rg * ks;
        const int b0 = (int) ((long long) s * nblk / ks), b1 = (int) ((long long) (s + 1) * nblk / ks);
        if (b1 <= b0) {
            continue;
        }
        const uint32_t * tw = a.blob + (size_t) rg * grpWords + 64 + 2 * g;

        float acc[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[r][c] = 0.f;
            }
        }

        auto load = [&](trl_tc_pair (&P)[4], int b) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int ta = b * 8 + 2 * p;
                P[p].wa = __ldcs(reinterpret_cast<const uint2 *>(tw + (size_t) ta * 16));
                P[p].wb = __ldcs(reinterpret_cast<const uint2 *>(tw + (size_t) (ta + 1) * 16));
                P[p].aa = tok ? __ldg(aqg + ta) : 0;
                P[p].ab = tok ? __ldg(aqg + ta + 1) : 0;
            }
        };

        trl_tc_pair cur[4], nxt[4];
        load(cur, b0);
        for (int b = b0; b < b1; ++b) {
            if (b + 1 < b1) {
                load(nxt, b + 1);
            }
            int C[4][4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    C[r][c] = 0;
                }
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const uint32_t pa = __shfl_up_sync(0xffffffffu, cur[p].wa.y, 4);
                const uint32_t pb = __shfl_up_sync(0xffffffffu, cur[p].wb.y, 4);
                const uint32_t w0a = g == 0 ? 0u : pa;      // the state restarts from 0 at every tile
                const uint32_t w0b = g == 0 ? 0u : pb;
                const uint32_t repa = __byte_perm(cur[p].aa, 0, sel);
                const uint32_t repb = __byte_perm(cur[p].ab, 0, sel);
                trl_tc_mma<0>(C[0], w0a, cur[p].wa, w0b, cur[p].wb, repa, repb, q2);
                trl_tc_mma<1>(C[1], w0a, cur[p].wa, w0b, cur[p].wb, repa, repb, q2);
                trl_tc_mma<2>(C[2], w0a, cur[p].wa, w0b, cur[p].wb, repa, repb, q2);
                trl_tc_mma<3>(C[3], w0a, cur[p].wa, w0b, cur[p].wb, repa, repb, q2);
            }

            const float dA = va ? __ldg(a.ad + ca * bstr + b) : 0.f;
            const float dB = vb ? __ldg(a.ad + cb * bstr + b) : 0.f;
            const int   sA = va ? __ldg(a.asum + ca * bstr + b) : 0;
            const int   sB = vb ? __ldg(a.asum + cb * bstr + b) : 0;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r][0] = fmaf(dA, (float) (C[r][0] - 510 * sA), acc[r][0]);
                acc[r][1] = fmaf(dB, (float) (C[r][1] - 510 * sB), acc[r][1]);
                acc[r][2] = fmaf(dA, (float) (C[r][2] - 510 * sA), acc[r][2]);
                acc[r][3] = fmaf(dB, (float) (C[r][3] - 510 * sB), acc[r][3]);
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                cur[p] = nxt[p];
            }
        }

        const float * sc = reinterpret_cast<const float *>(a.blob + (size_t) rg * grpWords);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {                   // h = 0: row A g, h = 1: row A g+8
                const int rl = 8 * g + 2 * r + h;
                const int row = rg * 64 + rl;
                const float scr = sc[rl] * (1.0f / TRL_MUL1_STD);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = e == 0 ? ca : cb;
                    if (col < ncols) {
                        const float val = acc[r][2 * h + e] * scr;
                        if (ks == 1) {
                            a.y[(size_t) col * a.N + row] = val;
                        } else {
                            atomicAdd(a.y + (size_t) col * a.N + row, val);
                        }
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(a); GGML_UNUSED(ncols);
    NO_DEVICE_CODE;
#endif
}

static bool trl_env_tc() {
    static const bool v = getenv("TRL_TC") == nullptr || atoi(getenv("TRL_TC")) != 0;
    return v;
}

// columns from which it pays off (measured 31/10 on the 3060 Ti, step at 2/3/4 requests: 47.8/53.3/58.7 ms against
// 43.3/57.4/70.5 of the GEMV): at 2 columns the GEMV is still better
static int trl_env_tc_min() {
    static const int v = getenv("TRL_TC_MIN") ? atoi(getenv("TRL_TC_MIN")) : 3;
    return v < 2 ? 2 : v;
}

static void trl_tc_launch(trl_args a, int ncols, int device, int nsm, cudaStream_t stream) {
    static int occ_dev[GGML_CUDA_MAX_DEVICES] = {0};
    int & occ = occ_dev[device];
    if (occ == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, trl_gemv_tc, 256, 0));
        occ = occ < 1 ? 1 : occ;
    }
    const int grid = nsm * occ;
    const int nrg = a.N >> 6;
    const int nblk = (a.K >> 2) / 8;
    // K slices: ~1.5 items per resident warp, like the GEMV
    int ks = (3 * grid * 8 + 2 * nrg - 1) / (2 * nrg);
    ks = ks < 1 ? 1 : (ks > nblk ? nblk : ks);
    a.ks = ks;
    trl_gemv_tc<<<grid, 256, 0, stream>>>(a, ncols);
}

#define TRL_V3_HAVE_ARGS
#include "trl_gemv_v3.cuh"   // GEMV for Pascal/Turing (bench_gemv_turing, 29/10): 1660 S 181 -> 214 GB/s at one column

// Input rotation carried by the operation (dst->src[2] = d, op_params[1..4] = GDN block and reordering):
//   x~ = Hblk(d . P x), P = reordering of the V heads (ssm_out of the 27B) or identity.
// Without src[2] the input arrives already rotated by the graph (old path, ggml_hadamard).
struct trl_rot {
    const float * d = nullptr;
    int blk = 0, hd = 1, nk = 1, rep = 1;
};

static trl_rot trl_rot_of(const ggml_tensor * dst) {
    trl_rot r;
    if (dst->src[2] != nullptr) {
        r.d   = (const float *) dst->src[2]->data;
        r.blk = dst->op_params[1];
        r.hd  = dst->op_params[2];
        r.nk  = dst->op_params[3];
        r.rep = dst->op_params[4];
    }
    return r;
}

// One CUDA block = one Hadamard block of a column: (reordering, d, FWHT) in shared memory, then
//   MODE 0: int8 in blocks of 32 (q, scale, sum) for the GEMV · MODE 1: bf16 · MODE 2: fp32 (GEMM).
// If y0 != nullptr it also zeroes the output (the GEMV with K slices sums with atomicAdd): one launch less.
template <int MODE, typename T>
static __global__ void __launch_bounds__(256) trl_prep(const float * __restrict__ x, int64_t ldx, trl_rot r, int K,
                                                      int8_t * __restrict__ q, float * __restrict__ qd,
                                                      int * __restrict__ qs, T * __restrict__ out,
                                                      float * __restrict__ y0, int N) {
    extern __shared__ float sm[];
    const int blk = r.d ? r.blk : 256;
    const int col = blockIdx.y;
    const int k0 = blockIdx.x * blk;
    const int tid = threadIdx.x;
    const float * xc = x + col * ldx;
    for (int o = tid; o < blk; o += blockDim.x) {
        const int gi = k0 + o;
        float v;
        if (r.d) {
            int src = gi;
            if (r.rep > 1) {          // V heads: from the tiled order [hd, nk, rep] to the HF order [hd, rep, nk]
                const int hdi = gi % r.hd, t = gi / r.hd, repi = t % r.rep, nki = t / r.rep;
                src = hdi + r.hd * (nki + r.nk * repi);
            }
            v = xc[src] * r.d[gi];
        } else {
            v = xc[gi];
        }
        sm[o] = v;
    }
    __syncthreads();
    float scale = 1.0f;
    if (r.d) {
        for (int h = 1; h < blk; h <<= 1) {
            for (int p = tid; p < blk / 2; p += blockDim.x) {
                const int i = (p / h) * 2 * h + (p % h), j = i + h;
                const float a = sm[i], b = sm[j];
                sm[i] = a + b;
                sm[j] = a - b;
            }
            __syncthreads();
        }
        scale = rsqrtf((float) blk);
    }
    if constexpr (MODE == 0) {
        const int lane = tid & 31, warp = tid >> 5;
        for (int g = warp; g < blk / TRL_QB; g += blockDim.x / 32) {
            const float v = sm[g * TRL_QB + lane] * scale;
            float amax = fabsf(v);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            }
            const float dd = amax / 127.0f;
            const int qi = amax == 0.0f ? 0 : __float2int_rn(v / dd);
            const int64_t base = (int64_t) col * K + k0 + g * TRL_QB;
            q[base + lane] = (int8_t) qi;
            int sum = qi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            if (lane == 0) {
                qd[base / TRL_QB] = dd;
                qs[base / TRL_QB] = sum;
            }
        }
    } else {
        for (int o = tid; o < blk; o += blockDim.x) {
            const float v = sm[o] * scale;
            if constexpr (MODE == 1) {
                out[(int64_t) col * K + k0 + o] = __float2bfloat16(v);
            } else {
                out[(int64_t) col * K + k0 + o] = v;
            }
        }
    }
    if (y0) {
        for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
            y0[(int64_t) col * N + i] = 0.0f;
        }
    }
}

// For the int8 GEMM trl_mmq3: one CUDA block = one whole column (token). Reordering, d and FWHT of all the Hadamard
// blocks in smem, then int8 with ONE scale per token over the whole K (q [M][K], qd [M]). Reads x once.
static __global__ void __launch_bounds__(512) trl_prep_tok(const float * __restrict__ x, int64_t ldx, trl_rot r, int K,
                                                          int8_t * __restrict__ q, float * __restrict__ qd) {
    extern __shared__ float sm[];
    __shared__ float red[16];
    const int col = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float * xc = x + col * ldx;
    for (int gi = tid; gi < K; gi += nt) {
        float v;
        if (r.d) {
            int src = gi;
            if (r.rep > 1) {          // V heads: from the tiled order [hd, nk, rep] to the HF order [hd, rep, nk]
                const int hdi = gi % r.hd, t = gi / r.hd, repi = t % r.rep, nki = t / r.rep;
                src = hdi + r.hd * (nki + r.nk * repi);
            }
            v = xc[src] * r.d[gi];
        } else {
            v = xc[gi];
        }
        sm[gi] = v;
    }
    __syncthreads();
    float scale = 1.0f;
    if (r.d) {
        const int lb = __ffs(r.blk) - 1;                  // power-of-2 block
        for (int h = 1; h < r.blk; h <<= 1) {
            for (int p = tid; p < K / 2; p += nt) {
                const int b = p >> (lb - 1), pp = p & ((r.blk >> 1) - 1);
                const int i = (b << lb) + (pp / h) * 2 * h + (pp % h), j = i + h;
                const float a = sm[i], c = sm[j];
                sm[i] = a + c;
                sm[j] = a - c;
            }
            __syncthreads();
        }
        scale = rsqrtf((float) r.blk);
    }
    float amax = 0.0f;
    for (int gi = tid; gi < K; gi += nt) amax = fmaxf(amax, fabsf(sm[gi]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((tid & 31) == 0) red[tid >> 5] = amax;
    __syncthreads();
    amax = red[0];
    for (int i = 1; i < nt / 32; ++i) amax = fmaxf(amax, red[i]);
    amax *= scale;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int8_t * qc = q + (int64_t) col * K;
    for (int gi = 4 * tid; gi < K; gi += 4 * nt) {
        char4 v;
        v.x = (int8_t) __float2int_rn(sm[gi + 0] * scale * inv);
        v.y = (int8_t) __float2int_rn(sm[gi + 1] * scale * inv);
        v.z = (int8_t) __float2int_rn(sm[gi + 2] * scale * inv);
        v.w = (int8_t) __float2int_rn(sm[gi + 3] * scale * inv);
        *reinterpret_cast<char4 *>(qc + gi) = v;
    }
    if (tid == 0) qd[col] = amax / 127.0f;
}

template <int MODE, typename T>
static void trl_prep_launch(const float * x, int64_t ldx, const trl_rot & r, int K, int ncols,
                            int8_t * q, float * qd, int * qs, T * out, float * y0, int N, cudaStream_t stream) {
    const int blk = r.d ? r.blk : 256;
    GGML_ASSERT(K % blk == 0 && blk % TRL_QB == 0 && blk <= 8192);
    const dim3 grid(K / blk, ncols);
    trl_prep<MODE, T><<<grid, 256, blk * sizeof(float), stream>>>(x, ldx, r, K, q, qd, qs, out, y0, N);
}

bool ggml_cuda_trl_mmv_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return (src0->type == GGML_TYPE_TRL2_M || src0->type == GGML_TYPE_TRL3_M) &&
           src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= 8 && src0->ne[2] * src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] == 1 &&
           src0->ne[0] % 256 == 0 && src0->ne[1] % 64 == 0 &&
           src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
           ggml_is_contiguous(dst);
}

struct trl_cfg { int grid, ks; };

template <int B, int WPL, int NC>
static trl_cfg trl_config(int N, int K, int device, int nsm) {
    constexpr int U = 2;
    constexpr int WARPS = 8;
    constexpr int SPL = 32 * WPL / B, RPL = SPL / 4, LPT = 64 / RPL, TPS = 32 / LPT;
    static int occ_dev[GGML_CUDA_MAX_DEVICES] = {0};    // resident blocks per SM, per GPU and per NC
    int & occ = occ_dev[device];
    if (occ == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, trl_gemv<B, WPL, U, NC>, WARPS * 32, 0));
        occ = occ < 1 ? 1 : occ;
    }
    const int grid = nsm * occ;
    const int nrg = N / 64;
    const int nchunks = (K / 4) / (TPS * 8);
    // K slices: ~1.5 items per resident warp (from the best of the bench), at most one per block of 32 columns
    int ks = (3 * grid * WARPS + 2 * nrg - 1) / (2 * nrg);
    ks = ks < 1 ? 1 : (ks > nchunks ? nchunks : ks);
    return { grid, ks };
}

// Diagnostics (sentinel): TRL_KS1=1 removes the K slices, hence the atomicAdd, and makes the GEMV deterministic;
// TRL_NC_MAX=n limits the columns per launch (1 = one column at a time).
static bool trl_env_ks1() {
    static const bool v = getenv("TRL_KS1") != nullptr;
    return v;
}
static int trl_env_nc_max() {
    static const int v = getenv("TRL_NC_MAX") ? atoi(getenv("TRL_NC_MAX")) : 4;
    return v < 1 ? 1 : (v > 4 ? 4 : v);
}

template <int B, int WPL, int NC>
static void trl_launch(trl_args a, int device, int nsm, cudaStream_t stream) {
    // Pascal/Turing: v3 kernel (exact work distribution); TRL_GEMV_V3=0 goes back to the previous one (sentinel)
    static const bool v3 = getenv("TRL_GEMV_V3") == nullptr || atoi(getenv("TRL_GEMV_V3")) != 0;
    if (v3 && !trl_env_ks1() && trl_gemv_v3<B, NC>(a, ggml_cuda_info().devices[device].cc, nsm, stream)) {
        return;
    }
    const trl_cfg c = trl_config<B, WPL, NC>(a.N, a.K, device, nsm);
    a.ks = trl_env_ks1() ? 1 : c.ks;
    trl_gemv<B, WPL, 2, NC><<<c.grid, 8 * 32, 0, stream>>>(a);
}

// Columns per group from which the kernel runs with half the words per lane (half the rows per lane, so half
// the accumulator registers: with 16 rows x 4 columns the accumulators alone fill the register file and one block
// fits per SM). TRL_NC_HALF=n overrides (0 = never).
static int trl_env_nc_half() {
    static const int v = getenv("TRL_NC_HALF") ? atoi(getenv("TRL_NC_HALF")) : 2;
    return v;
}

template <int B, int WPL, int WPL_H, int NC>
static void trl_launch_w(trl_args a, int device, int nsm, cudaStream_t stream) {
    const int h = trl_env_nc_half();
    if (WPL_H != WPL && h > 0 && NC >= h) {
        trl_launch<B, WPL_H, NC>(a, device, nsm, stream);
    } else {
        trl_launch<B, WPL, NC>(a, device, nsm, stream);
    }
}

template <int B, int WPL, int WPL_H = WPL>
static void trl_run(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                    ggml_tensor * dst, const trl_rot & r) {
    const int K = (int) src0->ne[0], N = (int) src0->ne[1], ncols = (int) src1->ne[1];
    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) ncols * K);
    ggml_cuda_pool_alloc<float>  qd(ctx.pool(), (size_t) ncols * K / TRL_QB);
    ggml_cuda_pool_alloc<int>    qs(ctx.pool(), (size_t) ncols * K / TRL_QB);
    // the output is always zeroed: each NC instance chooses its K slices and those with ks > 1 sum with atomicAdd
    trl_prep_launch<0, float>((const float *) src1->data, src1->nb[1] / sizeof(float), r, K, ncols,
                              q.get(), qd.get(), qs.get(), nullptr, (float *) dst->data, N, stream);
    // 2..8 columns at 2 bit on Ampere: tensor units, in groups of 8
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (B == 2 && cc >= CC_AMPERE && ncols >= trl_env_tc_min() && trl_env_tc() && K % 32 == 0) {
        static const bool tc_check = getenv("TRL_TC_CHECK") != nullptr;
        ggml_cuda_pool_alloc<float> ref(ctx.pool());
        if (tc_check) {
            // the usual GEMV into the reference buffer, then the comparison
            ref.alloc((size_t) ncols * N);
            CUDA_CHECK(cudaMemsetAsync(ref.get(), 0, (size_t) ncols * N * sizeof(float), stream));
            for (int col = 0; col < ncols; ) {
                const int nc = ncols - col >= 4 ? 4 : ncols - col;
                const trl_args a = { (const uint32_t *) src0->data, (const int *) (q.get() + (size_t) col * K),
                                     qd.get() + (size_t) col * K / TRL_QB, qs.get() + (size_t) col * K / TRL_QB,
                                     ref.get() + (size_t) col * N, N, K, 0 };
                switch (nc) {
                    case 1:  trl_launch<B, WPL, 1>(a, ctx.device, nsm, stream); break;
                    case 2:  trl_launch<B, WPL, 2>(a, ctx.device, nsm, stream); break;
                    case 3:  trl_launch<B, WPL, 3>(a, ctx.device, nsm, stream); break;
                    default: trl_launch<B, WPL, 4>(a, ctx.device, nsm, stream); break;
                }
                col += nc;
            }
        }
        for (int col = 0; col < ncols; col += 8) {
            const int nc = ncols - col >= 8 ? 8 : ncols - col;
            const trl_args a = { (const uint32_t *) src0->data, (const int *) (q.get() + (size_t) col * K),
                                 qd.get() + (size_t) col * K / TRL_QB, qs.get() + (size_t) col * K / TRL_QB,
                                 (float *) dst->data + (size_t) col * N, N, K, 0 };
            trl_tc_launch(a, nc, ctx.device, nsm, stream);
        }
        if (tc_check) {
            std::vector<float> x((size_t) ncols * N), y((size_t) ncols * N);
            CUDA_CHECK(cudaMemcpyAsync(x.data(), dst->data, x.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(y.data(), ref.get(), y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double dmax = 0, ymax = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                dmax = std::max(dmax, (double) fabsf(x[i] - y[i]));
                ymax = std::max(ymax, (double) fabsf(y[i]));
            }
            static int n_check = 0;
            if (n_check++ < 40 || dmax > 1e-3 * ymax) {
                fprintf(stderr, "TRL_TC_CHECK N=%d K=%d ncols=%d: max|tc-gemv| = %.3e, max|gemv| = %.3e, relative %.2e\n",
                        N, K, ncols, dmax, ymax, ymax > 0 ? dmax / ymax : 0.0);
            }
        }
        return;
    }
    // in groups of at most 4 columns (beyond that, the accumulator registers overflow)
    for (int col = 0; col < ncols; ) {
        const int nmax = trl_env_nc_max();
        const int nc = ncols - col >= nmax ? nmax : ncols - col;
        const trl_args a = { (const uint32_t *) src0->data, (const int *) (q.get() + (size_t) col * K),
                             qd.get() + (size_t) col * K / TRL_QB, qs.get() + (size_t) col * K / TRL_QB,
                             (float *) dst->data + (size_t) col * N, N, K, 0 };
        switch (nc) {
            case 1:  trl_launch_w<B, WPL, WPL_H, 1>(a, ctx.device, nsm, stream); break;
            case 2:  trl_launch_w<B, WPL, WPL_H, 2>(a, ctx.device, nsm, stream); break;
            case 3:  trl_launch_w<B, WPL, WPL_H, 3>(a, ctx.device, nsm, stream); break;
            default: trl_launch_w<B, WPL, WPL_H, 4>(a, ctx.device, nsm, stream); break;
        }
        col += nc;
    }
}

static void trl_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                      ggml_tensor * dst);

void ggml_cuda_trl_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
    static const bool check = getenv("TRL_CHECK") != nullptr;
    if (check) {
        trl_check(ctx, src0, src1, dst);
    }
    const trl_rot r = trl_rot_of(dst);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // TRL_TIME=1 (only without CUDA graphs): GPU time of every call, summed per (bit, columns), printed every 2000
    static const bool timing = getenv("TRL_TIME") != nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (timing) {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, ctx.stream()));
    }
    struct trl_time_end {
        cudaEvent_t e0, e1; cudaStream_t st; int b, nc;
        ~trl_time_end() {
            if (!e0) return;
            CUDA_CHECK(cudaEventRecord(e1, st));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            static double tot[2][9] = {}; static long cnt[2][9] = {};
            const int i = b == 2 ? 0 : 1, j = nc < 8 ? nc : 8;
            tot[i][j] += ms; cnt[i][j] += 1;
            if (cnt[i][j] % 2000 == 0) {
                fprintf(stderr, "TRL_TIME %d bit, %d columns: %ld calls, avg %.1f us\n", b, nc, cnt[i][j], 1000.0 * tot[i][j] / cnt[i][j]);
            }
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    } trl_time_guard { ev0, ev1, ctx.stream(), src0->type == GGML_TYPE_TRL2_M ? 2 : 3, (int) src1->ne[1] };
    // words per lane from the best of the bench: at 2 bit 4 (2 on the 1080), at 3 bit 6 (3 on the 1080)
    if (src0->type == GGML_TYPE_TRL2_M) {
        if (cc < 700) trl_run<2, 2>(ctx, src0, src1, dst, r);
        else          trl_run<2, 4, 2>(ctx, src0, src1, dst, r);
    } else {
        if (cc < 700) trl_run<3, 3>(ctx, src0, src1, dst, r);
        else          trl_run<3, 6, 3>(ctx, src0, src1, dst, r);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------
// Prompt reading on Ampere (sm_80+): fused GEMM with the int8 tensor units.
// The integer weight q = bytesum(state*K) - 510 in [-510, 510] becomes h = min((q + 2) >> 2, 127) in int8 (rebuilt
// weight 4h: uniform error in [-2, 2], std 1.15 = 0.78% of the weight std; measured 0.82% on the outputs
// of the single product). Int8 activations in blocks of 32 like the GEMV (trl_prep MODE 0).
// Block = 64 rows (a band) x 128 tokens, 4 warps; each warp 64 rows x 32 tokens (4 x 4 mma m16n8k32).
// The 128 threads decode the 64x32 weights of an activation block into smem (double buffer); the words of
// block kb+2 are read into registers while working on kb. TRL_MMQ=0 goes back to reconstruction + cuBLAS (exact).
// ---------------------------------------------------------------------------------------------------------
#define TRL_MMQ_BN 128

static __device__ __forceinline__ void trl_mma_s8s8(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= CC_AMPERE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a0); GGML_UNUSED(a1); GGML_UNUSED(a2); GGML_UNUSED(a3); GGML_UNUSED(b0); GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void trl_ldsm_x4(uint32_t addr, uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= CC_AMPERE
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#else
    GGML_UNUSED(addr); r0 = r1 = r2 = r3 = 0;
    NO_DEVICE_CODE;
#endif
}

// smem [64 rows][32 bytes]: 16 B half shuffled with (row >> 2) & 1 (ldmatrix without conflicts)
static __device__ __forceinline__ uint32_t trl_mmq_swz(int r, int w8) {
    return (uint32_t) (r * 8 + ((((w8 >> 2) ^ (r >> 2)) & 1) << 2) + (w8 & 3));
}

template <int B>
static __device__ __forceinline__ void trl_mmq_load(const uint32_t * __restrict__ tile, int o, uint32_t S[B + 1]) {
    S[0] = o == 0 ? 0u : __ldg(tile + o * B - 1);
#pragma unroll
    for (int q = 0; q < B; ++q) S[q + 1] = __ldg(tile + o * B + q);
}

template <int B, int HALF>
static __device__ __forceinline__ void trl_mmq_decode(const uint32_t S[B + 1], int o, uint32_t * sm, int t) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= CC_AMPERE
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        int h[4];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
            const int u = (HALF * 4 + rr) * 4 + cc;
            const int E = 32 + B * (u + 1), st0 = E - TRL_L, wi = st0 >> 5, off = st0 & 31;
            const uint32_t nx = (wi < B) ? S[wi + 1] : 0u;
            const uint32_t w = (__funnelshift_l(nx, S[wi], off) >> (32 - TRL_L)) * TRL_MUL1_K;
            // (bytesum - 510 + 2) >> 2, with the offset inside the dp4a accumulator
            h[cc] = trl_dp4a_us(w, 0x01010101, -508) >> 2;
        }
        // packing with saturation to int8 (h = 128 only for bytesum = 1020): bytes 0..3 = h0..h3
        uint32_t hi, v;
        asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(h[3]), "r"(h[2]), "r"(0));
        asm("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(h[1]), "r"(h[0]), "r"(hi));
        sm[trl_mmq_swz(8 * o + HALF * 4 + rr, t)] = v;
    }
#else
    GGML_UNUSED(S); GGML_UNUSED(o); GGML_UNUSED(sm); GGML_UNUSED(t);
    NO_DEVICE_CODE;
#endif
}

struct trl_mmq_args {
    const uint32_t * blob;
    const int8_t   * aq;     // [M][K]
    const float    * ad;     // [M][K/32]
    float          * y;      // [M][N]
    int N, K, M;
};

template <int B>
static __global__ void __launch_bounds__(128) trl_mmq(trl_mmq_args a) {
    __shared__ __align__(16) uint32_t smA[2][64 * 8];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, tig = lane & 3;
    const int rg = blockIdx.x, n_base = blockIdx.y * TRL_MMQ_BN + w * 32;
    const int nk32 = a.K >> 5;
    const size_t bandWords = 64 + (size_t) (a.K >> 2) * 8 * B;
    const uint32_t * gb = a.blob + (size_t) rg * bandWords;
    const uint32_t * tw = gb + 64;
    const int dt = tid & 7, dob = (tid >> 3) & 7, dh = tid >> 6;
    const uint32_t smb0 = (uint32_t) __cvta_generic_to_shared(&smA[0][0]);
    const int lm_r = (lane & 7) + (((lane >> 3) & 1) << 3), lm_w = (lane >> 4) << 2;

    float acc[4][4][4];
#pragma unroll
    for (int f = 0; f < 4; ++f)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[f][j][i] = 0.f;

    uint32_t Sc[B + 1], Sn[B + 1];
    auto load = [&](int kb, uint32_t S[B + 1]) { trl_mmq_load<B>(tw + (size_t) (kb * 8 + dt) * (8 * B), dob, S); };
    auto decode = [&](const uint32_t S[B + 1], int buf) {
        if (dh == 0) trl_mmq_decode<B, 0>(S, dob, smA[buf], dt);
        else         trl_mmq_decode<B, 1>(S, dob, smA[buf], dt);
    };
    load(0, Sc);
    decode(Sc, 0);
    if (nk32 > 1) load(1, Sc);
    __syncthreads();
    for (int kb = 0; kb < nk32; ++kb) {
        const int buf = kb & 1;
        if (kb + 2 < nk32) load(kb + 2, Sn);
        int c[4][4][4];
#pragma unroll
        for (int f = 0; f < 4; ++f)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) c[f][j][i] = 0;
        uint32_t bf[4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n_base + 8 * j + g;
            const bool ok = n < a.M;
            const uint32_t * ac = reinterpret_cast<const uint32_t *>(a.aq + (size_t) (ok ? n : 0) * a.K + kb * 32);
            bf[j][0] = ok ? __ldg(ac + tig) : 0u;
            bf[j][1] = ok ? __ldg(ac + 4 + tig) : 0u;
        }
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            uint32_t a0, a1, a2, a3;
            trl_ldsm_x4(smb0 + buf * (64 * 8 * 4) + 4 * trl_mmq_swz(16 * f + lm_r, lm_w), a0, a1, a2, a3);
#pragma unroll
            for (int j = 0; j < 4; ++j) trl_mma_s8s8(c[f][j], a0, a1, a2, a3, bf[j][0], bf[j][1]);
        }
        if (kb + 1 < nk32) {
            decode(Sc, buf ^ 1);
#pragma unroll
            for (int q = 0; q <= B; ++q) Sc[q] = Sn[q];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n0 = n_base + 8 * j + 2 * tig, n1 = n0 + 1;
            const float dA0 = n0 < a.M ? __ldg(a.ad + (size_t) n0 * nk32 + kb) : 0.f;
            const float dA1 = n1 < a.M ? __ldg(a.ad + (size_t) n1 * nk32 + kb) : 0.f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                acc[f][j][0] = fmaf(dA0, (float) c[f][j][0], acc[f][j][0]);
                acc[f][j][1] = fmaf(dA1, (float) c[f][j][1], acc[f][j][1]);
                acc[f][j][2] = fmaf(dA0, (float) c[f][j][2], acc[f][j][2]);
                acc[f][j][3] = fmaf(dA1, (float) c[f][j][3], acc[f][j][3]);
            }
        }
        __syncthreads();
    }
    const float * sc = reinterpret_cast<const float *>(gb);
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        const int r0 = 16 * f + g, r1 = r0 + 8;
        const float s0 = sc[r0] * (4.0f / TRL_MUL1_STD), s1 = sc[r1] * (4.0f / TRL_MUL1_STD);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n0 = n_base + 8 * j + 2 * tig, n1 = n0 + 1;
            if (n0 < a.M) { a.y[(size_t) n0 * a.N + rg * 64 + r0] = acc[f][j][0] * s0; a.y[(size_t) n0 * a.N + rg * 64 + r1] = acc[f][j][2] * s1; }
            if (n1 < a.M) { a.y[(size_t) n1 * a.N + rg * 64 + r0] = acc[f][j][1] * s0; a.y[(size_t) n1 * a.N + rg * 64 + r1] = acc[f][j][3] * s1; }
        }
    }
}

// TRL_MMQ: 2 (default) trl_mmq3, int8 activations with one scale per token; 1 trl_mmq, scales in blocks of 32;
// 0 reconstruction + cuBLAS (exact). The comparison among the three is the perplexity sentinel.
static int trl_env_mmq() {
    static const int v = getenv("TRL_MMQ") == nullptr ? 2 : atoi(getenv("TRL_MMQ"));
    return v;
}

template <int B>
static void trl_mmq3_launch(const uint32_t * blob, const int8_t * q, const float * qd, float * y, int N, int K, int M,
                            int device, cudaStream_t stream) {
    constexpr int WT = 4, NS = 3;
    constexpr size_t smem = trl3::smem_bytes<B, WT, NS>();
    static bool attr[GGML_CUDA_MAX_DEVICES] = {false};
    if (!attr[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(trl3::mmq<B, WT, NS, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr[device] = true;
    }
    const dim3 grid((unsigned) ((M + 32 * WT - 1) / (32 * WT)), (unsigned) ((N + 127) / 128));
    trl3::mmq<B, WT, NS, 2><<<grid, 64 * WT, smem, stream>>>(blob, q, qd, y, N, K, M);
}

bool ggml_cuda_trl_mm_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type >= GGML_TYPE_TRL2_G && src0->type <= GGML_TYPE_TRL3_M &&
           src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[2] * src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] == 1 &&
           src0->ne[0] % 256 == 0 && src0->ne[1] % 64 == 0 &&
           src1->nb[0] == sizeof(float) && ggml_is_contiguous(dst);
}

void ggml_cuda_trl_mul_mat_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst) {
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    const int id = ctx.device;
    const int cc = ggml_cuda_info().devices[id].cc;
    cudaStream_t stream = ctx.stream();
    const trl_rot r = trl_rot_of(dst);
    const int64_t ldx = src1->nb[1] / sizeof(float);
    if (cc >= CC_AMPERE && (src0->type == GGML_TYPE_TRL2_M || src0->type == GGML_TYPE_TRL3_M) && trl_env_mmq() == 2) {
        ggml_cuda_pool_alloc<int8_t> q(ctx.pool(id), (size_t) M * K);
        ggml_cuda_pool_alloc<float>  qd(ctx.pool(id), (size_t) M);
        static bool attr[GGML_CUDA_MAX_DEVICES] = {false};
        if (!attr[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(trl_prep_tok, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
            attr[id] = true;
        }
        GGML_ASSERT(K * sizeof(float) <= 96 * 1024 && K % 4 == 0);
        trl_prep_tok<<<(unsigned) M, 512, K * sizeof(float), stream>>>((const float *) src1->data, ldx, r, (int) K, q.get(), qd.get());
        if (src0->type == GGML_TYPE_TRL2_M) trl_mmq3_launch<2>((const uint32_t *) src0->data, q.get(), qd.get(), (float *) dst->data, (int) N, (int) K, (int) M, id, stream);
        else                                trl_mmq3_launch<3>((const uint32_t *) src0->data, q.get(), qd.get(), (float *) dst->data, (int) N, (int) K, (int) M, id, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cc >= CC_AMPERE && (src0->type == GGML_TYPE_TRL2_M || src0->type == GGML_TYPE_TRL3_M) && trl_env_mmq() == 1) {
        ggml_cuda_pool_alloc<int8_t> q(ctx.pool(id), (size_t) M * K);
        ggml_cuda_pool_alloc<float>  qd(ctx.pool(id), (size_t) M * K / TRL_QB);
        ggml_cuda_pool_alloc<int>    qs(ctx.pool(id), (size_t) M * K / TRL_QB);
        trl_prep_launch<0, float>((const float *) src1->data, ldx, r, (int) K, (int) M, q.get(), qd.get(), qs.get(),
                                  nullptr, nullptr, 0, stream);
        const trl_mmq_args ma = { (const uint32_t *) src0->data, q.get(), qd.get(), (float *) dst->data,
                                  (int) N, (int) K, (int) M };
        const dim3 grid((unsigned) (N / 64), (unsigned) ((M + TRL_MMQ_BN - 1) / TRL_MMQ_BN));
        if (src0->type == GGML_TYPE_TRL2_M) trl_mmq<2><<<grid, 128, 0, stream>>>(ma);
        else                                trl_mmq<3><<<grid, 128, 0, stream>>>(ma);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cc < CC_AMPERE && (src0->type == GGML_TYPE_TRL2_M || src0->type == GGML_TYPE_TRL3_M)) {
        // GPUs without int8/bf16 tensor cores: fused kernel (decode in smem, no reconstruction + cuBLAS).
        // Pascal -> f32 (exact), Turing -> h2 (fast fp16, error below bf16). TRL_GEMM_NT=off|f32|h2|i8 forces the path.
        static const char * env = getenv("TRL_GEMM_NT");
        int nt = cc < CC_VOLTA ? TRL_NT_F32 : (cc == CC_TURING ? TRL_NT_H2 : -1);
        if (env != nullptr) {
            nt = !strcmp(env, "f32") ? TRL_NT_F32 : !strcmp(env, "h2") ? TRL_NT_H2 : !strcmp(env, "i8") ? TRL_NT_I8 : -1;
            if (nt == TRL_NT_H2 && !fast_fp16_available(cc)) nt = TRL_NT_F32;   // on Pascal fp16 runs at 1/64
        }
        if (nt >= 0) {
            ggml_cuda_pool_alloc<float> x32(ctx.pool(id), (size_t) K * M);
            trl_prep_launch<2, float>((const float *) src1->data, ldx, r, (int) K, (int) M,
                                      nullptr, nullptr, nullptr, x32.get(), nullptr, 0, stream);
            ggml_cuda_pool_alloc<char> scr(ctx.pool(id), trl_nt_scratch_bytes(nt, (int) K, (int) M, 32) + 16);
            trl_nt_run(nt, src0->type == GGML_TYPE_TRL2_M ? 2 : 3, (const uint32_t *) src0->data, x32.get(), scr.get(),
                       (float *) dst->data, (int) N, (int) K, (int) M, 32, stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const size_t row_bytes = ggml_row_size(src0->type, K);
    // slices of rows (multiples of 64): at most ~32M weights rebuilt at a time (64 MB in bf16)
    int64_t R = ((int64_t(1) << 25) / K) / 64 * 64;
    R = R < 64 ? 64 : (R > N ? N : R);
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    if (fast_fp16_available(cc)) {
        // ⚠️ bf16, not fp16: cuBLAS's fp16 GEMM was 97% of prompt reading on the 1660
        ggml_cuda_pool_alloc<nv_bfloat16> x16(ctx.pool(id), K * M);
        trl_prep_launch<1, nv_bfloat16>((const float *) src1->data, ldx, r, (int) K, (int) M,
                                        nullptr, nullptr, nullptr, x16.get(), nullptr, 0, stream);
        ggml_cuda_pool_alloc<nv_bfloat16> w16(ctx.pool(id), R * K);
        const to_bf16_cuda_t deq = ggml_get_to_bf16_cuda(src0->type);
        for (int64_t r0 = 0; r0 < N; r0 += R) {
            const int64_t nr = R < N - r0 ? R : N - r0;
            deq((const char *) src0->data + r0 * row_bytes, w16.get(), nr, K, stream);
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N, nr, M, K,
                                      &alpha, w16.get(), CUDA_R_16BF, K, x16.get(), CUDA_R_16BF, K,
                                      &beta, (float *) dst->data + r0, CUDA_R_32F, N,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else {
        ggml_cuda_pool_alloc<float> x32(ctx.pool(id), K * M);
        trl_prep_launch<2, float>((const float *) src1->data, ldx, r, (int) K, (int) M,
                                  nullptr, nullptr, nullptr, x32.get(), nullptr, 0, stream);
        ggml_cuda_pool_alloc<float> w32(ctx.pool(id), R * K);
        const to_fp32_cuda_t deq = ggml_get_to_fp32_cuda(src0->type);
        for (int64_t r0 = 0; r0 < N; r0 += R) {
            const int64_t nr = R < N - r0 ? R : N - r0;
            deq((const char *) src0->data + r0 * row_bytes, w32.get(), nr, K, stream);
            CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N, nr, M, K,
                                     &alpha, w32.get(), K, x32.get(), K,
                                     &beta, (float *) dst->data + r0, N));
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

// Diagnostics (TRL_CHECK=1): the GEMV of dst compared with the GEMM path (rebuilt weights + same rotation).
static void trl_check(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                      ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), n);
    ggml_tensor tmp = *dst;
    tmp.data = ref.get();
    ggml_cuda_trl_mul_mat_gemm(ctx, src0, src1, &tmp);
    const trl_rot r = trl_rot_of(dst);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (src0->type == GGML_TYPE_TRL2_M) {
        if (cc < 700) trl_run<2, 2>(ctx, src0, src1, dst, r); else trl_run<2, 4>(ctx, src0, src1, dst, r);
    } else {
        if (cc < 700) trl_run<3, 3>(ctx, src0, src1, dst, r); else trl_run<3, 6>(ctx, src0, src1, dst, r);
    }
    std::vector<float> a(n), b(n), x(ggml_nelements(src1));
    CUDA_CHECK(cudaMemcpyAsync(a.data(), dst->data, n * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(b.data(), ref.get(), n * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(x.data(), src1->data, x.size() * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    double num = 0, den = 0, xs = 0;
    for (int64_t i = 0; i < n; ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    for (float v : x) xs += (double) v * v;
    fprintf(stderr, "TRL_CHECK %-28s %-28s dev %d N %6lld cols %lld rel %.2e |x| %.3e nb1 %zu\n", src0->name, dst->name,
            ctx.device, (long long) src0->ne[1], (long long) src1->ne[1], sqrt(num / (den + 1e-30)), sqrt(xs), src1->nb[1]);
    // TRL_DUMP=<file>: for the head (output.weight) saves the engine's input and logits, one slice per record,
    // for the offline comparison with the original bf16 head: int32 [row0, columns, K, N], x [columns][K],
    // logits [columns][N].
    static const char * dump = getenv("TRL_DUMP");
    if (dump && strncmp(src0->name, "output.weight", 13) == 0) {
        FILE * f = fopen(dump, "ab");
        if (f) {
            const int32_t h[4] = { (int32_t) (src0->view_offs / src0->nb[1]), (int32_t) src1->ne[1],
                                   (int32_t) src0->ne[0], (int32_t) src0->ne[1] };
            fwrite(h, sizeof(h), 1, f);
            fwrite(x.data(), sizeof(float), x.size(), f);
            fwrite(a.data(), sizeof(float), a.size(), f);
            fclose(f);
        }
    }
}
