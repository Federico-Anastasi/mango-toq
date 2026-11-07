// TRL: trellis weights (the format of the TOQ models).
//
// A linear W [N, K]: rows in bands of 64. Band i (rows 64i..64i+63) is contiguous:
//   [64 x float]           row scale s_n
//   [K/4 tiles x 32*b bytes] tile j = columns 4j..4j+3 of the 64 rows, 256 symbols of b bits in the order
//                          (r0,c0)(r0,c1)(r0,c2)(r0,c3)(r1,c0)...(r63,c3)
// The tile's bit stream is big-endian over 256*b bits, in uint32 words: bit p is in word
// p>>5 at bit 31-(p&31). The state of weight t (L = 14) is the 14-bit window ending at bit
// (t+1)*b, with zeros before the start of the tile. Value: W[n,k] = s_n * cb(state).
//   *_G: cb = gaussian table (ggml-trl-cb.h)   *_M: cb = (bytesum(state*0x83DCD12D) - 510) / 147.80
// The input must be rotated before the product: x~ = Hblk(d . x) (tensor `<name>.rot`, see llama).
#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define TRL_HD __host__ __device__ __forceinline__
#else
#define TRL_HD static inline
#endif

#define TRL_L      14
#define TRL_TX     64
#define TRL_G      4
#define TRL_TILE   256
#define TRL_MASK   ((1u << TRL_L) - 1u)
#define TRL_MUL1_K 0x83DCD12Du
#define TRL_MUL1_STD 147.80054127099805f

// State of weight t in tile `tw` (8*b words) at b bits per symbol.
TRL_HD uint32_t trl_state(const uint32_t * tw, int t, int b) {
    const int end = (t + 1) * b;
    const int start = end - TRL_L;
    if (start < 0) {
        return (tw[0] >> (32 - end)) & ((1u << end) - 1u);
    }
    const int i0 = start >> 5, off = start & 31;
    const uint64_t hi = tw[i0];
    const uint64_t lo = (i0 + 1 < 8 * b) ? tw[i0 + 1] : 0;
    const uint64_t v = (hi << 32) | lo;
    return (uint32_t)(v >> (64 - off - TRL_L)) & TRL_MASK;
}

// Sum of the 4 bytes of the multiplied state (integer, 0..1020): the mul1 codebook is (S-510)/std.
TRL_HD int trl_mul1_sum(uint32_t state) {
    const uint32_t x = state * TRL_MUL1_K;
    return (int)(x & 0xff) + (int)((x >> 8) & 0xff) + (int)((x >> 16) & 0xff) + (int)(x >> 24);
}

TRL_HD float trl_mul1_value(uint32_t state) {
    return (float)(trl_mul1_sum(state) - 510) / TRL_MUL1_STD;
}

// Bytes of a band of 64 rows with K columns at b bits.
TRL_HD int64_t trl_band_bytes(int64_t K, int b) {
    return TRL_TX * (4 + K * b / 8);
}
