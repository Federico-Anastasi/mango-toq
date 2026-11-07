// TRL: matrix-vector product, one token at a time, on trellis weights with the computed
// codebook (TRL2_M / TRL3_M). Kernel variant C: int8 activations in blocks of 32, one dp4a per weight, measured at 69% of the 3060 Ti
// bandwidth at 2 bit. Prompt reading (more than 8 columns) stays on reconstruction + cuBLAS.
#pragma once

#include "common.cuh"

bool ggml_cuda_trl_mmv_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_trl_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

// More than 8 columns (prompt reading): weights rebuilt in slices of rows, then a cuBLAS GEMM with
// fp32 accumulation (fp16 input on GPUs with fast fp16, fp32 on the 1080).
bool ggml_cuda_trl_mm_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_trl_mul_mat_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst);
