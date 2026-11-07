// Decode attention with q8_0 cache and GQA, without conversion to f16 (see fattn-q8-gqa.cu).
#pragma once

#include "common.cuh"

bool ggml_cuda_fattn_q8_gqa_supported(const ggml_tensor * dst);

void ggml_cuda_fattn_q8_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
