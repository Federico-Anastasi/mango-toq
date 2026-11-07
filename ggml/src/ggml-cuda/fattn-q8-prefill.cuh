// Attention for prompt reading (many query tokens) with q8_0 cache and GQA, without converting the cache to f16
// (see fattn-q8-prefill.cu).
#pragma once

#include "common.cuh"

bool ggml_cuda_fattn_q8_prefill_supported(const ggml_tensor * dst);

void ggml_cuda_fattn_q8_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
