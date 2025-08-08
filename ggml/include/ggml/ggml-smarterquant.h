#pragma once

#include "ggml.h"
#include "gguf.h" // for smarterquant_permutation

#ifdef __cplusplus
extern "C" {
#endif

void ggml_sq_unpermute_fp32(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src,
    int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif
