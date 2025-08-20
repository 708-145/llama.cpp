#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

LLAMA_API uint32_t llama_model_do_analysis(
    const char * fname_inp,
    const char * analyze_file,
    const void * imatrix_data_ptr
);

#ifdef __cplusplus
}
#endif