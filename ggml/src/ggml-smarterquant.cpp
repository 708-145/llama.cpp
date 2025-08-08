#include "ggml/ggml-smarterquant.h"
#include "ggml-impl.h"
#include "gguf.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"


#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

// unpermute a float32 vector using the inverse map
static void ggml_sq_unpermute_fp32(struct ggml_tensor * dst , const struct ggml_tensor * src) {
    const int64_t ne0 = src->ne[0];
    const int64_t ne1 = src->ne[1];

    const smarterquant_permutation * p = (const smarterquant_permutation *) (src->extra);

    assert(p != NULL);
    assert(p->iperm.size() == (size_t) ne0);

    for (int64_t i1 = 0; i1 < ne1; i1++) {
        for (int64_t i = 0; i < ne0; i++) {
            ((float *) dst->data)[i1*ne0 + p->iperm[i]] = ((const float *) src->data)[i1*ne0 + i];
        }
    }
}

void ggml_compute_forward_sq_unpermute(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_UNUSED(params);

    switch (src->type) {
        case GGML_TYPE_F32:
            ggml_sq_unpermute_fp32(dst, src);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
