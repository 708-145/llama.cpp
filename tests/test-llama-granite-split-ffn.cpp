#include "llama.h"
#include "llama-model.h" // For llama_layer and direct model manipulation if needed
#include "ggml.h"
#include "common.h" // For common test utilities, if any (might not be strictly needed here)

#include <cassert>
#include <vector>
#include <cmath> // For fabs
#include <cstdio> // For printf

// Helper to compare float values with tolerance
bool are_floats_equal(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

void test_granite_model_loading_conceptual() {
    printf("Testing Granite Model Loading (Conceptual)\n");

    // This test is conceptual as it requires a pre-existing GGUF file.
    // Steps would be:
    // 1. Create a minimal GGUF file for LLM_ARCH_GRANITE with split ffn_down.x, ffn_down.y, and ffn_down.bias tensors.
    //    - E.g., n_embd=4, n_ff=8 (ffn_down_x: [2,4], ffn_down_y: [6,4] - assuming n_ff is split for ffn_down's first dim)
    //      From previous subtask: ffn_down_x (256, n_embd), ffn_down_y (n_ff-256, n_embd)
    //      Let n_embd = 4, n_ff = 8.
    //      ffn_down_x: (2, 4) -> if 256 is conceptual for "part X" size, let "part X size" = 2
    //      ffn_down_y: (8-2, 4) = (6,4)
    //      So, ffn_down_x->ne = {2, 4}, ffn_down_y->ne = {6, 4}

    /*
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file("tests/test-granite-split-ffn-xy.gguf", model_params);

    if (model == nullptr) {
        fprintf(stderr, "Failed to load test-granite-split-ffn-xy.gguf. Ensure the file exists and is valid.\n");
        assert(false);
        return;
    }

    assert(model->arch == LLM_ARCH_GRANITE);
    assert(model->hparams.n_layer > 0); // Assuming at least one layer in GGUF

    const auto & layer = model->layers[0];
    assert(layer.ffn_down_x != nullptr);
    assert(layer.ffn_down_y != nullptr);
    assert(layer.ffn_down_b != nullptr); // Bias should be loaded
    assert(layer.ffn_down == nullptr); // As set by loader for Granite

    // Example dimension check (adjust to actual GGUF values)
    // assert(layer.ffn_down_x->ne[0] == 2); // "part X size"
    // assert(layer.ffn_down_x->ne[1] == model->hparams.n_embd);
    // assert(layer.ffn_down_y->ne[0] == model->hparams.n_ff_arr[0] - 2);
    // assert(layer.ffn_down_y->ne[1] == model->hparams.n_embd);
    // assert(layer.ffn_down_b->ne[0] == model->hparams.n_embd);


    printf("  Conceptual: ffn_down_x found with ne[0]=%lld, ne[1]=%lld\n", layer.ffn_down_x->ne[0], layer.ffn_down_x->ne[1]);
    printf("  Conceptual: ffn_down_y found with ne[0]=%lld, ne[1]=%lld\n", layer.ffn_down_y->ne[0], layer.ffn_down_y->ne[1]);
    printf("  Conceptual: ffn_down_b found with ne[0]=%lld\n", layer.ffn_down_b->ne[0]);

    llama_free_model(model);
    */
    printf("  Test conceptually passed (GGUF loading part skipped).\n");
}


// Helper to initialize a tensor with a specific value
void init_tensor(struct ggml_tensor * tensor, float value) {
    float * data = (float *)tensor->data;
    for (int i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = value;
    }
}

// Helper to print a tensor's data (for debugging)
void print_tensor(const struct ggml_tensor * tensor, const char * name) {
    printf("Tensor: %s, shape: (%lld, %lld), type: %s\n", name, tensor->ne[0], tensor->ne[1], ggml_type_name(tensor->type));
    if (tensor->data == nullptr) {
        printf(" (no data)\n");
        return;
    }
    float * data = (float *)tensor->data;
    for (int r = 0; r < tensor->ne[1]; ++r) {
        for (int c = 0; c < tensor->ne[0]; ++c) {
            printf("%.2f ", data[r * tensor->ne[0] + c]);
        }
        printf("\n");
    }
}


void test_granite_ffn_computation() {
    printf("Testing Granite FFN Computation\n");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024, // 128 MB, should be plenty
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false, // we need to allocate tensors
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    llama_model model_mock;
    model_mock.arch = LLM_ARCH_GRANITE;
    llama_hparams & hparams = model_mock.hparams;

    // Define dimensions
    hparams.n_embd = 4; // num_rows for ffn_down_x/y, output dimension of FFN
    const int n_ff = 8;  // total input dimension to ffn_down stage
    const int n_ff_x_cols = 2; // Corresponds to 256 in real model, columns for ffn_down_x's weights (transposed)
                               // Or rows of ffn_hidden_x
    const int n_ff_y_cols = n_ff - n_ff_x_cols; // = 6
    const int n_tokens = 1; // For simplicity

    hparams.n_layer = 1;
    model_mock.layers.resize(1);
    // Initialize n_ff_arr for the model
    hparams.n_ff_arr.fill(0); // zero out first
    hparams.n_ff_arr[0] = n_ff;


    auto & layer = model_mock.layers[0];

    // Create and initialize tensors with known values
    // Weights are (cols, rows) for ggml_mul_mat's first argument perspective after transposition
    // i.e. ffn_down_x is stored as (n_ff_x_cols, n_embd)
    layer.ffn_down_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff_x_cols, hparams.n_embd);
    ggml_set_name(layer.ffn_down_x, "blk.0.ffn_down.x");
    layer.ffn_down_y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff_y_cols, hparams.n_embd);
    ggml_set_name(layer.ffn_down_y, "blk.0.ffn_down.y");
    layer.ffn_down_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd); // Bias tensor
    ggml_set_name(layer.ffn_down_b, "blk.0.ffn_down.bias");

    init_tensor(layer.ffn_down_x, 1.0f); // W_x: all 1s
    init_tensor(layer.ffn_down_y, 2.0f); // W_y: all 2s
    init_tensor(layer.ffn_down_b, 0.5f); // Bias: all 0.5s


    // Mock ffn_hidden input (output of silu(gate)*up)
    // Shape: (n_ff, n_tokens)
    struct ggml_tensor * ffn_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_tokens);
    float ffn_hidden_data[n_ff * n_tokens];
    for(int i=0; i<n_ff * n_tokens; ++i) ffn_hidden_data[i] = 1.0f + i; // 1, 2, ..., 8 for n_tokens=1
    memcpy(ffn_hidden->data, ffn_hidden_data, ggml_nbytes(ffn_hidden));

    // --- GGML Computation ---
    // This replicates the logic from llm_build_llama for the Granite FFN down part
    ggml_tensor * inp_ff_x = ggml_view_2d(ctx, ffn_hidden, n_ff_x_cols, n_tokens, ffn_hidden->nb[1], 0);
    ggml_tensor * inp_ff_y = ggml_view_2d(ctx, ffn_hidden, n_ff_y_cols, n_tokens, ffn_hidden->nb[1], n_ff_x_cols * ggml_element_size(ffn_hidden->type));

    ggml_tensor * cur_x = ggml_mul_mat(ctx, layer.ffn_down_x, inp_ff_x);
    ggml_tensor * cur_y = ggml_mul_mat(ctx, layer.ffn_down_y, inp_ff_y);
    ggml_tensor * computed_output_ggml = ggml_add(ctx, cur_x, cur_y);

    if (layer.ffn_down_b) {
        computed_output_ggml = ggml_add(ctx, computed_output_ggml, layer.ffn_down_b);
    }

    // Build graph and compute (minimal)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, computed_output_ggml);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // --- Manual/Reference Calculation ---
    // ffn_down_x (W_dx) is (n_ff_x_cols, n_embd) e.g. (2, 4)
    // ffn_down_y (W_dy) is (n_ff_y_cols, n_embd) e.g. (6, 4)
    // inp_ff_x (X_x) is (n_ff_x_cols, n_tokens) e.g. (2, 1)
    // inp_ff_y (X_y) is (n_ff_y_cols, n_tokens) e.g. (6, 1)
    // ggml_mul_mat(W, X) effectively computes W^T * X if we think of standard matmul.
    // Result C has ne[0]=X->ne[1], ne[1]=W->ne[1]. So C is (n_tokens, n_embd).

    std::vector<float> expected_output_data(hparams.n_embd * n_tokens, 0.0f);

    // Output_x = ffn_down_x^T * inp_ff_x (element-wise)
    // ffn_down_x (2x4) all 1s. Transposed: (4x2)
    // inp_ff_x (2x1) = [1, 2]^T
    // Result_x (4x1)
    // [1 1]   [1]   [1*1 + 1*2]   [3]
    // [1 1] * [2] = [1*1 + 1*2] = [3]
    // [1 1]         [1*1 + 1*2]   [3]
    // [1 1]         [1*1 + 1*2]   [3]
    for (int r = 0; r < hparams.n_embd; ++r) { // iter over rows of output / n_embd
        for (int t = 0; t < n_tokens; ++t) { // iter over columns of output / n_tokens
            float sum_x = 0.0f;
            for (int k = 0; k < n_ff_x_cols; ++k) { // iter over shared dim
                float w_val = ((float*)layer.ffn_down_x->data)[r * n_ff_x_cols + k]; // W_x[k,r] if W_x is (cols,rows)
                float x_val = ((float*)inp_ff_x->data)[t * n_ff_x_cols + k];         // X_x[k,t]
                sum_x += w_val * x_val;
            }
            expected_output_data[t * hparams.n_embd + r] += sum_x;
        }
    }

    // Output_y = ffn_down_y^T * inp_ff_y
    // ffn_down_y (6x4) all 2s. Transposed: (4x6)
    // inp_ff_y (6x1) = [3, 4, 5, 6, 7, 8]^T
    // Result_y (4x1)
    // [2 ... 2] * [3..8]^T = [2*3 + ... + 2*8] = [2 * (3+4+5+6+7+8)] = [2 * 33] = [66]
    // repeated for each of 4 rows
    for (int r = 0; r < hparams.n_embd; ++r) { // iter over rows of output / n_embd
        for (int t = 0; t < n_tokens; ++t) { // iter over columns of output / n_tokens
            float sum_y = 0.0f;
            for (int k = 0; k < n_ff_y_cols; ++k) { // iter over shared dim
                float w_val = ((float*)layer.ffn_down_y->data)[r * n_ff_y_cols + k]; // W_y[k,r]
                float x_val = ((float*)inp_ff_y->data)[t * n_ff_y_cols + k];         // X_y[k,t]
                sum_y += w_val * x_val;
            }
            expected_output_data[t * hparams.n_embd + r] += sum_y;
        }
    }
    // For n_tokens=1, before bias: expected_output_data = [3+66, 3+66, 3+66, 3+66] = [69, 69, 69, 69]

    // Add bias manually
    if (layer.ffn_down_b) {
        float * bias_data = (float *)layer.ffn_down_b->data;
        for (int r = 0; r < hparams.n_embd; ++r) {
            for (int t = 0; t < n_tokens; ++t) {
                expected_output_data[t * hparams.n_embd + r] += bias_data[r];
            }
        }
    }
    // For n_tokens=1, after bias of 0.5: expected_output_data = [69.5, 69.5, 69.5, 69.5]


    // --- Assertions ---
    assert(computed_output_ggml->ne[0] == n_tokens);
    assert(computed_output_ggml->ne[1] == hparams.n_embd);

    float * computed_data = (float *)computed_output_ggml->data;
    for (int i = 0; i < ggml_nelements(computed_output_ggml); ++i) {
        // printf("idx %d: expected %.2f, computed %.2f\n", i, expected_output_data[i], computed_data[i]);
        assert(are_floats_equal(expected_output_data[i], computed_data[i]));
    }

    printf("  Computation test passed.\n");
    ggml_free(ctx);
}


int main(int argc, char ** argv) {
    UNUSED(argc);
    UNUSED(argv);

    test_granite_model_loading_conceptual();
    test_granite_ffn_computation();

    return 0;
}
