#include "../src/llama-dna.h"
#include "../src/llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static constexpr int64_t N_EMBD = 64;
static constexpr int64_t N_FF = 4;
static constexpr int N_LAYERS = 2;

struct llama_model_deleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

struct layer_fixture {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * gate_out = nullptr;
    ggml_tensor * up_out = nullptr;
    ggml_tensor * delta = nullptr;
    ggml_tensor * output = nullptr;
    llama_dna_layer * state = nullptr;
};

static void fill_layer(layer_fixture & f, int il) {
    for (int64_t j = 0; j < N_EMBD; ++j) {
        ((float *) f.x->data)[j] = 0.1f + 0.001f * j;
    }

    float * gate = (float *) f.gate->data;
    float * up = (float *) f.up->data;
    float * down = (float *) f.down->data;
    const float layer_scale = 1.0f + 0.1f * il;
    for (int64_t i = 0; i < N_FF; ++i) {
        for (int64_t j = 0; j < N_EMBD; ++j) {
            gate[i * N_EMBD + j] = layer_scale * 0.01f * (1 + (i + j) % 5);
            up[i * N_EMBD + j] = layer_scale * 0.02f * (1 + (2 * i + j) % 7);
            down[j * N_FF + i] = ((i + j) % 2 ? -3.375f : 3.375f);
        }
    }

    float * gate_out = (float *) f.gate_out->data;
    float * up_out = (float *) f.up_out->data;
    for (int64_t i = 0; i < N_FF; ++i) {
        gate_out[i] = 0.0f;
        up_out[i] = 0.0f;
        for (int64_t j = 0; j < N_EMBD; ++j) {
            gate_out[i] += gate[i * N_EMBD + j] * ((float *) f.x->data)[j];
            up_out[i] += up[i * N_EMBD + j] * ((float *) f.x->data)[j];
        }
    }
}

static ggml_tensor * new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t n0, int64_t n1) {
    ggml_tensor * result = ggml_new_tensor_2d(ctx, type, n0, n1);
    CHECK(result != nullptr);
    return result;
}

static void observe(llama_dna & dna, layer_fixture & f, uint32_t & epoch, bool phase_changed) {
    ggml_tensor dst = {};
    dst.src[0] = f.x;
    dst.src[1] = f.gate_out;
    dst.src[2] = f.up_out;
    llama_dna::op_observe(&dst, 0, 1, f.state);

    const uint32_t next_epoch = dna.update();
    CHECK(next_epoch == epoch + (phase_changed ? 1 : 0));
    epoch = next_epoch;
}

static void check_predict(layer_fixture & f) {
    f.delta->src[0] = f.x;
    llama_dna::op_route(f.delta, 0, 1, f.state);

    const float * gate = (const float *) f.gate->data;
    const float * up = (const float *) f.up->data;
    const float * x = (const float *) f.x->data;
    const float * delta = (const float *) f.delta->data;
    for (int64_t i = 0; i < N_FF; ++i) {
        float g = 0.0f;
        float u = 0.0f;
        for (int64_t j = 0; j < N_EMBD; ++j) {
            g += gate[i * N_EMBD + j] * x[j];
            u += up[i * N_EMBD + j] * x[j];
        }
        const float expected = (g / (1.0f + std::exp(-g))) * u;
        CHECK(std::isfinite(delta[i]));
        CHECK(std::fabs(delta[i] - expected) < 1e-5f);
    }

    f.output->src[0] = f.delta;
    llama_dna::op_down(f.output, 0, 1, f.state);

    const float * down = (const float *) f.down->data;
    const float * output = (const float *) f.output->data;
    for (int64_t j = 0; j < N_EMBD; ++j) {
        float expected = 0.0f;
        for (int64_t i = 0; i < N_FF; ++i) {
            expected += delta[i] * down[j * N_FF + i];
        }
        CHECK(std::isfinite(output[j]));
        CHECK(std::fabs(output[j] - expected) < 1e-5f);
    }
}

int main(void) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    CHECK(backend != nullptr);

    std::unique_ptr<llama_model, llama_model_deleter> model(
        llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    CHECK(model != nullptr);
    model->layers.resize(N_LAYERS);

    ggml_context_ptr ctx(ggml_init({ 1 << 20, nullptr, true }));
    CHECK(ctx != nullptr);

    layer_fixture layers[N_LAYERS];
    for (int il = 0; il < N_LAYERS; ++il) {
        layer_fixture & f = layers[il];
        f.gate = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_EMBD, N_FF);
        f.up = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_EMBD, N_FF);
        f.down = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_FF, N_EMBD);
        f.x = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_EMBD, 1);
        f.gate_out = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_FF, 1);
        f.up_out = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_FF, 1);
        f.delta = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_FF, 1);
        f.output = new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_EMBD, 1);

        model->layers[il].ffn_gate = f.gate;
        model->layers[il].ffn_up = f.up;
        model->layers[il].ffn_down = f.down;
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    CHECK(buffer != nullptr);

    llama_dna_params params;
    params.n_calib = 2;
    params.thresh = 0.0f;
    params.min_active = 0.0f;
    params.pred_type = GGML_TYPE_Q8_0;
    llama_dna dna(*model, backend.get(), params);

    for (int il = 0; il < N_LAYERS; ++il) {
        layer_fixture & f = layers[il];
        fill_layer(f, il);
        f.state = dna.get_layer(
            il, f.gate, f.up, f.down,
            nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            LLAMA_DNA_ACT_SILU);
        CHECK(f.state != nullptr);
    }

    uint32_t epoch = dna.update();
    CHECK(dna.get_stats().n_phase_stats == N_LAYERS);

    observe(dna, layers[0], epoch, false);
    observe(dna, layers[1], epoch, false);
    CHECK(dna.get_stats().n_phase_stats == N_LAYERS);

    observe(dna, layers[1], epoch, true);
    CHECK(dna.get_stats().n_phase_stats == 1);
    CHECK(dna.get_stats().n_phase_bias == 1);

    observe(dna, layers[1], epoch, false);
    observe(dna, layers[0], epoch, true);
    CHECK(dna.get_stats().n_phase_stats == 0);
    CHECK(dna.get_stats().n_phase_bias == N_LAYERS);

    observe(dna, layers[0], epoch, false);
    observe(dna, layers[1], epoch, true);
    CHECK(dna.get_stats().n_phase_predict == 1);
    CHECK(dna.get_stats().n_phase_bias == 1);
    check_predict(layers[1]);

    observe(dna, layers[0], epoch, true);
    CHECK(dna.get_stats().n_phase_predict == N_LAYERS);
    for (int il = 0; il < N_LAYERS; ++il) {
        CHECK(llama_dna::layer_is_predict(layers[il].state));
        check_predict(layers[il]);
    }

    std::printf("DNA lifecycle test passed\n");
    return 0;
}
