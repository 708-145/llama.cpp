#pragma once

// DNA - Dynamic Neuron Activation
//
// Runtime calibrated neuron skipping for dense SwiGLU FFNs on the CPU backend.
// A binary (sign + block scale) estimate of the gate projection routes each neuron to silent or active.
// Active neurons get the full precision gate/up/down, silent neurons contribute a calibrated bias.
// Calibration runs on the first tokens of normal inference, no model file or API change is needed.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <map>
#include <memory>

struct llama_model;
struct llama_dna_layer;

struct llama_dna_params {
    float    thresh     = 0.5f;  // dead zone half-width, in units of mean |g~| or per-neuron std (see zmode)
    int      block      = 32;    // estimator block size, 8, 16, 32 or 64
    uint32_t n_calib    = 4096;  // calibration tokens per phase
    float    min_active = 0.02f; // floor on the active fraction per layer
    int      zmode      = 0;     // 0: band = thresh * mean |g~|, 1: band = thresh * per-neuron std of g~
    int      max_batch  = 32;    // larger batches use the dense FFN, skipping neurons only pays off when memory bound
    int      verbose    = 0;

    // env: LLAMA_DNA_T, LLAMA_DNA_BLOCK, LLAMA_DNA_CALIB, LLAMA_DNA_MIN_ACTIVE, LLAMA_DNA_ZMODE, LLAMA_DNA_MAX_BATCH, LLAMA_DNA_VERBOSE
    static llama_dna_params from_env();
};

class llama_dna {
public:
    llama_dna(const llama_model & model, ggml_backend_t backend_cpu, const llama_dna_params & params);
    ~llama_dna();

    // state for the dense FFN of layer il, nullptr when the layer cannot use DNA
    llama_dna_layer * get_layer(int il, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down);

    // apply pending calibration results, returns a counter that changes when the graph topology must change
    uint32_t update();

    static bool layer_is_predict(const llama_dna_layer * layer);

    // ggml custom ops, userdata is the llama_dna_layer
    static void op_observe(ggml_tensor * dst, int ith, int nth, void * userdata); // src: x, gate out, up out
    static void op_route  (ggml_tensor * dst, int ith, int nth, void * userdata); // src: x, gate w, up w -> delta [n_ff, n_tokens]
    static void op_down   (ggml_tensor * dst, int ith, int nth, void * userdata); // src: delta -> y [n_embd, n_tokens]

    void print_stats() const;

    const llama_dna_params params;

private:
    typedef const ggml_type_traits_cpu * (*get_type_traits_cpu_t)(enum ggml_type);

    const llama_model & model;

    get_type_traits_cpu_t get_traits = nullptr;

    // keyed by the gate tensor, nullptr for rejected layers
    std::map<const ggml_tensor *, std::unique_ptr<llama_dna_layer>> layers;

    uint32_t epoch = 1;
};
