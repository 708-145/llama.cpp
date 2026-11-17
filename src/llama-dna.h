#pragma once

// DNA - Dynamic Neuron Activation
//
// Runtime calibrated neuron skipping for dense and MoE FFNs on the CPU backend.
// An approximate gate projection (IQ4_XS by default, or any ggml quant reusing the
// native CPU vec_dot kernel, selected via --dnapred) routes each neuron to silent
// or active. Active neurons get the full precision gate/up/down, silent neurons
// contribute a calibrated bias.
// Calibration runs on the first tokens of normal inference, no model file or API change is needed.
//
// Supported affine terms (Restrictions 3, 4):
//   up_b / gate_b  / down_b  : per-neuron / per-output biases, stored as f32 vectors
//   up_s / gate_s  / down_s  : per-tensor scalar scales, shape {1}; applied in dense graph order
//                              (bias then scale for gate/up, scale after down projection for down)
// Activations (Restriction 7): non-F32 inputs are cast to F32 in the graph before the DNA ops;
//                              the kernels always see F32.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

enum llama_dna_phase {
    LLAMA_DNA_PHASE_STATS,   // per-neuron gate statistics and estimator bias
    LLAMA_DNA_PHASE_BIAS,    // silent-neuron bias mu with the final dead zone rule
    LLAMA_DNA_PHASE_PREDICT,
};

struct llama_dna_layer;
struct llama_dna_moe_layer;

struct llama_dna_moe_layer {
    int il = -1;
    int n_expert = 0;
    int n_expert_used = 0;
    std::vector<llama_dna_layer *> experts;
};

struct llama_dna_stats {
    bool enabled = false;
    int  n_layers_on  = 0;
    int  n_layers_off = 0;
    int  n_phase_stats   = 0;
    int  n_phase_bias    = 0;
    int  n_phase_predict = 0;
    uint32_t calib_done = 0;
    uint32_t calib_total = 0;
    double act_min = 0.0;
    double act_max = 0.0;
    double act_avg = 0.0;
    uint64_t pred_tokens = 0;
};

// predictor kind: kQuant = re-quantized gate weights in any ggml type with a CPU
// vec_dot kernel (default IQ4_XS; the XOR sign predictor is known-bad and removed)
enum llama_dna_pred_kind {
    LLAMA_DNA_PRED_QUANT = 0,
};

// parse a predictor name (a ggml type name such as "iq4_xs", case-insensitive)
// returns false when the name is not a ggml type
bool llama_dna_pred_from_name(const char * name, enum llama_dna_pred_kind & kind, enum ggml_type & type);

// gate activation applied to g before the up multiply (dna_act): SILU = silu(g),
// GELU = tanh-approximation gelu(g) as used by ggml_geglu_split
enum llama_dna_act {
    LLAMA_DNA_ACT_SILU = 0,
    LLAMA_DNA_ACT_GELU,
    LLAMA_DNA_ACT_GELU_ERF,
    LLAMA_DNA_ACT_RELU,
};

struct llama_dna_params {
    float    thresh     = 0.5f;  // dead zone half-width, in units of mean |g~| or per-neuron std (see zmode)
    uint32_t n_calib    = 256;   // calibration tokens per phase (calibration runs on the first tokens of inference)
    float    min_active = 0.02f; // floor on the active fraction per layer
    int      zmode      = 0;     // 0: band = thresh * mean |g~|, 1: band = thresh * per-neuron std of g~
    int      max_batch  = 32;    // larger batches use the dense FFN, skipping neurons only pays off when memory bound
    int      verbose    = 0;
    enum llama_dna_pred_kind pred_kind = LLAMA_DNA_PRED_QUANT;
    enum ggml_type           pred_type = GGML_TYPE_IQ4_XS; // quant predictor type

    // env: LLAMA_DNA_T, LLAMA_DNA_CALIB, LLAMA_DNA_MIN_ACTIVE, LLAMA_DNA_ZMODE, LLAMA_DNA_MAX_BATCH, LLAMA_DNA_VERBOSE, LLAMA_DNA_PRED
    static llama_dna_params from_env();

    // select the predictor by name (ggml type name); throws std::runtime_error on unknown names
    void set_pred_from_name(const char * name);
};

class llama_dna {
public:
    llama_dna(const llama_model & model, ggml_backend_t backend_cpu, const llama_dna_params & params);
    ~llama_dna();

    // state for the dense FFN of layer il, nullptr when the layer cannot use DNA
    // the *_b / *_s tensors may be nullptr (absent affine terms); non-null ones must be
    // readable f32 host tensors with shapes gate_b/up_b={n_ff}, down_b={n_embd}, *_s={1}
    llama_dna_layer * get_layer(int il,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * gate_b, const ggml_tensor * up_b, const ggml_tensor * down_b,
        const ggml_tensor * gate_s, const ggml_tensor * up_s, const ggml_tensor * down_s,
        enum llama_dna_act act);

    // state for the MoE experts of layer il. The per-expert dna_layers live in
    // `layers` under synthetic keys (the expert's down_exps row pointer range);
    // the returned holder only carries the expert count for the graph.
    llama_dna_layer * get_moe_expert(int il, int expert,
        const ggml_tensor * gate_up_exps, const ggml_tensor * gate_exps, const ggml_tensor * up_exps,
        const ggml_tensor * down_exps, enum llama_dna_act act);

    // state for the shared dense FFN, nullptr when the layer cannot use DNA
    // (same affine rules as get_layer; act is the shared FFN's gate activation;
    //  stored in `layers` under the shared ffn_gate key, not a MoE holder)
    llama_dna_layer * get_shared_layer(int il,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * gate_b, const ggml_tensor * up_b, const ggml_tensor * down_b,
        const ggml_tensor * gate_s, const ggml_tensor * up_s, const ggml_tensor * down_s,
        enum llama_dna_act act);

    // MoE holder for layer il (fallback for kernels that need the holder). The
    // common path is per-expert get_moe_expert + the static ops below.
    llama_dna_moe_layer * get_moe_layer(int il,
        const ggml_tensor * gate_up_exps, const ggml_tensor * gate_exps, const ggml_tensor * up_exps,
        const ggml_tensor * down_exps, enum llama_dna_act act);

    static bool moe_is_predict(const llama_dna_moe_layer * moe);

    // per-expert routed delta: userdata is the llama_dna_moe_layer*, src0 = x,
    // src1 = gate_up_exps (or gate_exps), src2 = selected_experts ->
    // delta [n_ff, n_expert_used, n_tokens]
    static void op_expert_route(ggml_tensor * dst, int ith, int nth, void * userdata);
    // weighted down: userdata is the llama_dna_moe_layer*, src0 = delta,
    // src1 = router weights [1, n_expert_used, n_tokens], src2 = selected_experts ->
    // y [n_embd, n_tokens] = sum_s w[s] * (D[e_s] @ delta[:,s] + c0[e_s])
    static void op_expert_down(ggml_tensor * dst, int ith, int nth, void * userdata);
    // calibration observe: userdata is the llama_dna_moe_layer*, src0 = x,
    // src1 = gate out, src2 = up out, src3 = selected_experts
    static void op_expert_observe(ggml_tensor * dst, int ith, int nth, void * userdata);

    // apply pending calibration results, returns a counter that changes when the graph topology must change
    uint32_t update();

    static bool layer_is_predict(const llama_dna_layer * layer);

    // ggml custom ops, userdata is the llama_dna_layer
    static void op_observe(ggml_tensor * dst, int ith, int nth, void * userdata); // src: x, gate out, up out
    static void op_route  (ggml_tensor * dst, int ith, int nth, void * userdata); // src: x, gate w, up w -> delta [n_ff, n_tokens]
    static void op_down   (ggml_tensor * dst, int ith, int nth, void * userdata); // src: delta -> y [n_embd, n_tokens]

    void print_stats() const;
    void print_turn_stats() const;
    llama_dna_stats get_stats() const;

    const llama_dna_params params;

private:
    bool all_layers_at_least(uint32_t need, llama_dna_phase phase) const;

    typedef const ggml_type_traits_cpu * (*get_type_traits_cpu_t)(enum ggml_type);

    const llama_model & model;

    get_type_traits_cpu_t get_traits = nullptr;

    friend void llama_dna_init_layer_traits(llama_dna *, llama_dna_layer &);

    // keyed by the gate tensor (dense/shared) or by the MoE holder key
    // (down_exps data pointer range for experts), nullptr entries are rejected layers
    std::map<const ggml_tensor *, std::unique_ptr<llama_dna_layer>> layers;
    std::map<const ggml_tensor *, std::unique_ptr<llama_dna_moe_layer>> moe_layers;

    uint32_t epoch = 1;
};
