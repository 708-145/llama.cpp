#include "llama-dna.h"

#include "dna-iq3-cxs2.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define LLAMA_DNA_X86 1
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static constexpr int LLAMA_DNA_RESERVOIR = 256;

struct llama_dna_layer {
    const llama_dna_params * params = nullptr;

    int il = -1;

    int64_t n_embd = 0;
    int64_t n_ff   = 0;

    enum llama_dna_act act = LLAMA_DNA_ACT_SILU;

    // MoE expert fields (when is_moe): expert index and the two slices of the
    // concatenated gate_up_exps tensor. For dense layers gate/up/down point at the
    // layer's ffn_gate/ffn_up/ffn_down tensors.
    bool is_moe = false;
    int expert = -1;
    const ggml_tensor * gate_up_exps = nullptr; // [n_embd, 2*n_ff, n_expert] PAR
    int64_t n_ff_expert_rows = 0; // n_ff for gate_up_exps row layout

    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up   = nullptr;
    const ggml_tensor * down = nullptr;

    // affine terms (Restrictions 3, 4), copied out at get_layer time; empty = absent
    std::vector<float> gate_b; // [n_ff]
    std::vector<float> up_b;   // [n_ff]
    std::vector<float> down_b; // [n_embd]
    float gate_s = 1.0f; // per-tensor scalar, applied after gate_b
    float up_s   = 1.0f; // per-tensor scalar, applied after up_b
    float down_s = 1.0f; // per-tensor scalar, applied to the down projection result

    ggml_vec_dot_t   vec_dot_gate    = nullptr;
    ggml_vec_dot_t   vec_dot_up      = nullptr;
    ggml_from_float_t from_float_gate = nullptr;
    ggml_from_float_t from_float_up   = nullptr;
    enum ggml_type   vdt_gate = GGML_TYPE_F32;
    enum ggml_type   vdt_up   = GGML_TYPE_F32;

    llama_dna_phase phase = LLAMA_DNA_PHASE_STATS;

    uint32_t n_tok      = 0; // tokens seen in the current phase, synced by llama_dna::update()
    uint32_t n_tok_next = 0; // running total written by op_observe, applied by llama_dna::update()
    uint32_t n_slot     = 0; // tokens already consumed by op_* (reservoir indices)
    uint32_t n_read     = 0; // last synced n_tok (for n_slot)

    // packed gate estimator (quant predictor, --dnapred <type>): re-quantized gate rows
    // in pred_type layout, plus the native CPU vec_dot / from_float for pred_type
    std::vector<uint8_t>     pred_q; // [n_ff][pred_row_size]
    ggml_vec_dot_t           pred_vec_dot     = nullptr;
    ggml_from_float_t        pred_from_float  = nullptr; // quantizes the F32 input x (pred vec_dot_type)
    enum ggml_type           pred_vdt = GGML_TYPE_F32;
    size_t                   pred_row_size = 0;

    // calibration accumulators, one entry per neuron
    std::vector<float>    res;     // [n_ff][R] reservoir of g (stats phase), then of dev/band (bias phase)
    std::vector<float>    res_est; // [n_ff][R] reservoir of g~
    std::vector<double>   sum_g, sum_g2, sum_ge, sum_ge2, sum_gge, sum_abs_ge;
    std::vector<double>   mu_sum, sum_h2, sum_h2_act;
    std::vector<uint32_t> mu_cnt, n_act;

    // calibrated
    std::vector<float>       med_est; // [n_ff] per-neuron bias proxy in estimator units
    std::vector<float>       band;    // [n_ff] dead zone half-width
    std::vector<float>       mu;      // [n_ff] mean h when silent
    std::vector<float>       c0;      // [n_embd] down projection of mu
    std::vector<dna_iq3_cxs2_block> down_blocks; // [n_ff][down_blocks_per_neuron]
    std::vector<uint8_t> down_scales;            // [n_ff][8] E5M3 scale table
    size_t down_blocks_per_neuron = 0;

    float m_est = 0.0f; // mean |g~|

    std::atomic<uint64_t> pred_tok{0};
    std::atomic<uint64_t> pred_act{0};

    // wall time of thread 0 in each op, for the verbose stats
    int64_t t_observe_us = 0;
    int64_t t_route_us   = 0;
    int64_t t_down_us    = 0;
};

//
// helpers
//

static std::vector<float> g_f16_table; // fp16 -> fp32, filled once

#if defined(LLAMA_DNA_X86)
static bool g_x86_fast = false; // avx2 + fma + f16c + popcnt
#endif

static inline float llama_dna_silu(float x) {
    return x / (1.0f + expf(-x));
}

// GeGLU uses the tanh approximation, exactly as ggml_vec_gelu_f32 in ggml-cpu/vec.h:
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
static inline float llama_dna_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

static inline float llama_dna_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static inline float llama_dna_act(enum llama_dna_act act, float x) {
    switch (act) {
        case LLAMA_DNA_ACT_SILU:     return llama_dna_silu(x);
        case LLAMA_DNA_ACT_GELU:     return llama_dna_gelu(x);
        case LLAMA_DNA_ACT_GELU_ERF: return llama_dna_gelu_erf(x);
        case LLAMA_DNA_ACT_RELU:     return x > 0.0f ? x : 0.0f;
    }
    return x;
}

static void llama_dna_f16_to_f32_generic(const ggml_fp16_t * x, float * y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = g_f16_table[x[i]];
    }
}

#if defined(LLAMA_DNA_X86)
__attribute__((target("avx2,fma,f16c")))
static void llama_dna_f16_to_f32_x86(const ggml_fp16_t * x, float * y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))));
    }
    llama_dna_f16_to_f32_generic(x + i, y + i, n - i);
}
#endif

static inline void llama_dna_f16_to_f32(const ggml_fp16_t * x, float * y, int n) {
#if defined(LLAMA_DNA_X86)
    if (g_x86_fast) {
        llama_dna_f16_to_f32_x86(x, y, n);
        return;
    }
#endif
    llama_dna_f16_to_f32_generic(x, y, n);
}

// quant predictor: native CPU vec_dot of the re-quantized gate row against the
// already quantized input row xq (pred_vdt layout); mirrors llama_dna_estimate
static inline float llama_dna_estimate_quant(const llama_dna_layer & L, int64_t i, const void * xq) {
    float ge = 0.0f;
    L.pred_vec_dot((int) L.n_embd, &ge, 0,
            (const char *) L.pred_q.data() + i*L.pred_row_size, 0,
            (const char *) xq, 0, 1);
    return ge;
}

static bool llama_dna_uses_flat_imatrix(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        default:
            return false;
    }
}

// reservoir sampling, k is the sample index, all neurons see the same k
static inline void llama_dna_reservoir_push(float * res, uint32_t salt, uint32_t k, float v) {
    if (k < LLAMA_DNA_RESERVOIR) {
        res[k] = v;
        return;
    }
    uint32_t h = salt * 2654435761u ^ (k * 2246822519u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    const uint32_t r = h % (k + 1);
    if (r < LLAMA_DNA_RESERVOIR) {
        res[r] = v;
    }
}

static float llama_dna_median(float * v, int n) {
    if (n <= 0) {
        return 0.0f;
    }
    std::nth_element(v, v + n/2, v + n);
    return v[n/2];
}

static void llama_dna_row_to_float(const ggml_tensor * w, int64_t row, float * dst) {
    const void * src = (const char *) w->data + row*w->nb[1];
    if (w->type == GGML_TYPE_F32) {
        memcpy(dst, src, w->ne[0]*sizeof(float));
        return;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(w->type);
    GGML_ASSERT(tr->to_float);
    tr->to_float(src, dst, w->ne[0]);
}

static void llama_dna_down_row_to_float(
    const llama_dna_layer & L,
    int64_t                 row,
    float                 * dst)
{
    const size_t expert_offset = L.is_moe ? (size_t) L.expert * L.down->nb[2] : 0;
    const char * src =
        (const char *) L.down->data + expert_offset + (size_t) row * L.down->nb[1];
    if (L.down->type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) L.down->ne[0] * sizeof(float));
        return;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(L.down->type);
    GGML_ASSERT(tr->to_float);
    tr->to_float(src, dst, L.down->ne[0]);
}

// read a 1-D dense tensor (bias {n} or scale {1}) into an f32 vector; rejects non-host / unreadable tensors
// forward-declared: llama_dna_row_is_readable is defined below
static bool llama_dna_row_is_readable(const ggml_tensor * w);

static bool llama_dna_read_dense(const ggml_tensor * w, int64_t n_expect, std::vector<float> & dst) {
    if (!w) {
        dst.clear();
        return true;
    }
    if (w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] != n_expect) {
        return false;
    }
    if (!llama_dna_row_is_readable(w)) {
        return false;
    }
    dst.resize((size_t) n_expect);
    llama_dna_row_to_float(w, 0, dst.data());
    return true;
}

// read a scalar scale tensor {1}; absent -> 1.0f
static bool llama_dna_read_scale(const ggml_tensor * w, float & s) {
    s = 1.0f;
    if (!w) {
        return true;
    }
    std::vector<float> v;
    if (!llama_dna_read_dense(w, 1, v)) {
        return false;
    }
    s = v[0];
    return true;
}

static bool llama_dna_row_is_readable(const ggml_tensor * w) {
    if (!w || !w->data || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
        return false;
    }
    // repacked extra buffer types have a different data layout
    const std::string name = ggml_backend_buffer_name(w->buffer);
    if (name != "CPU" && name != "CPU_Mapped") {
        return false;
    }
    return w->type == GGML_TYPE_F32 || ggml_get_type_traits(w->type)->to_float != nullptr;
}

static void llama_dna_pack_gate(llama_dna_layer & L) {
    std::vector<float> wf(L.n_embd);

    // IQ types without a reference row quantizer need an imatrix; use unit
    // weights so DNA can pack gate rows without model-specific calibration data.
    const ggml_type_traits * tr = ggml_get_type_traits(L.params->pred_type);
    const bool use_flat_imatrix = llama_dna_uses_flat_imatrix(L.params->pred_type);
    GGML_ASSERT(use_flat_imatrix || tr->from_float_ref);
    std::vector<float> flat_imatrix;
    if (use_flat_imatrix) {
        flat_imatrix.assign((size_t) L.n_embd, 1.0f);
    }
    L.pred_q.assign((size_t) L.n_ff*L.pred_row_size, 0);
    for (int64_t i = 0; i < L.n_ff; ++i) {
        llama_dna_row_to_float(L.gate, i, wf.data());
        void * dst = L.pred_q.data() + i*L.pred_row_size;
        if (use_flat_imatrix) {
            ggml_quantize_chunk(L.params->pred_type, wf.data(), dst, 0, 1, L.n_embd, flat_imatrix.data());
        } else {
            tr->from_float_ref(wf.data(), dst, L.n_embd);
        }
    }
}

// predictor memory size in bytes for the pack log line
static size_t llama_dna_pred_bytes(const llama_dna_layer & L) {
    return L.pred_q.size();
}

static void llama_dna_finalize_stats(llama_dna_layer & L) {
    const int R = LLAMA_DNA_RESERVOIR;
    const int n = (int) std::min<uint32_t>(L.n_tok, R);
    const double nt = L.n_tok;

    L.med_est.resize(L.n_ff);
    L.band.resize(L.n_ff);

    double acc_abs_ge = 0.0;
    for (int64_t i = 0; i < L.n_ff; ++i) {
        acc_abs_ge += L.sum_abs_ge[i];
    }
    L.m_est = (float) (acc_abs_ge / (nt * L.n_ff));

    double sum_bias = 0.0, sum_sd = 0.0, sum_sd_est = 0.0, sum_corr = 0.0;
    int n_corr = 0;
    for (int64_t i = 0; i < L.n_ff; ++i) {
        const float  med    = llama_dna_median(L.res.data() + i*R, n);
        const double mean_g = L.sum_g[i] / nt;
        const double var_g  = std::max(0.0, L.sum_g2[i] / nt - mean_g*mean_g);
        const double sd     = sqrt(var_g);

        const double mean_ge = L.sum_ge[i] / nt;
        const double var_ge  = std::max(0.0, L.sum_ge2[i] / nt - mean_ge*mean_ge);
        const double sd_est  = sqrt(var_ge);

        if (sd > 0.0 && sd_est > 0.0) {
            sum_corr += (L.sum_gge[i] / nt - mean_g*mean_ge) / (sd*sd_est);
            n_corr++;
            sum_bias += fabs(med) / sd;
        }
        sum_sd     += sd;
        sum_sd_est += sd_est;

        L.med_est[i] = llama_dna_median(L.res_est.data() + i*R, n);
        if (L.params->zmode) {
            L.band[i] = L.params->thresh * (float) std::max(sd_est, 1e-3*L.m_est);
        } else {
            L.band[i] = L.params->thresh * L.m_est;
        }
    }

    LLAMA_LOG_ERROR("%s: layer %3d: stats over %u tokens: |med|/sd = %.2f, sd = %.3f, sd_est = %.3f, mean |g~| = %.3f, corr(g, g~) = %.3f\n",
            __func__, L.il, L.n_tok, sum_bias / std::max(1, n_corr), sum_sd / L.n_ff, sum_sd_est / L.n_ff, L.m_est, sum_corr / std::max(1, n_corr));

    std::vector<float>().swap(L.res_est);
    std::vector<double>().swap(L.sum_g);
    std::vector<double>().swap(L.sum_g2);
    std::vector<double>().swap(L.sum_ge);
    std::vector<double>().swap(L.sum_ge2);
    std::vector<double>().swap(L.sum_gge);
    std::vector<double>().swap(L.sum_abs_ge);

    std::fill(L.res.begin(), L.res.end(), 0.0f);
    L.mu_sum    .assign(L.n_ff, 0.0);
    L.sum_h2    .assign(L.n_ff, 0.0);
    L.sum_h2_act.assign(L.n_ff, 0.0);
    L.mu_cnt    .assign(L.n_ff, 0);
    L.n_act     .assign(L.n_ff, 0);
}

static void llama_dna_finalize_bias(llama_dna_layer & L) {
    const int R = LLAMA_DNA_RESERVOIR;
    const int n = (int) std::min<uint32_t>(L.n_tok, R);

    L.mu.resize(L.n_ff);

    double sum_h2 = 0.0, sum_h2_act = 0.0, sum_h2_mu = 0.0;
    uint64_t n_act = 0;
    for (int64_t i = 0; i < L.n_ff; ++i) {
        L.mu[i] = L.mu_cnt[i] > 0 ? (float) (L.mu_sum[i] / L.mu_cnt[i]) : 0.0f;
        sum_h2     += L.sum_h2[i];
        sum_h2_act += L.sum_h2_act[i];
        sum_h2_mu  += (double) L.mu_cnt[i] * L.mu[i] * L.mu[i];
        n_act      += L.n_act[i];
    }

    const double act_frac = (double) n_act / ((double) L.n_tok * L.n_ff);

    // the dead zone band is shrunk when fewer than min_active neurons are active
    float band_scale = 1.0f;
    if (act_frac < L.params->min_active && n > 0) {
        std::vector<float> dev;
        dev.reserve((size_t) L.n_ff * n);
        for (int64_t i = 0; i < L.n_ff; ++i) {
            dev.insert(dev.end(), L.res.begin() + i*R, L.res.begin() + i*R + n);
        }
        const size_t pos = (size_t) ((1.0 - L.params->min_active) * (dev.size() - 1));
        std::nth_element(dev.begin(), dev.begin() + pos, dev.end());
        band_scale = std::min(1.0f, dev[pos]);
        for (int64_t i = 0; i < L.n_ff; ++i) {
            L.band[i] *= band_scale;
        }
    }

    // c0 = down_s * (Wdown @ mu + down_b). The sparse delta projection stores
    // down_s * Wdown in IQ3_CXS2 blocks, so op_down needs no extra multiply.
    L.c0.assign(L.n_embd, 0.0f);
    L.down_blocks_per_neuron =
        ((size_t) L.n_embd + DNA_IQ3_CXS2_WEIGHTS_PER_LINE - 1) /
        DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t padded_n_weights =
        L.down_blocks_per_neuron * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    L.down_blocks.resize((size_t) L.n_ff * L.down_blocks_per_neuron);
    L.down_scales.resize((size_t) L.n_ff * 8);

    std::vector<float> rowf(L.n_ff);
    std::vector<float> rowscaled(L.n_ff);
    std::vector<ggml_fp16_t> rowh(L.n_ff);
    std::vector<ggml_fp16_t> down_scaled_t((size_t) L.n_ff * L.n_embd);
    for (int64_t j = 0; j < L.n_embd; ++j) {
        llama_dna_down_row_to_float(L, j, rowf.data());
        for (int64_t i = 0; i < L.n_ff; ++i) {
            rowscaled[i] = rowf[i] * L.down_s;
        }
        ggml_fp32_to_fp16_row(rowscaled.data(), rowh.data(), L.n_ff);
        double c = 0.0;
        const double bj = L.down_b.empty() ? 0.0 : (double) L.down_b[(size_t) j];
        for (int64_t i = 0; i < L.n_ff; ++i) {
            down_scaled_t[(size_t) i * L.n_embd + j] = rowh[i];
            c += (double) rowf[i] * L.mu[i];
        }
        L.c0[j] = (float) ((c + bj) * (double) L.down_s);
    }

    GGML_ASSERT(L.n_embd <= 0x7fffffff);
    std::vector<float> quant_row(padded_n_weights, 0.0f);
    for (int64_t i = 0; i < L.n_ff; ++i) {
        float * row = quant_row.data();
        llama_dna_f16_to_f32(
            down_scaled_t.data() + (size_t) i * L.n_embd,
            row,
            (int) L.n_embd);
        std::fill(row + L.n_embd, row + padded_n_weights, 0.0f);
        dna_iq3_cxs2_quantize_row(
            row,
            padded_n_weights,
            L.down_blocks.data() + (size_t) i * L.down_blocks_per_neuron,
            L.down_scales.data() + (size_t) i * 8);
    }

    const size_t down_bytes =
        L.down_blocks.size() * sizeof(dna_iq3_cxs2_block) + L.down_scales.size();
    LLAMA_LOG_ERROR("%s: layer %3d: bias over %u tokens: active = %.1f%%, energy(active) = %.1f%%, energy(bias) = %.1f%%, band scale = %.3f, down IQ3_CXS2 = %.1f MiB (%.3f bpw)\n",
            __func__, L.il, L.n_tok, 100.0*act_frac, 100.0*sum_h2_act / std::max(sum_h2, 1e-30), 100.0*sum_h2_mu / std::max(sum_h2, 1e-30),
            band_scale, down_bytes / (1024.0*1024.0),
            8.0 * down_bytes / ((double) L.n_ff * L.n_embd));

    std::vector<float>().swap(L.res);
    std::vector<double>().swap(L.mu_sum);
    std::vector<double>().swap(L.sum_h2);
    std::vector<double>().swap(L.sum_h2_act);
    std::vector<uint32_t>().swap(L.mu_cnt);
    std::vector<uint32_t>().swap(L.n_act);
}

//
// llama_dna_params
//

// locale independent, tools may call setlocale() which changes the decimal separator for atof()
static float llama_dna_env_float(const char * name, float def) {
    const char * v = getenv(name);
    float res = def;
    if (v && std::from_chars(v, v + strlen(v), res).ec != std::errc()) {
        throw std::runtime_error(std::string("DNA: invalid value for ") + name);
    }
    return res;
}

static int llama_dna_env_int(const char * name, int def) {
    const char * v = getenv(name);
    int res = def;
    if (v && std::from_chars(v, v + strlen(v), res).ec != std::errc()) {
        throw std::runtime_error(std::string("DNA: invalid value for ") + name);
    }
    return res;
}

bool llama_dna_pred_from_name(const char * name, enum llama_dna_pred_kind & kind, enum ggml_type & type) {
    if (!name || !*name) {
        kind = LLAMA_DNA_PRED_QUANT;
        type = GGML_TYPE_IQ4_XS; // IQ4_XS is the default predictor
        return true;
    }
    std::string s(name);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (s == ggml_type_name((enum ggml_type) t)) {
            kind = LLAMA_DNA_PRED_QUANT;
            type = (enum ggml_type) t;
            return true;
        }
    }
    return false;
}

void llama_dna_params::set_pred_from_name(const char * name) {
    enum llama_dna_pred_kind kind;
    enum ggml_type type = GGML_TYPE_IQ4_XS;
    if (!llama_dna_pred_from_name(name, kind, type)) {
        throw std::runtime_error(std::string("DNA: unknown predictor \"") + (name ? name : "") +
                "\", expected a ggml type name (e.g. iq4_xs, q8_0, f16)");
    }
    pred_kind = kind;
    pred_type = type;
}

llama_dna_params llama_dna_params::from_env() {
    llama_dna_params p;
    // legacy env path (kept for compatibility). Primary entry is:
    //   --dna (LLAMA_ARG_DNA) -> common_params.dna -> llama_context_params.dna
    // DNA tuning knobs remain env-overridable here.
    p.thresh     = llama_dna_env_float("LLAMA_DNA_T",          p.thresh);
    p.n_calib    = llama_dna_env_int  ("LLAMA_DNA_CALIB",      p.n_calib);
    p.min_active = llama_dna_env_float("LLAMA_DNA_MIN_ACTIVE", p.min_active);
    p.zmode      = llama_dna_env_int  ("LLAMA_DNA_ZMODE",      p.zmode);
    p.max_batch  = llama_dna_env_int  ("LLAMA_DNA_MAX_BATCH",  p.max_batch);
    p.verbose    = llama_dna_env_int  ("LLAMA_DNA_VERBOSE",    p.verbose);
    if (const char * v = getenv("LLAMA_DNA_PRED"); v && *v) {
        p.set_pred_from_name(v);
    }
    return p;
}

//
// llama_dna
//

llama_dna::llama_dna(const llama_model & model, ggml_backend_t backend_cpu, const llama_dna_params & params) : params(params), model(model) {
    {
        const ggml_type_traits * tr = ggml_get_type_traits(params.pred_type);
        if (!tr || (!tr->from_float_ref && !llama_dna_uses_flat_imatrix(params.pred_type))) {
            throw std::runtime_error(std::string("DNA: predictor type ") + ggml_type_name(params.pred_type) +
                    " has no supported CPU quantization path");
        }
    }
    if (params.n_calib == 0 || params.thresh < 0.0f) {
        throw std::runtime_error("DNA: invalid calibration parameters");
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    get_traits = reg ? (get_type_traits_cpu_t) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu") : nullptr;
    if (!get_traits) {
        throw std::runtime_error("DNA: CPU backend does not expose ggml_get_type_traits_cpu");
    }

    if (g_f16_table.empty()) {
        g_f16_table.resize(1 << 16);
        for (int i = 0; i < (1 << 16); ++i) {
            g_f16_table[i] = ggml_fp16_to_fp32((ggml_fp16_t) i);
        }
    }

#if defined(LLAMA_DNA_X86)
    g_x86_fast = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("popcnt");
#endif

    bool fast = false;
#if defined(LLAMA_DNA_X86)
    fast = g_x86_fast;
#endif
    LLAMA_LOG_ERROR("%s: DNA enabled: pred = %s, T = %.2f, calib = %u tokens/phase, min_active = %.1f%%, zmode = %d, max_batch = %d, simd = %d\n",
            __func__,
            ggml_type_name(params.pred_type),
            params.thresh, params.n_calib, 100.0f*params.min_active, params.zmode, params.max_batch, fast);
}

llama_dna::~llama_dna() {
    if (params.verbose) {
        print_stats();
    }
}

void llama_dna_init_layer_traits(llama_dna * self, llama_dna_layer & L) {
    const ggml_type_traits_cpu * tr_pred = self->get_traits(self->params.pred_type);
    L.pred_vec_dot    = tr_pred->vec_dot;
    L.pred_vdt        = tr_pred->vec_dot_type;
    L.pred_from_float = self->get_traits(L.pred_vdt)->from_float;
    L.pred_row_size   = ggml_row_size(self->params.pred_type, L.n_embd);

    const int R = LLAMA_DNA_RESERVOIR;
    L.res       .assign((size_t) L.n_ff*R, 0.0f);
    L.res_est   .assign((size_t) L.n_ff*R, 0.0f);
    L.sum_g     .assign(L.n_ff, 0.0);
    L.sum_g2    .assign(L.n_ff, 0.0);
    L.sum_ge    .assign(L.n_ff, 0.0);
    L.sum_ge2   .assign(L.n_ff, 0.0);
    L.sum_gge   .assign(L.n_ff, 0.0);
    L.sum_abs_ge.assign(L.n_ff, 0.0);
}

llama_dna_layer * llama_dna::get_layer(int il,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * gate_b, const ggml_tensor * up_b, const ggml_tensor * down_b,
        const ggml_tensor * gate_s, const ggml_tensor * up_s, const ggml_tensor * down_s,
        enum llama_dna_act act) {
    auto it = layers.find(gate);
    if (it != layers.end()) {
        return it->second.get();
    }

    auto & slot = layers[gate];

    auto reject = [&](const char * why) -> llama_dna_layer * {
        LLAMA_LOG_ERROR("%s: layer %d: DNA disabled: %s\n", __func__, il, why);
        return nullptr;
    };

    if (il < 0 || (size_t) il >= model.layers.size() ||
        model.layers[il].ffn_gate != gate || model.layers[il].ffn_up != up || model.layers[il].ffn_down != down) {
        return reject("not the dense FFN of the layer");
    }
    if (!llama_dna_row_is_readable(gate) || !llama_dna_row_is_readable(up) || !llama_dna_row_is_readable(down)) {
        return reject("weights are not in plain host memory");
    }

    const int64_t n_embd = gate->ne[0];
    const int64_t n_ff   = gate->ne[1];
    if (up->ne[0] != n_embd || up->ne[1] != n_ff || down->ne[0] != n_ff || down->ne[1] != n_embd) {
        return reject("unexpected FFN shapes");
    }
    if (n_embd % 64 != 0) {
        return reject("n_embd is not a multiple of 64");
    }
    {
        if (n_embd % ggml_blck_size(params.pred_type) != 0) {
            return reject("n_embd is not a multiple of the predictor block size");
        }
        const ggml_type_traits_cpu * tr_pred = get_traits(params.pred_type);
        if (!tr_pred->vec_dot) {
            return reject("no CPU vec_dot for the predictor type");
        }
        const ggml_from_float_t ff_pred = get_traits(tr_pred->vec_dot_type)->from_float;
        if (!ff_pred) {
            return reject("no CPU from_float for the predictor activation type");
        }
    }
    // affine terms: bias shapes must match the neuron / output dims, scales must be scalars;
    // all must be readable host tensors (no repacked / device-resident affine tensors)
    std::vector<float> gate_b_v, up_b_v, down_b_v;
    float gate_s_v = 1.0f, up_s_v = 1.0f, down_s_v = 1.0f;
    if (!llama_dna_read_dense(gate_b, n_ff, gate_b_v)) {
        return reject("unreadable or unexpected-shape gate_b");
    }
    if (!llama_dna_read_dense(up_b, n_ff, up_b_v)) {
        return reject("unreadable or unexpected-shape up_b");
    }
    if (!llama_dna_read_dense(down_b, n_embd, down_b_v)) {
        return reject("unreadable or unexpected-shape down_b");
    }
    if (!llama_dna_read_scale(gate_s, gate_s_v)) {
        return reject("unreadable or unexpected-shape gate_s");
    }
    if (!llama_dna_read_scale(up_s, up_s_v)) {
        return reject("unreadable or unexpected-shape up_s");
    }
    if (!llama_dna_read_scale(down_s, down_s_v)) {
        return reject("unreadable or unexpected-shape down_s");
    }

    const ggml_type_traits_cpu * tr_gate = get_traits(gate->type);
    const ggml_type_traits_cpu * tr_up   = get_traits(up->type);
    if (!tr_gate->vec_dot || !tr_up->vec_dot) {
        return reject("no CPU dot product for the gate/up types");
    }
    const ggml_from_float_t ff_gate = get_traits(tr_gate->vec_dot_type)->from_float;
    const ggml_from_float_t ff_up   = get_traits(tr_up->vec_dot_type)->from_float;
    if (!ff_gate || !ff_up) {
        return reject("no CPU quantization for the gate/up activation types");
    }

    slot = std::make_unique<llama_dna_layer>();
    llama_dna_layer & L = *slot;

    L.params   = &params;
    L.il       = il;
    L.act      = act;
    L.n_embd   = n_embd;
    L.n_ff     = n_ff;
    L.gate     = gate;
    L.up       = up;
    L.down     = down;

    L.gate_b = std::move(gate_b_v);
    L.up_b   = std::move(up_b_v);
    L.down_b = std::move(down_b_v);
    L.gate_s = gate_s_v;
    L.up_s   = up_s_v;
    L.down_s = down_s_v;

    L.vec_dot_gate    = tr_gate->vec_dot;
    L.vec_dot_up      = tr_up->vec_dot;
    L.vdt_gate        = tr_gate->vec_dot_type;
    L.vdt_up          = tr_up->vec_dot_type;
    L.from_float_gate = ff_gate;
    L.from_float_up   = ff_up;

    llama_dna_init_layer_traits(this, L);

    const int64_t t0 = ggml_time_us();
    llama_dna_pack_gate(L);
    const int64_t t1 = ggml_time_us();

    LLAMA_LOG_ERROR("%s: layer %3d: packed gate %s [%" PRId64 ", %" PRId64 "] with pred %s into %.1f MiB (%.2f bpw) in %.0f ms\n",
            __func__, il, ggml_type_name(gate->type), n_embd, n_ff,
            ggml_type_name(params.pred_type),
            llama_dna_pred_bytes(L) / (1024.0*1024.0),
            8.0*llama_dna_pred_bytes(L) / ((double) n_embd*n_ff),
            (t1 - t0) / 1000.0);

    return slot.get();
}

llama_dna_layer * llama_dna::get_shared_layer(int il,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * gate_b, const ggml_tensor * up_b, const ggml_tensor * down_b,
        const ggml_tensor * gate_s, const ggml_tensor * up_s, const ggml_tensor * down_s,
        enum llama_dna_act act) {
    // shared expert is the same dense FFN layout; reuse get_layer (its arch check
    // verifies ffn_gate/ffn_up/ffn_down of the layer, which is the shared expert for
    // MoE layers in gemma4/deepseek-style models)
    return get_layer(il, gate, up, down, gate_b, up_b, down_b, gate_s, up_s, down_s, act);
}

uint32_t llama_dna::update() {
    // Epoch must change whenever any layer's phase changes: the reused graph bakes in
    // op_observe's STATS vs BIAS branch and the dense-vs-route topology.
    for (auto & [gate, L] : layers) {
        if (!L) {
            continue;
        }
        // Layers finalize independently (option 1, 2026-11-17): each layer's
        // n_tok counts its own routed tokens, and the stats/bias phases for
        // that layer advance as soon as it reaches n_calib. A global barrier
        // (all layers must reach n_calib in the same phase) stalled perplexity
        // forever: layer 34's 1-token ubatch never accumulates 4096 tokens,
        // so no layer left phase STATS/BIAS and DNA never predicted. Per-layer
        // finalize matches the actual dependency: med_est/band/mu/c0 are
        // per-layer, and the graph reuse key (epoch) only needs to change when
        // *that* layer's topology changes.
        const uint32_t n_tok_prev = L->n_tok;
        L->n_tok = L->n_tok_next;
        L->n_read = L->n_tok_next;
        L->n_slot = L->n_slot + (L->n_tok > n_tok_prev ? L->n_tok - n_tok_prev : 0);

        // Note: perplexity clears the KV cache per chunk (llama_memory_clear),
        // but that does NOT touch DNA state. DNA's phase counters live in
        // llama_dna_layer (n_tok/n_tok_next/n_slot), not in the KV cache, so
        // they persist across chunks. The layer's graph topology for a chunk
        // is rebuilt when epoch changes; the previous chunk's graph is invalid
        // as soon as one layer flips phase. No reset is needed, and no reset
        // should be added - calibration is a property of the trained weights'
        // activation distribution, not of the current chunk's KV state.
        if (L->phase == LLAMA_DNA_PHASE_STATS && L->n_tok >= params.n_calib) {
            llama_dna_finalize_stats(*L);
            L->phase = LLAMA_DNA_PHASE_BIAS;
            L->n_tok = L->n_tok_next = 0;
            L->n_slot = 0;
            epoch++;
        } else if (L->phase == LLAMA_DNA_PHASE_BIAS && L->n_tok >= params.n_calib) {
            llama_dna_finalize_bias(*L);
            L->phase = LLAMA_DNA_PHASE_PREDICT;
            L->n_tok = L->n_tok_next = 0;
            L->n_slot = 0;
            epoch++;
        }
    }
    return epoch;
}

bool llama_dna::layer_is_predict(const llama_dna_layer * layer) {
    return layer && layer->phase == LLAMA_DNA_PHASE_PREDICT;
}



// MoE: one dna_layer per expert. gate_up_exps is [n_embd, 2*n_ff, n_expert] PAR
// (gate rows 0..n_ff-1, up rows n_ff..2*n_ff-1); when null, gate_exps/up_exps are
// the separate [n_embd, n_ff, n_expert] tensors. down_exps is [n_ff, n_embd, n_expert].
// Per-expert layers live in `layers` keyed by the down_exps tensor pointer.
llama_dna_layer * llama_dna::get_moe_expert(int il, int expert,
        const ggml_tensor * gate_up_exps, const ggml_tensor * gate_exps, const ggml_tensor * up_exps,
        const ggml_tensor * down_exps, enum llama_dna_act act) {
    const ggml_tensor * key = down_exps;
    auto it = layers.find(key);
    if (it != layers.end()) {
        return it->second.get();
    }
    auto & slot = layers[key];

    auto reject = [&](const char * why) -> llama_dna_layer * {
        LLAMA_LOG_ERROR("%s: layer %d expert %d: DNA disabled: %s\n", __func__, il, expert, why);
        return nullptr;
    };

    const ggml_tensor * gate_w = gate_up_exps ? gate_up_exps : gate_exps;
    const ggml_tensor * up_w   = gate_up_exps ? gate_up_exps : up_exps;
    if (!gate_w || !up_w || !down_exps) {
        return reject("missing expert weights");
    }
    if (!llama_dna_row_is_readable(gate_w) || !llama_dna_row_is_readable(up_w) || !llama_dna_row_is_readable(down_exps)) {
        return reject("weights are not in plain host memory");
    }
    const int64_t n_embd = gate_w->ne[0];
    const int64_t n_ff2  = gate_w->ne[1];
    const int64_t n_expert = gate_w->ne[2];
    if (expert < 0 || expert >= n_expert) {
        return reject("expert out of range");
    }
    const int64_t n_ff = gate_up_exps ? n_ff2 / 2 : n_ff2;
    if (down_exps->ne[0] != n_ff || down_exps->ne[1] != n_embd || down_exps->ne[2] != n_expert) {
        return reject("unexpected expert down shape");
    }
    if (n_embd % 64 != 0) {
        return reject("n_embd is not a multiple of 64");
    }
    if (n_embd % ggml_blck_size(params.pred_type) != 0) {
        return reject("n_embd is not a multiple of the predictor block size");
    }
    const ggml_type_traits_cpu * tr_pred = get_traits(params.pred_type);
    if (!tr_pred->vec_dot) {
        return reject("no CPU vec_dot for the predictor type");
    }
    if (!get_traits(tr_pred->vec_dot_type)->from_float) {
        return reject("no CPU from_float for the predictor activation type");
    }
    const ggml_type_traits_cpu * tr_gate = get_traits(gate_w->type);
    const ggml_type_traits_cpu * tr_up   = get_traits(up_w->type);
    if (!tr_gate->vec_dot || !tr_up->vec_dot) {
        return reject("no CPU dot product for the expert gate/up types");
    }
    if (!get_traits(tr_gate->vec_dot_type)->from_float || !get_traits(tr_up->vec_dot_type)->from_float) {
        return reject("no CPU quantization for the expert gate/up activation types");
    }

    slot = std::make_unique<llama_dna_layer>();
    llama_dna_layer & L = *slot;

    L.params   = &params;
    L.il       = il;
    L.act      = act;
    L.n_embd   = n_embd;
    L.n_ff     = n_ff;
    L.is_moe   = true;
    L.expert   = expert;
    L.gate_up_exps = gate_up_exps;
    // for packing/layout; rows are expert slices (see pack_gate in this function)
    L.gate     = gate_w;
    L.up       = up_w;
    L.down     = down_exps;

    L.vec_dot_gate    = tr_gate->vec_dot;
    L.vec_dot_up      = tr_up->vec_dot;
    L.vdt_gate        = tr_gate->vec_dot_type;
    L.vdt_up          = tr_up->vec_dot_type;
    L.from_float_gate = get_traits(tr_gate->vec_dot_type)->from_float;
    L.from_float_up   = get_traits(tr_up->vec_dot_type)->from_float;

    llama_dna_init_layer_traits(this, L);

    {
        std::vector<float> wf(L.n_embd);
        const ggml_type_traits * tr = ggml_get_type_traits(L.params->pred_type);
        L.pred_q.assign((size_t) L.n_ff*L.pred_row_size, 0);
        for (int64_t i = 0; i < L.n_ff; ++i) {
            const ggml_tensor * src = gate_up_exps ? gate_up_exps : gate_exps;
            const char * row = (const char *) src->data + expert*src->nb[2] + i*src->nb[1];
            const ggml_type_traits * wtr = ggml_get_type_traits(src->type);
            wtr->to_float(row, wf.data(), L.n_embd);
            tr->from_float_ref(wf.data(), L.pred_q.data() + i*L.pred_row_size, L.n_embd);
        }
    }

    LLAMA_LOG_ERROR("%s: layer %3d expert %d: packed gate %s [%" PRId64 ", %" PRId64 "] with pred %s into %.1f MiB (%.2f bpw)\n",
            __func__, il, expert, ggml_type_name(gate_w->type), n_embd, n_ff,
            ggml_type_name(params.pred_type),
            llama_dna_pred_bytes(L) / (1024.0*1024.0),
            8.0*llama_dna_pred_bytes(L) / ((double) n_embd*n_ff));

    return slot.get();
}

llama_dna_moe_layer * llama_dna::get_moe_layer(int il,
        const ggml_tensor * gate_up_exps, const ggml_tensor * gate_exps, const ggml_tensor * up_exps,
        const ggml_tensor * down_exps, enum llama_dna_act act) {
    const ggml_tensor * key = down_exps ? down_exps : gate_up_exps;
    auto it = moe_layers.find(key);
    if (it != moe_layers.end()) {
        return it->second.get();
    }
    auto & slot = moe_layers[key];
    slot = std::make_unique<llama_dna_moe_layer>();
    llama_dna_moe_layer & M = *slot;
    M.il = il;
    const ggml_tensor * ref = gate_up_exps ? gate_up_exps : (gate_exps ? gate_exps : up_exps);
    M.n_expert = ref ? (int) ref->ne[2] : 0;
    M.experts.assign(M.n_expert, nullptr);
    for (int e = 0; e < M.n_expert; ++e) {
        M.experts[e] = get_moe_expert(il, e, gate_up_exps, gate_exps, up_exps, down_exps, act);
    }
    return slot.get();
}

void llama_dna::print_stats() const {
    for (const auto & [gate, L] : layers) {
        if (!L) {
            continue;
        }
        const uint64_t n_tok = L->pred_tok.load();
        if (n_tok == 0) {
            LLAMA_LOG_ERROR("%s: layer %3d: phase %d, %u calibration tokens\n", __func__, L->il, (int) L->phase, L->n_tok);
            continue;
        }
        LLAMA_LOG_ERROR("%s: layer %3d: %" PRIu64 " tokens predicted, active = %.1f%%, per token: observe %.3f ms, route %.3f ms, down %.3f ms\n",
                __func__, L->il, n_tok, 100.0 * (double) L->pred_act.load() / ((double) n_tok * L->n_ff),
                L->t_observe_us / (1000.0 * 2 * L->params->n_calib), L->t_route_us / (1000.0 * n_tok), L->t_down_us / (1000.0 * n_tok));
    }
}

llama_dna_stats llama_dna::get_stats() const {
    llama_dna_stats s;
    s.enabled = true;
    s.calib_total = 2 * params.n_calib;

    if (layers.empty()) {
        return s;
    }

    for (const auto & [gate, L] : layers) {
        if (!L) {
            s.n_layers_off++;
            continue;
        }
        s.n_layers_on++;
        switch (L->phase) {
            case LLAMA_DNA_PHASE_STATS:
                s.n_phase_stats++;
                s.calib_done = std::min(s.calib_done, L->n_tok);
                break;
            case LLAMA_DNA_PHASE_BIAS:
                s.n_phase_bias++;
                s.calib_done = std::min(s.calib_done, params.n_calib + L->n_tok);
                break;
            case LLAMA_DNA_PHASE_PREDICT:
                s.n_phase_predict++;
                break;
        }
        const uint64_t n_tok = L->pred_tok.load();
        if (n_tok > 0) {
            const double act = 100.0 * (double) L->pred_act.load() / ((double) n_tok * L->n_ff);
            if (s.pred_tokens == 0) {
                s.act_min = s.act_max = act;
                s.pred_tokens = n_tok;
            } else {
                s.act_min = std::min(s.act_min, act);
                s.act_max = std::max(s.act_max, act);
                s.pred_tokens = std::max(s.pred_tokens, n_tok);
            }
        }
    }

    if (s.n_phase_predict > 0) {
        // calibrate done, even if no layer has predicted yet this turn
        s.calib_done = s.calib_total;
    } else {
        // still calibrating: slowest layer drives the reported progress
        uint32_t done = s.calib_total;
        for (const auto & [gate, L] : layers) {
            if (!L) continue;
            uint32_t d = L->phase == LLAMA_DNA_PHASE_STATS ? L->n_tok : params.n_calib + L->n_tok;
            done = std::min(done, d);
        }
        s.calib_done = done;
    }

    {
        double sum = 0.0;
        int n = 0;
        for (const auto & [gate, L] : layers) {
            if (!L || L->pred_tok.load() == 0) continue;
            const double act = 100.0 * (double) L->pred_act.load() / ((double) L->pred_tok.load() * L->n_ff);
            sum += act;
            n++;
            s.pred_tokens = std::max(s.pred_tokens, (uint64_t) L->pred_tok.load());
        }
        s.act_avg = n > 0 ? sum / n : 0.0;
    }

    return s;
}

bool llama_dna::moe_is_predict(const llama_dna_moe_layer * moe) {
    if (!moe || moe->experts.empty()) {
        return false;
    }
    for (llama_dna_layer * L : moe->experts) {
        if (L && L->phase != LLAMA_DNA_PHASE_PREDICT) {
            return false;
        }
    }
    return true;
}

void llama_dna::print_turn_stats() const {
    llama_dna_stats s = get_stats();
    // ALWAYS visible at ERROR level so the CLI (verbosity default ERROR) actually prints it.
    // Priority: if DNA is enabled, this line must surface, even without -lv 3.
    // LLAMA_LOG_ERROR is LOG_LEVEL_ERROR (1) <= threshold (1 by default).
    if (s.n_phase_predict > 0) {
        LLAMA_LOG_ERROR("DNA on (%d/%d layers) — prediction: %" PRIu64 " tokens, active min %.1f%% avg %.1f%% max %.1f%%\n",
                s.n_layers_on, s.n_layers_on + s.n_layers_off, s.pred_tokens, s.act_min, s.act_avg, s.act_max);
    } else {
        LLAMA_LOG_ERROR("DNA on (%d/%d layers) — calibration %u/%u (stats %d, bias %d)\n",
                s.n_layers_on, s.n_layers_on + s.n_layers_off, s.calib_done, s.calib_total, s.n_phase_stats, s.n_phase_bias);
    }
}

//
// custom ops
//

void llama_dna::op_observe(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_layer & L = *(llama_dna_layer *) userdata;

    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * go = dst->src[1];
    const ggml_tensor * uo = dst->src[2];

    const int64_t n_tokens = x->ne[1];
    const int64_t i0 = L.n_ff*ith/nth;
    const int64_t i1 = L.n_ff*(ith + 1)/nth;

    const int R = LLAMA_DNA_RESERVOIR;

    const int64_t t_start = ggml_time_us();

    thread_local std::vector<uint8_t> xq_pred;
    xq_pred.resize(ggml_row_size(L.pred_vdt, L.n_embd));

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * xr = (const float *) ((const char *) x->data  + t*x->nb[1]);
        const float * gr = (const float *) ((const char *) go->data + t*go->nb[1]);
        const float * ur = (const float *) ((const char *) uo->data + t*uo->nb[1]);

        L.pred_from_float(xr, xq_pred.data(), L.n_embd);

        const uint32_t k = L.n_slot + (uint32_t) t; // global reservoir index in this phase; n_tok_next accumulates across ubatches

        for (int64_t i = i0; i < i1; ++i) {
            const float ge = llama_dna_estimate_quant(L, i, xq_pred.data());
            const float g  = gr[i];

            if (L.phase == LLAMA_DNA_PHASE_STATS) {
                llama_dna_reservoir_push(L.res.data()     + i*R, (uint32_t) (2*i),     k, g);
                llama_dna_reservoir_push(L.res_est.data() + i*R, (uint32_t) (2*i + 1), k, ge);
                L.sum_g[i]      += g;
                L.sum_g2[i]     += (double) g*g;
                L.sum_ge[i]     += ge;
                L.sum_ge2[i]    += (double) ge*ge;
                L.sum_gge[i]    += (double) g*ge;
                L.sum_abs_ge[i] += fabsf(ge);
            } else {
                const float h   = llama_dna_act(L.act, g) * ur[i];
                const float dev = fabsf(ge - L.med_est[i]);
                L.sum_h2[i] += (double) h*h;
                if (dev < L.band[i]) {
                    L.mu_sum[i] += h;
                    L.mu_cnt[i]++;
                } else {
                    L.sum_h2_act[i] += (double) h*h;
                    L.n_act[i]++;
                }
                llama_dna_reservoir_push(L.res.data() + i*R, (uint32_t) (2*i), k, L.band[i] > 0.0f ? dev / L.band[i] : 0.0f);
            }
        }
    }

    if (ith == 0) {
        L.n_tok_next += (uint32_t) n_tokens; // accumulate; update() syncs n_tok = n_tok_next between graph builds
        L.t_observe_us += ggml_time_us() - t_start;
    }
}

void llama_dna::op_route(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_layer & L = *(llama_dna_layer *) userdata;

    const ggml_tensor * x = dst->src[0];

    const int64_t n_tokens = x->ne[1];
    const int64_t i0 = L.n_ff*ith/nth;
    const int64_t i1 = L.n_ff*(ith + 1)/nth;

    const int64_t t_start = ggml_time_us();

    const size_t sz_gate = ggml_row_size(L.vdt_gate, L.n_embd);
    const size_t sz_up   = ggml_row_size(L.vdt_up,   L.n_embd);

    thread_local std::vector<uint8_t> xq_gate, xq_up, xq_pred;
    xq_gate.resize(sz_gate);
    xq_up.resize(sz_up);
    xq_pred.resize(ggml_row_size(L.pred_vdt, L.n_embd));

    const bool same_vdt = L.vdt_gate == L.vdt_up;

    uint64_t n_act = 0;

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * xr  = (const float *) ((const char *) x->data + t*x->nb[1]);
        float       * out = (float *) ((char *) dst->data + t*dst->nb[1]);

        L.pred_from_float(xr, xq_pred.data(), L.n_embd);
        L.from_float_gate(xr, xq_gate.data(), L.n_embd);
        if (!same_vdt) {
            L.from_float_up(xr, xq_up.data(), L.n_embd);
        }
        const void * xq_u = same_vdt ? xq_gate.data() : xq_up.data();

        for (int64_t i = i0; i < i1; ++i) {
            const float ge = llama_dna_estimate_quant(L, i, xq_pred.data());
            if (fabsf(ge - L.med_est[i]) < L.band[i]) {
                out[i] = 0.0f;
                continue;
            }
            float g, u;
            L.vec_dot_gate((int) L.n_embd, &g, 0, (const char *) L.gate->data + i*L.gate->nb[1], 0, xq_gate.data(), 0, 1);
            L.vec_dot_up  ((int) L.n_embd, &u, 0, (const char *) L.up->data   + i*L.up->nb[1],   0, xq_u,           0, 1);
            // dense order: (W x + b) * s, then silu(g) * u
            if (!L.gate_b.empty()) {
                g += L.gate_b[(size_t) i];
            }
            g *= L.gate_s;
            if (!L.up_b.empty()) {
                u += L.up_b[(size_t) i];
            }
            u *= L.up_s;
            out[i] = llama_dna_act(L.act, g) * u - L.mu[i];
            n_act++;
        }
    }

    L.pred_act += n_act;
    if (ith == 0) {
        L.pred_tok += n_tokens;
        L.t_route_us += ggml_time_us() - t_start;
    }
}

void llama_dna::op_down(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_layer & L = *(llama_dna_layer *) userdata;

    const ggml_tensor * delta = dst->src[0];

    const int64_t n_tokens = delta->ne[1];

    const int64_t t_start = ggml_time_us();

    const size_t n_tiles =
        ((size_t) L.n_embd + DNA_IQ3_CXS2_WEIGHTS_PER_LINE - 1) /
        DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t tiles_per_thread = (n_tiles + nth - 1) / nth;
    const size_t tile_0 = std::min((size_t) ith * tiles_per_thread, n_tiles);
    const size_t tile_1 = std::min(tile_0 + tiles_per_thread, n_tiles);
    if (tile_0 >= tile_1) {
        return;
    }

    thread_local std::vector<dna_iq3_cxs2_active_neuron> active_neurons;
    alignas(64) float tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * d = (const float *) ((const char *) delta->data + t*delta->nb[1]);
        float       * y = (float *) ((char *) dst->data + t*dst->nb[1]);

        active_neurons.clear();
        for (int64_t i = 0; i < L.n_ff; ++i) {
            if (d[i] != 0.0f) {
                active_neurons.push_back({
                    (int32_t) i,
                    d[i],
                });
            }
        }

        for (size_t tile = tile_0; tile < tile_1; ++tile) {
            const size_t j_offset =
                tile * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
            const size_t j_count = std::min(
                (size_t) DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
                (size_t) L.n_embd - j_offset);
            const size_t n_subblocks =
                (j_count + DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK - 1) /
                DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;

            memcpy(tile_accum, L.c0.data() + j_offset, j_count * sizeof(float));
            std::fill(tile_accum + j_count,
                      tile_accum + DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
                      0.0f);

#if defined(LLAMA_DNA_X86)
            if (g_x86_fast) {
                dna_iq3_cxs2_down_tile_accum_avx2(
                    tile_accum,
                    active_neurons.data(),
                    active_neurons.size(),
                    tile,
                    L.down_blocks_per_neuron,
                    L.down_blocks.data(),
                    L.down_scales.data(),
                    n_subblocks);
            } else
#endif
            {
                dna_iq3_cxs2_down_tile_accum(
                    tile_accum,
                    active_neurons.data(),
                    active_neurons.size(),
                    tile,
                    L.down_blocks_per_neuron,
                    L.down_blocks.data(),
                    L.down_scales.data(),
                    n_subblocks);
            }

            memcpy(y + j_offset, tile_accum, j_count * sizeof(float));
        }
    }

    if (ith == 0) {
        L.t_down_us += ggml_time_us() - t_start;
    }
}

// op_expert_observe: userdata is llama_dna_moe_layer*, src0 = x, src1 = gate out,
// src2 = up out, src3 = selected_experts
void llama_dna::op_expert_observe(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_moe_layer & M = *(llama_dna_moe_layer *) userdata;
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * go = dst->src[1];
    const ggml_tensor * uo = dst->src[2];
    const ggml_tensor * sel = dst->src[3];

    const int64_t n_tokens = x->ne[1];
    const int64_t n_slots = sel->ne[0];
    const int64_t n_ff = go->ne[0];
    const int R = LLAMA_DNA_RESERVOIR;

    const int64_t s0 = n_slots*ith/nth;
    const int64_t s1 = n_slots*(ith + 1)/nth;

    for (int64_t s = s0; s < s1; ++s) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int32_t e = ((const int32_t *) sel->data)[t*sel->nb[1]/sizeof(int32_t) + s];
            llama_dna_layer * L = M.experts[e];
            if (!L) {
                continue;
            }
            const float * xr = (const float *) ((const char *) x->data + t*x->nb[1]);
            const float * gr = (const float *) ((const char *) go->data + s*go->nb[1] + t*go->nb[2]);
            const float * ur = (const float *) ((const char *) uo->data + s*uo->nb[1] + t*uo->nb[2]);
            thread_local std::vector<uint8_t> xq_pred;
            xq_pred.assign(ggml_row_size(L->pred_vdt, L->n_embd), 0);
            L->pred_from_float(xr, xq_pred.data(), L->n_embd);
            const uint32_t k = L->n_slot + (uint32_t) t;
            for (int64_t i = 0; i < n_ff; ++i) {
                const float ge = llama_dna_estimate_quant(*L, i, xq_pred.data());
                const float g  = gr[i];
                if (L->phase == LLAMA_DNA_PHASE_STATS) {
                    llama_dna_reservoir_push(L->res.data()     + i*R, (uint32_t) (2*i),     k, g);
                    llama_dna_reservoir_push(L->res_est.data() + i*R, (uint32_t) (2*i + 1), k, ge);
                    L->sum_g[i]      += g;
                    L->sum_g2[i]     += (double) g*g;
                    L->sum_ge[i]     += ge;
                    L->sum_ge2[i]    += (double) ge*ge;
                    L->sum_gge[i]    += (double) g*ge;
                    L->sum_abs_ge[i] += fabsf(ge);
                } else {
                    const float h   = llama_dna_act(L->act, g) * ur[i];
                    const float dev = fabsf(ge - L->med_est[i]);
                    L->sum_h2[i] += (double) h*h;
                    if (dev < L->band[i]) {
                        L->mu_sum[i] += h;
                        L->mu_cnt[i]++;
                    } else {
                        L->sum_h2_act[i] += (double) h*h;
                        L->n_act[i]++;
                    }
                    llama_dna_reservoir_push(L->res.data() + i*R, (uint32_t) (2*i), k, L->band[i] > 0.0f ? dev / L->band[i] : 0.0f);
                }
            }
        }
    }

    if (ith == 0) {
        for (llama_dna_layer * L : M.experts) {
            if (L) {
                L->n_tok_next += (uint32_t) n_tokens;
            }
        }
    }
}

// MoE ops: userdata is llama_dna_moe_layer*
// op_expert_route: src0 = x, src1 = gate_up_exps, src2 = selected_experts
//   -> delta [n_ff, n_expert_used, n_tokens]
void llama_dna::op_expert_route(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_moe_layer & M = *(llama_dna_moe_layer *) userdata;
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * gate_up = dst->src[1];
    const ggml_tensor * sel = dst->src[2];

    const int64_t n_tokens = x->ne[1];
    const int64_t n_slots  = sel->ne[0];
    const int64_t n_ff     = dst->ne[0];

    const int64_t s0 = n_slots*ith/nth;
    const int64_t s1 = n_slots*(ith + 1)/nth;

    for (int64_t s = s0; s < s1; ++s) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int32_t e = ((const int32_t *) sel->data)[t*sel->nb[1]/sizeof(int32_t) + s];
            llama_dna_layer * L = M.experts[e];
            float * out = (float *) ((char *) dst->data + s*dst->nb[1] + t*dst->nb[2]);
            if (!L || L->phase != LLAMA_DNA_PHASE_PREDICT) {
                memset(out, 0, n_ff*sizeof(float));
                continue;
            }
            const float * xr = (const float *) ((const char *) x->data + t*x->nb[1]);
            thread_local std::vector<uint8_t> xq_gate, xq_up, xq_pred;
            xq_gate.assign(ggml_row_size(L->vdt_gate, L->n_embd), 0);
            xq_up.assign(ggml_row_size(L->vdt_up, L->n_embd), 0);
            xq_pred.assign(ggml_row_size(L->pred_vdt, L->n_embd), 0);
            L->pred_from_float(xr, xq_pred.data(), L->n_embd);
            L->from_float_gate(xr, xq_gate.data(), L->n_embd);
            if (L->vdt_gate != L->vdt_up) {
                L->from_float_up(xr, xq_up.data(), L->n_embd);
            }
            const void * xq_u = L->vdt_gate == L->vdt_up ? xq_gate.data() : xq_up.data();
            for (int64_t i = 0; i < n_ff; ++i) {
                const float ge = llama_dna_estimate_quant(*L, i, xq_pred.data());
                if (fabsf(ge - L->med_est[i]) < L->band[i]) {
                    out[i] = 0.0f;
                    continue;
                }
                float g, u;
                const char * grow = (const char *) gate_up->data + e*gate_up->nb[2] + i*gate_up->nb[1];
                const char * urow = (const char *) gate_up->data + e*gate_up->nb[2] + (i + n_ff)*gate_up->nb[1];
                L->vec_dot_gate((int) L->n_embd, &g, 0, grow, 0, xq_gate.data(), 0, 1);
                L->vec_dot_up  ((int) L->n_embd, &u, 0, urow, 0, xq_u,           0, 1);
                out[i] = llama_dna_act(L->act, g) * u - L->mu[i];
            }
        }
    }
}

// op_expert_down: src0 = delta, src1 = weights [1, n_expert_used, n_tokens],
// src2 = selected_experts -> y [n_embd, n_tokens]
void llama_dna::op_expert_down(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_moe_layer & M = *(llama_dna_moe_layer *) userdata;
    const ggml_tensor * delta = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * sel = dst->src[2];

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_slots = delta->ne[1];
    const int64_t n_ff = delta->ne[0];

    const size_t n_tiles =
        ((size_t) n_embd + DNA_IQ3_CXS2_WEIGHTS_PER_LINE - 1) /
        DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t tiles_per_thread = (n_tiles + nth - 1) / nth;
    const size_t tile_0 = std::min((size_t) ith * tiles_per_thread, n_tiles);
    const size_t tile_1 = std::min(tile_0 + tiles_per_thread, n_tiles);
    if (tile_0 >= tile_1) {
        return;
    }

    thread_local std::vector<std::vector<dna_iq3_cxs2_active_neuron>> active_by_slot;
    active_by_slot.resize((size_t) n_slots);
    alignas(64) float tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];

    for (int64_t t = 0; t < n_tokens; ++t) {
        float * y = (float *) ((char *) dst->data + t*dst->nb[1]);

        for (int64_t s = 0; s < n_slots; ++s) {
            std::vector<dna_iq3_cxs2_active_neuron> & active = active_by_slot[s];
            active.clear();

            const int32_t e =
                ((const int32_t *) sel->data)[t*sel->nb[1]/sizeof(int32_t) + s];
            llama_dna_layer * L = M.experts[e];
            if (!L) {
                continue;
            }

            const float weight = ((const float *) weights->data)[
                t*weights->nb[1]/sizeof(float) + s*weights->nb[0]/sizeof(float)];
            const float * d = (const float *) (
                (const char *) delta->data + s*delta->nb[1] + t*delta->nb[2]);
            for (int64_t i = 0; i < n_ff; ++i) {
                if (d[i] != 0.0f) {
                    active.push_back({
                        (int32_t) i,
                        weight * d[i],
                    });
                }
            }
        }

        for (size_t tile = tile_0; tile < tile_1; ++tile) {
            const size_t j_offset =
                tile * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
            const size_t j_count = std::min(
                (size_t) DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
                (size_t) n_embd - j_offset);
            const size_t n_subblocks =
                (j_count + DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK - 1) /
                DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
            std::fill(tile_accum,
                      tile_accum + DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
                      0.0f);

            for (int64_t s = 0; s < n_slots; ++s) {
                const int32_t e =
                    ((const int32_t *) sel->data)[t*sel->nb[1]/sizeof(int32_t) + s];
                llama_dna_layer * L = M.experts[e];
                if (!L) {
                    continue;
                }

                const float weight = ((const float *) weights->data)[
                    t*weights->nb[1]/sizeof(float) + s*weights->nb[0]/sizeof(float)];
                for (size_t j = 0; j < j_count; ++j) {
                    tile_accum[j] += weight * L->c0[j_offset + j];
                }

#if defined(LLAMA_DNA_X86)
                if (g_x86_fast) {
                    dna_iq3_cxs2_down_tile_accum_avx2(
                        tile_accum,
                        active_by_slot[s].data(),
                        active_by_slot[s].size(),
                        tile,
                        L->down_blocks_per_neuron,
                        L->down_blocks.data(),
                        L->down_scales.data(),
                        n_subblocks);
                } else
#endif
                {
                    dna_iq3_cxs2_down_tile_accum(
                        tile_accum,
                        active_by_slot[s].data(),
                        active_by_slot[s].size(),
                        tile,
                        L->down_blocks_per_neuron,
                        L->down_blocks.data(),
                        L->down_scales.data(),
                        n_subblocks);
                }
            }

            memcpy(y + j_offset, tile_accum, j_count * sizeof(float));
        }
    }
}
