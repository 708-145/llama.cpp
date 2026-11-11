#include "llama-dna.h"

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

enum llama_dna_phase {
    LLAMA_DNA_PHASE_STATS,   // per-neuron gate statistics and estimator bias
    LLAMA_DNA_PHASE_BIAS,    // silent-neuron bias mu with the final dead zone rule
    LLAMA_DNA_PHASE_PREDICT,
};

static constexpr int LLAMA_DNA_RESERVOIR = 256;

struct llama_dna_layer {
    const llama_dna_params * params = nullptr;

    int il = -1;

    int64_t n_embd = 0;
    int64_t n_ff   = 0;

    int n_words  = 0; // uint32 sign words per gate row
    int n_blocks = 0; // scale blocks per gate row

    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up   = nullptr;
    const ggml_tensor * down = nullptr;

    ggml_vec_dot_t   vec_dot_gate    = nullptr;
    ggml_vec_dot_t   vec_dot_up      = nullptr;
    ggml_from_float_t from_float_gate = nullptr;
    ggml_from_float_t from_float_up   = nullptr;
    enum ggml_type   vdt_gate = GGML_TYPE_F32;
    enum ggml_type   vdt_up   = GGML_TYPE_F32;

    llama_dna_phase phase = LLAMA_DNA_PHASE_STATS;

    uint32_t n_tok      = 0; // tokens seen in the current phase
    uint32_t n_tok_next = 0; // written by op_observe, applied by llama_dna::update()

    // packed gate estimator
    std::vector<uint32_t>    sw; // [n_ff][n_words]  sign bits, bit = (w >= 0)
    std::vector<ggml_fp16_t> aw; // [n_ff][n_blocks] mean |w| per block

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
    std::vector<ggml_fp16_t> down_t;  // [n_ff][n_embd] transposed down projection

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

static inline int llama_dna_popcount(uint32_t x) {
#if defined(_MSC_VER)
    return (int) __popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

static inline float llama_dna_silu(float x) {
    return x / (1.0f + expf(-x));
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

// y += d * w
static void llama_dna_axpy_f16_generic(float * y, const ggml_fp16_t * w, float d, int64_t n) {
    float tmp[64];
    for (int64_t j = 0; j < n; j += 64) {
        const int m = (int) std::min<int64_t>(64, n - j);
        llama_dna_f16_to_f32_generic(w + j, tmp, m);
        for (int k = 0; k < m; ++k) {
            y[j + k] += d * tmp[k];
        }
    }
}

#if defined(LLAMA_DNA_X86)
__attribute__((target("avx2,fma,f16c")))
static void llama_dna_axpy_f16_x86(float * y, const ggml_fp16_t * w, float d, int64_t n) {
    const __m256 vd = _mm256_set1_ps(d);
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        const __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + j)));
        _mm256_storeu_ps(y + j, _mm256_fmadd_ps(vd, vw, _mm256_loadu_ps(y + j)));
    }
    llama_dna_axpy_f16_generic(y + j, w + j, d, n - j);
}
#endif

static inline void llama_dna_axpy_f16(float * y, const ggml_fp16_t * w, float d, int64_t n) {
#if defined(LLAMA_DNA_X86)
    if (g_x86_fast) {
        llama_dna_axpy_f16_x86(y, w, d, n);
        return;
    }
#endif
    llama_dna_axpy_f16_generic(y, w, d, n);
}

// sign bits and per-block mean |x| of a float row
static void llama_dna_pack_row(const float * x, int64_t n, int B, uint32_t * sx, float * ax) {
    const int n_words = (int) (n / 32);
    for (int w = 0; w < n_words; ++w) {
        uint32_t bits = 0;
        for (int b = 0; b < 32; ++b) {
            bits |= (uint32_t) (x[w*32 + b] >= 0.0f) << b;
        }
        sx[w] = bits;
    }
    const int n_blocks = (int) (n / B);
    for (int b = 0; b < n_blocks; ++b) {
        float s = 0.0f;
        for (int k = 0; k < B; ++k) {
            s += fabsf(x[b*B + k]);
        }
        ax[b] = s / B;
    }
}

// g~ = sum_b aw[b] * ax[b] * (B - 2 * popcount(sw[b] ^ sx[b]))
static inline float llama_dna_estimate_impl(const uint32_t * sw, const float * awf, const uint32_t * sx, const float * ax, int n_words, int B) {
    float acc = 0.0f;
    if (B >= 32) {
        const int wpb = B / 32;
        const int n_blocks = n_words / wpb;
        for (int b = 0; b < n_blocks; ++b) {
            int pc = 0;
            for (int k = 0; k < wpb; ++k) {
                pc += llama_dna_popcount(sw[b*wpb + k] ^ sx[b*wpb + k]);
            }
            acc += awf[b] * ax[b] * (float) (B - 2*pc);
        }
    } else {
        const int bpw = 32 / B;
        const uint32_t mask = (1u << B) - 1;
        for (int w = 0; w < n_words; ++w) {
            const uint32_t x = sw[w] ^ sx[w];
            for (int k = 0; k < bpw; ++k) {
                const int pc = llama_dna_popcount((x >> (k*B)) & mask);
                const int b = w*bpw + k;
                acc += awf[b] * ax[b] * (float) (B - 2*pc);
            }
        }
    }
    return acc;
}

#if defined(LLAMA_DNA_X86)
__attribute__((target("avx2,fma,f16c,popcnt")))
static float llama_dna_estimate_x86(const uint32_t * sw, const float * awf, const uint32_t * sx, const float * ax, int n_words, int B) {
    return llama_dna_estimate_impl(sw, awf, sx, ax, n_words, B);
}
#endif

static inline float llama_dna_estimate(const llama_dna_layer & L, int64_t i, const uint32_t * sx, const float * ax, float * awf) {
    const uint32_t    * sw = L.sw.data() + i*L.n_words;
    const ggml_fp16_t * aw = L.aw.data() + i*L.n_blocks;
    const int B = L.params->block;
#if defined(LLAMA_DNA_X86)
    if (g_x86_fast) {
        llama_dna_f16_to_f32_x86(aw, awf, L.n_blocks);
        return llama_dna_estimate_x86(sw, awf, sx, ax, L.n_words, B);
    }
#endif
    llama_dna_f16_to_f32_generic(aw, awf, L.n_blocks);
    return llama_dna_estimate_impl(sw, awf, sx, ax, L.n_words, B);
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
    const int B = L.params->block;

    L.sw.resize((size_t) L.n_ff*L.n_words);
    L.aw.resize((size_t) L.n_ff*L.n_blocks);

    std::vector<float> wf(L.n_embd);
    std::vector<float> awf(L.n_blocks);

    for (int64_t i = 0; i < L.n_ff; ++i) {
        llama_dna_row_to_float(L.gate, i, wf.data());
        llama_dna_pack_row(wf.data(), L.n_embd, B, L.sw.data() + i*L.n_words, awf.data());
        ggml_fp32_to_fp16_row(awf.data(), L.aw.data() + i*L.n_blocks, L.n_blocks);
    }
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

    LLAMA_LOG_INFO("%s: layer %3d: stats over %u tokens: |med|/sd = %.2f, sd = %.3f, sd_est = %.3f, mean |g~| = %.3f, corr(g, g~) = %.3f\n",
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

    // c0 = Wdown @ mu, and the transposed f16 copy of Wdown for the sparse delta projection
    L.c0.assign(L.n_embd, 0.0f);
    L.down_t.resize((size_t) L.n_ff * L.n_embd);

    std::vector<float> rowf(L.n_ff);
    std::vector<ggml_fp16_t> rowh(L.n_ff);
    for (int64_t j = 0; j < L.n_embd; ++j) {
        llama_dna_row_to_float(L.down, j, rowf.data());
        ggml_fp32_to_fp16_row(rowf.data(), rowh.data(), L.n_ff);
        double c = 0.0;
        for (int64_t i = 0; i < L.n_ff; ++i) {
            L.down_t[(size_t) i*L.n_embd + j] = rowh[i];
            c += (double) rowf[i] * L.mu[i];
        }
        L.c0[j] = (float) c;
    }

    LLAMA_LOG_INFO("%s: layer %3d: bias over %u tokens: active = %.1f%%, energy(active) = %.1f%%, energy(bias) = %.1f%%, band scale = %.3f, down_t = %.1f MiB\n",
            __func__, L.il, L.n_tok, 100.0*act_frac, 100.0*sum_h2_act / std::max(sum_h2, 1e-30), 100.0*sum_h2_mu / std::max(sum_h2, 1e-30),
            band_scale, L.down_t.size()*sizeof(ggml_fp16_t) / (1024.0*1024.0));

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

llama_dna_params llama_dna_params::from_env() {
    llama_dna_params p;
    p.thresh     = llama_dna_env_float("LLAMA_DNA_T",          p.thresh);
    p.block      = llama_dna_env_int  ("LLAMA_DNA_BLOCK",      p.block);
    p.n_calib    = llama_dna_env_int  ("LLAMA_DNA_CALIB",      p.n_calib);
    p.min_active = llama_dna_env_float("LLAMA_DNA_MIN_ACTIVE", p.min_active);
    p.zmode      = llama_dna_env_int  ("LLAMA_DNA_ZMODE",      p.zmode);
    p.max_batch  = llama_dna_env_int  ("LLAMA_DNA_MAX_BATCH",  p.max_batch);
    p.verbose    = llama_dna_env_int  ("LLAMA_DNA_VERBOSE",    p.verbose);
    return p;
}

//
// llama_dna
//

llama_dna::llama_dna(const llama_model & model, ggml_backend_t backend_cpu, const llama_dna_params & params) : params(params), model(model) {
    if (params.block != 8 && params.block != 16 && params.block != 32 && params.block != 64) {
        throw std::runtime_error("DNA: block size must be 8, 16, 32 or 64");
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
    LLAMA_LOG_INFO("%s: DNA enabled: T = %.2f, block = %d, calib = %u tokens/phase, min_active = %.1f%%, zmode = %d, max_batch = %d, simd = %d\n",
            __func__, params.thresh, params.block, params.n_calib, 100.0f*params.min_active, params.zmode, params.max_batch, fast);
}

llama_dna::~llama_dna() {
    if (params.verbose) {
        print_stats();
    }
}

llama_dna_layer * llama_dna::get_layer(int il, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down) {
    auto it = layers.find(gate);
    if (it != layers.end()) {
        return it->second.get();
    }

    auto & slot = layers[gate];

    auto reject = [&](const char * why) -> llama_dna_layer * {
        LLAMA_LOG_WARN("%s: layer %d: DNA disabled: %s\n", __func__, il, why);
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
    L.n_embd   = n_embd;
    L.n_ff     = n_ff;
    L.n_words  = (int) (n_embd / 32);
    L.n_blocks = (int) (n_embd / params.block);
    L.gate     = gate;
    L.up       = up;
    L.down     = down;

    L.vec_dot_gate    = tr_gate->vec_dot;
    L.vec_dot_up      = tr_up->vec_dot;
    L.vdt_gate        = tr_gate->vec_dot_type;
    L.vdt_up          = tr_up->vec_dot_type;
    L.from_float_gate = ff_gate;
    L.from_float_up   = ff_up;

    const int R = LLAMA_DNA_RESERVOIR;
    L.res       .assign((size_t) n_ff*R, 0.0f);
    L.res_est   .assign((size_t) n_ff*R, 0.0f);
    L.sum_g     .assign(n_ff, 0.0);
    L.sum_g2    .assign(n_ff, 0.0);
    L.sum_ge    .assign(n_ff, 0.0);
    L.sum_ge2   .assign(n_ff, 0.0);
    L.sum_gge   .assign(n_ff, 0.0);
    L.sum_abs_ge.assign(n_ff, 0.0);

    const int64_t t0 = ggml_time_us();
    llama_dna_pack_gate(L);
    const int64_t t1 = ggml_time_us();

    LLAMA_LOG_INFO("%s: layer %3d: packed gate %s [%" PRId64 ", %" PRId64 "] into %.1f MiB (%.2f bpw) in %.0f ms\n",
            __func__, il, ggml_type_name(gate->type), n_embd, n_ff,
            (L.sw.size()*sizeof(uint32_t) + L.aw.size()*sizeof(ggml_fp16_t)) / (1024.0*1024.0),
            8.0*(L.sw.size()*sizeof(uint32_t) + L.aw.size()*sizeof(ggml_fp16_t)) / ((double) n_embd*n_ff),
            (t1 - t0) / 1000.0);

    return slot.get();
}

uint32_t llama_dna::update() {
    for (auto & [gate, L] : layers) {
        if (!L) {
            continue;
        }
        L->n_tok = L->n_tok_next;

        if (L->phase == LLAMA_DNA_PHASE_STATS && L->n_tok >= params.n_calib) {
            llama_dna_finalize_stats(*L);
            L->phase = LLAMA_DNA_PHASE_BIAS;
            L->n_tok = L->n_tok_next = 0;
        } else if (L->phase == LLAMA_DNA_PHASE_BIAS && L->n_tok >= params.n_calib) {
            llama_dna_finalize_bias(*L);
            L->phase = LLAMA_DNA_PHASE_PREDICT;
            L->n_tok = L->n_tok_next = 0;
            epoch++;
        }
    }
    return epoch;
}

bool llama_dna::layer_is_predict(const llama_dna_layer * layer) {
    return layer && layer->phase == LLAMA_DNA_PHASE_PREDICT;
}

void llama_dna::print_stats() const {
    for (const auto & [gate, L] : layers) {
        if (!L) {
            continue;
        }
        const uint64_t n_tok = L->pred_tok.load();
        if (n_tok == 0) {
            LLAMA_LOG_INFO("%s: layer %3d: phase %d, %u calibration tokens\n", __func__, L->il, (int) L->phase, L->n_tok);
            continue;
        }
        LLAMA_LOG_INFO("%s: layer %3d: %" PRIu64 " tokens predicted, active = %.1f%%, per token: observe %.3f ms, route %.3f ms, down %.3f ms\n",
                __func__, L->il, n_tok, 100.0 * (double) L->pred_act.load() / ((double) n_tok * L->n_ff),
                L->t_observe_us / (1000.0 * 2 * L->params->n_calib), L->t_route_us / (1000.0 * n_tok), L->t_down_us / (1000.0 * n_tok));
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

    const int B = L.params->block;
    const int R = LLAMA_DNA_RESERVOIR;

    const int64_t t_start = ggml_time_us();

    thread_local std::vector<uint32_t> sx;
    thread_local std::vector<float> ax, awf;
    sx.resize(L.n_words);
    ax.resize(L.n_blocks);
    awf.resize(L.n_blocks);

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * xr = (const float *) ((const char *) x->data  + t*x->nb[1]);
        const float * gr = (const float *) ((const char *) go->data + t*go->nb[1]);
        const float * ur = (const float *) ((const char *) uo->data + t*uo->nb[1]);

        llama_dna_pack_row(xr, L.n_embd, B, sx.data(), ax.data());

        const uint32_t k = L.n_tok + (uint32_t) t;

        for (int64_t i = i0; i < i1; ++i) {
            const float ge = llama_dna_estimate(L, i, sx.data(), ax.data(), awf.data());
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
                const float h   = llama_dna_silu(g) * ur[i];
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
        L.n_tok_next = L.n_tok + (uint32_t) n_tokens;
        L.t_observe_us += ggml_time_us() - t_start;
    }
}

void llama_dna::op_route(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_layer & L = *(llama_dna_layer *) userdata;

    const ggml_tensor * x = dst->src[0];

    const int64_t n_tokens = x->ne[1];
    const int64_t i0 = L.n_ff*ith/nth;
    const int64_t i1 = L.n_ff*(ith + 1)/nth;

    const int B = L.params->block;

    const int64_t t_start = ggml_time_us();

    const size_t sz_gate = ggml_row_size(L.vdt_gate, L.n_embd);
    const size_t sz_up   = ggml_row_size(L.vdt_up,   L.n_embd);

    thread_local std::vector<uint32_t> sx;
    thread_local std::vector<float> ax, awf;
    thread_local std::vector<uint8_t> xq_gate, xq_up;
    sx.resize(L.n_words);
    ax.resize(L.n_blocks);
    awf.resize(L.n_blocks);
    xq_gate.resize(sz_gate);
    xq_up.resize(sz_up);

    const bool same_vdt = L.vdt_gate == L.vdt_up;

    uint64_t n_act = 0;

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * xr  = (const float *) ((const char *) x->data + t*x->nb[1]);
        float       * out = (float *) ((char *) dst->data + t*dst->nb[1]);

        llama_dna_pack_row(xr, L.n_embd, B, sx.data(), ax.data());
        L.from_float_gate(xr, xq_gate.data(), L.n_embd);
        if (!same_vdt) {
            L.from_float_up(xr, xq_up.data(), L.n_embd);
        }
        const void * xq_u = same_vdt ? xq_gate.data() : xq_up.data();

        for (int64_t i = i0; i < i1; ++i) {
            const float ge = llama_dna_estimate(L, i, sx.data(), ax.data(), awf.data());
            if (fabsf(ge - L.med_est[i]) < L.band[i]) {
                out[i] = 0.0f;
                continue;
            }
            float g, u;
            L.vec_dot_gate((int) L.n_embd, &g, 0, (const char *) L.gate->data + i*L.gate->nb[1], 0, xq_gate.data(), 0, 1);
            L.vec_dot_up  ((int) L.n_embd, &u, 0, (const char *) L.up->data   + i*L.up->nb[1],   0, xq_u,           0, 1);
            out[i] = llama_dna_silu(g) * u - L.mu[i];
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

    // each thread owns a stripe of the output, no reduction is needed
    const int64_t per = ((L.n_embd + nth - 1)/nth + 7) & ~(int64_t) 7;
    const int64_t j0  = std::min<int64_t>(per*ith, L.n_embd);
    const int64_t j1  = std::min<int64_t>(j0 + per, L.n_embd);
    if (j0 >= j1) {
        return;
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * d = (const float *) ((const char *) delta->data + t*delta->nb[1]);
        float       * y = (float *) ((char *) dst->data + t*dst->nb[1]);

        memcpy(y + j0, L.c0.data() + j0, (j1 - j0)*sizeof(float));

        for (int64_t i = 0; i < L.n_ff; ++i) {
            if (d[i] != 0.0f) {
                llama_dna_axpy_f16(y + j0, L.down_t.data() + (size_t) i*L.n_embd + j0, d[i], j1 - j0);
            }
        }
    }

    if (ith == 0) {
        L.t_down_us += ggml_time_us() - t_start;
    }
}
