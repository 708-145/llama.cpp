# DNA - Dynamic Neuron Activation

## Key idea

For each layer's MLP, use an approximated gate projection to decide which neurons are active.  
Compute the active neurons' gate/up/down in full precision, and replace silent neurons with their calibrated bias contribution.  
Compute the down projection only for active neurons (delta form) on the transposed matrix, with blocking and skipping.  
Use CPU-first design and implementation approach. Use CPU flexibility and SIMD parallelism.  

## Core principles

1. **Per-neuron bias.** A neuron's gate pre-activation idles at a neuron-specific median (typically 0.7–2.1 std from 0). down_input median is near 0 because the symmetric up projection hides the offset. Bias holds 31–55% of down_input energy — ignoring it is not an option.

2. **Two-sided dead zone on gate deviation.** Silence is defined by distance from the per-neuron bias, not by raw gate magnitude. A neuron is silent when |z| < threshold where z = (g - median) / std (or a cheap proxy). Both large-positive and large-negative deviations are active. One-sided rules (only z > 0) fail end-to-end.

3. **Selection by distance-from-bias.** Select active neurons by |s - b| (deviation of SiLU(gate) from bias contribution), not by raw |gate| or |gate_approx|. Distance-from-bias selection + bias fill beats magnitude selection + zero fill by 15–22x in KL.

4. **Bias fill (mu[i], c0).** For silent neurons, h[i] = silu(g[i]) * u[i] is replaced by its calibration mean mu[i]. Precompute c0 = Wdown @ mu once per layer. Down projection: y = c0 + sum_active Wdown[:, i] * (h[i] - mu[i]). The delta form is exact: silent neurons contribute mu[i] (folded into c0); active neurons contribute their true h[i].

5. **Gate estimate is a router only.** The approximate gate (XOR sign scheme) is used only to decide silent vs active. It is never reused as the actual gate value — active neurons always get full-precision gate/up.

6. **Online calibration during llama execution, zero interface change.** Calibration runs inside normal inference with no model-file or interface changes:
   - **Collection mode (first N tokens / first pass):** every gate tensor that is computed for the first time is also packed into the binary gate predictor (`Sw` sign bits + per-block `aw` scales) on the fly, while simultaneously recording per-neuron gate statistics (`med`, `sd`, `b`) and the SwiGLU bias moments (`mu[i]` = mean h[i] when silent). No separate calibration script or precomputed npz is needed.
   - **Prediction mode (after collection):** once per-layer statistics are sufficient, the estimator and dead zone switch to using the calibrated values, and inference speeds up via active-neuron skipping. Collection can be re-triggered or frozen via a runtime flag.
   - **No interface impact:** model files, GGUF format, CLI args, and API surface are unchanged — calibration data lives in runtime memory alongside the existing K/V cache.

## Implementation

### Calibration (per layer, once)

For each gate neuron i:

1. Per-neuron bias statistics (from cached activations, ~10k tokens):
   - med[i] = median of gate_raw[i] over calibration tokens
   - sd[i] = std of gate_raw[i] over calibration tokens
   - b[i] = silu(med[i]) — the bias contribution to SwiGLU output
   - mu[i] = E[h[i] | silent] — mean SwiGLU output of neuron i when it is silent (computed with the same dead-zone rule used at inference)
   - c0 = Wdown @ mu — all-silent residual bias vector (n_embd), precomputed once per layer

2. Gate weight packing (for the XOR estimator):
   - For each gate row i, in blocks of B = 32 (B = 8 gives better quality):
     - Sw[i]: K bits, bit = (w >= 0), stored as K/32 uint32 per row
     - aw[i][b]: per-block scale, fp16 = mean(|w|) over the block (mean-abs is L2-optimal for a sign code; do not use block max — it inflates small weights)

   Cost: 1 + 16/B bits per weight (1.5 bpw at B=32, 3 bpw at B=8). Full-precision gate/up/down stay in memory unchanged.
   Optional: Encode fp16 as 3bit select for 5bit exponent + 3 bit mantissa into 6 bit. 1 + 6/B is 1.375 bpw at B=16.

### Online inference (per token, per layer)

1. Gate estimate (router signal only):
   - Quantize input x (after FFN norm) once per token:
     - Sx: K bits, bit = (x >= 0)
     - ax[b]: mean(|x|) per block of B
   - Per row i, per block b:
     agree = B - 2 * popcount(Sw[i][b] XOR Sx[b])   // sum of sign(w)*sign(x), range [-B, B]
     g~[i] += aw[i][b] * ax[b] * agree
   - g~[i] is used ONLY to decide silent vs active. Never reused as the actual gate value.
   - XOR + popcount = one op per 32 weights (AVX2 / AVX-512 VPOPCNT / NEON cnt). Memory traffic ~1/10 of f16 gate.
   - If estimator error is too high, first try 2-bit x (sign + large flag) or B = 8.

2. Dead zone (two-sided, bias-relative):
   - Per layer and token: m = mean_i |g~[i]| (one extra reduction), or calibrated per-layer constant.
   - silent[i] = |g~[i] - bias_proxy[i]| < T * m   // band around per-neuron bias
   - active[i] = !silent[i]                          // large positive OR large negative deviation
   - bias_proxy[i] is a cheap proxy for med[i] (e.g., per-row median of g~, or the calibrated med[i] itself).
   - Sweep T in {0.25, 0.5, 0.75, 1.0}; look for the knee. Floor: keep at least 2% active per layer.
   - Optional decode prior: active[i] |= prior_active[i] from previous token (exp11).

3. Active-neuron computation (full precision, delta down projection):
   For each active i:
   - g[i] = full-precision dot(Wgate[i], x)
   - u[i] = full-precision dot(Wup[i], x)
   - h[i] = silu(g[i]) * u[i]
   - d[i] = h[i] - mu[i]   // delta over silent baseline

   Down projection (transposed matrix, column i contiguous):
   y = c0 + sum_{i in active} Wdown[:, i] * (h[i] - mu[i])

   - c0 added once per layer (one vector add).
   - Axpy-style: for each active column i, add (h[i] - mu[i]) * Wdown[:, i] onto y.
   - Stripes of 16 columns; per-thread partial residual vectors summed at the end.

4. Optional sign-conditioned bias (only if calibration shows a gain):
   - mu_p[i], mu_n[i] = mean h[i] when silent with g~ > 0 / g~ < 0.
   - Build c0 from mu_n. For silent neurons with g~ > 0 add Wdown[:, i] * (mu_p[i] - mu_n[i]).
   - Restrict to a precomputed list of biased neurons (top ~1%).

## Validation plan (in order)

1. Collect offline information and analyze collected data.  
2. Check prediction quality and adjust deadzone size.  
3. Measure end-to-end perplexity and KL on a text set (wikipedia).  



## Implementation status (CPU, llama.cpp)

Code: `src/llama-dna.{h,cpp}` (state, calibration, kernels), hooks in `llm_graph_context::build_ffn` (`src/llama-graph.cpp`) and `llama_context` (`src/llama-context.cpp`). The CPU backend exports `ggml_get_type_traits_cpu` through `ggml_backend_reg_get_proc_address` so the active rows use the native quantized dot products.

## Enable / disable

DNA is controlled by `--dna` (general setting, `common/arg.cpp`, env `LLAMA_ARG_DNA`) and the booleans flow `common_params.dna` → `mparams` (`common_model_params_to_llama`) and `cparams.dna` (`common_context_params_to_llama`). The runtime enablement now checks `cparams.dna` (`src/llama-context.cpp`), not an env var. Per-layer checks remain in `llama_dna::get_layer()` (`src/llama-dna.cpp:525`) and `llama_dna_row_is_readable()` (`src/llama-dna.cpp:278).

When `--dna` is set, the CLI forces the load defaults DNA needs and will error if they were overridden explicitly:

- `--load-mode=mmap` (or `mmap+mlock`); if a device demotes mmap→none, that's now a hard error (`src/llama-model.cpp:1700`), matching `llama-model-loader.cpp:917` for unsupported platforms
- `--no-repack` (i.e. `use_extra_bufts=false`)

| switch | default | effect |
|---|---|---|
| `--dna` / `LLAMA_ARG_DNA` | off | enable DNA for dense SwiGLU FFNs whose weights are in plain host memory (no biases, no LoRA, no repacked buffers) |
| `--no-dna` | — | explicitly disable DNA |

| tuning variable | default | meaning |
|---|---|---|
| `LLAMA_DNA_T` | 0.5 | dead zone half-width T |
| `LLAMA_DNA_ZMODE` | 0 | 0: band = T * mean(abs(g~)) per layer, 1: band = T * std(g~) per neuron |
| `LLAMA_DNA_BLOCK` | 32 | estimator block size B (8, 16, 32, 64) |
| `LLAMA_DNA_CALIB` | 4096 | tokens per calibration phase (two phases: gate statistics, then silent bias mu) |
| `LLAMA_DNA_MIN_ACTIVE` | 0.02 | floor on the active fraction, the band is shrunk at the end of calibration if needed |
| `LLAMA_DNA_MAX_BATCH` | 32 | ubatches larger than this use the dense FFN after calibration (set to 512 to evaluate perplexity with DNA) |
| `LLAMA_DNA_VERBOSE` | 0 | print per-layer active fraction and kernel times when the context is freed |

## Usage / How to run

```bash
# Enable DNA; per-turn stats print server-side at the end of each slot completion.
# NOTE: the common CLI sets log verbosity to ERROR by default, so you must raise it
# with -lv 3 (INFO) or -lv 2 (WARN) to see DNA startup and per-turn stats.
./llama-cli --dna -m model-f32.gguf -p "hello world" -lv 3

# DNA disabled (default)
./llama-cli -m model-f32.gguf -p "hello world"
```

The common CLI (`tools/cli/`) sets `params.verbosity = LOG_LEVEL_ERROR` by default
(`tools/cli/cli.cpp:36`), which suppresses all `LLAMA_LOG_INFO` / `LLAMA_LOG_WARN`
messages — including the "DNA enabled" startup line and per-turn DNA stats.
Use `-lv 3` (INFO) or `-lv 2` (WARN) to see them.

Per-turn stats (server-side, after each completed turn), printed from
`server_slot::print_timings()` in `tools/server/server-context.cpp`:

```
DNA on (30/30 layers) — calibration 6/8192 (stats 30, bias 0)
DNA on (30/30 layers) — prediction: 33 tokens, active min 63.7% avg 68.9% max 72.5%
```

- calibration shows slowest-layer progress over the two phases (`2 * LLAMA_DNA_CALIB` tokens total),
  with per-phase layer counts
- prediction shows cumulative predicted tokens and per-layer active-fraction min/avg/max

Per-layer stats are also printed at context destruction (only when `LLAMA_DNA_VERBOSE=1`):

```
DNA: layer  3: 12345 tokens predicted, active = 12.5%, per token: observe 0.001 ms, route 0.004 ms, down 0.003 ms
```

- `active = 12.5%` is the fraction of FFN neurons actually computed — lower means more compute saved.
- Calibration messages are logged at info level (`--verbose` in the tools).
- With `LLAMA_DNA_T=0` every neuron is active and the output equals the dense FFN (checked: identical perplexity on F16 and Q4_0 test models).

DNA only takes effect on layers that pass all of these checks (enforced in `llm_graph_context::build_ffn` and `llama_dna_layer::get_layer`):

- Dense SwiGLU FFN (`type_op == LLM_FFN_SILU && type_gate == LLM_FFN_PAR`)
- Weights in plain host memory (buffer name `"CPU"` or `"CPU_Mapped"`; repacked extra buffers rejected)
- Weight type F32 or convertible via `ggml_get_type_traits(type)->to_float`
- Affine terms **supported** (Restrictions 3, 4 lifted): `up_b`/`gate_b`/`down_b` and `up_s`/`gate_s`/`down_s` are copied into the layer at `get_layer` time and reproduced in dense order — `(W·x + b)·s` for gate/up in `op_route`, `(W_down·h + down_b)·down_s` for the down projection (folded into `c0`/`down_t`). Each must be a readable host tensor with the right shape (`gate_b`/`up_b` `{n_ff}`, `down_b` `{n_embd}`, `*_s` `{1}`); otherwise the layer is rejected.
- No LoRA on gate/up/down
- Activations: any type via graph-level cast (Restriction 7 lifted) — non-F32 `cur` is `ggml_cast` to F32 before the DNA ops (`dna_x_f32`); the kernels always see F32. 2D tensor required (`cur->ne[2] == 1 && cur->ne[3] == 1`).
- Batch size ≤ `LLAMA_DNA_MAX_BATCH` (default 32); larger batches fall back to dense FFN

Note: quantized GGUF models (e.g. Q4_0, IQ4_XS) use repacked buffers, so DNA layers are rejected for those models via the host-memory check. Per-tensor scales alone no longer reject a layer. Use an unquantized F32 model to test DNA.

### The `no_alloc` / `--fit` footgun (root cause)

`llama_model_params.no_alloc` (`common/common.h:784`, default `false`) is **not** set by any
CLI flag. It flows unchanged through `common_model_params_to_llama()` (`common/common.cpp:1719`)
→ `common_model_load()` → `llama_model_load_from_file()`.

The trap: **`--fit` is on by default** (`common/common.h:497: bool fit_params = true;`).
Before the real model is loaded, `common_init_result` calls `common_fit_apply()`
(`common/common.cpp:1281`), which runs `common_fit_params()` → `common_params_fit_impl()`
(`common/fit.cpp:980`), which measures memory via

```cpp
// common/fit.cpp:70-72 (also :178-180 in common_fit_measure)
llama_model_params mparams_copy = *mparams;
mparams_copy.no_alloc  = true;
mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;
```

This is a *probe-only metadata pass* to estimate device memory. **But** the real inference
model is then loaded with the *original* `mparams` — and, crucially, the probe itself creates
a `llama_context` (`llama_init_from_model`, `fit.cpp:79`) whose constructor
(`llama_context()`, `src/llama-context.cpp:618-624`) evaluates

```cpp
if (const char * env = getenv("LLAMA_DNA"); env && atoi(env) != 0) {
    if (model.no_alloc()) {
        LLAMA_LOG_ERROR("%s: DNA requested via LLAMA_DNA but model is in no_alloc mode, DNA disabled\n", __func__);
    } else {
        dna = std::make_unique<llama_dna>(...);
    }
}
```

For the **probe** model this fires with `no_alloc == true`, so DNA is disabled **for every
probe context**, and the error is printed — even though the real model (loaded next with
`no_alloc == false`) would have worked. That is exactly the symptom:

```
Loading model... |0.00.382.135 E llama_context: DNA requested via LLAMA_DNA but model is in
no_alloc mode, DNA disabled
```

**Why `model.no_alloc()` is the wrong gate.** The `no_alloc` flag is a global, coarse
"metadata-only" marker plumbed through `LlamaFastSearch` — it is not a per-weight property.
DNA's real requirements are *per-tensor*: plain host memory, exact buffer name `"CPU"` /
`"CPU_Mapped"`, valid `w->data` pointer. Those are already enforced per layer by
`llama_dna_row_is_readable()` (`src/llama-dna.cpp:278-288`), which checks
`w->data`, `w->buffer`, `ggml_backend_buffer_is_host()` and the buffer name.

**Fix.** The whole-model `model.no_alloc()` gate was removed from `src/llama-context.cpp:618-624`:
DNA is now always created when `LLAMA_DNA != 0`; per-layer readability checks in
`llama_dna::get_layer()` (`src/llama-dna.cpp:525`) reject genuinely metadata-only loads
(same `LLAMA_LOG_ERROR` path, per layer). No other code depends on the model-level `no_alloc`
gate for DNA.

**Workarounds (without the fix):**

```bash
# 1) Disable the fit probe entirely (cleanest workaround)
LLAMA_DNA=1 LLAMA_DNA_VERBOSE=1 ./llama-cli -m model-f32.gguf -p "hello" -lv 3 --fit off

# 2) Same, via env (see arg.cpp:2873: -fit/--fit, env LLAMA_ARG_FIT)
LLAMA_DNA=1 LLAMA_ARG_FIT=0 ./llama-cli -m model-f32.gguf -p "hello"
```

> **--fit off is orthogonal to DNA enablement.** `--fit` uses `no_alloc=true` for the
> probe only; DNA's checks no longer gate on it. Use `--dna` (not `LLAMA_DNA=1`) — the
> env var is removed. DNA keeps tuning knobs via env (`LLAMA_DNA_T`, etc. in
> `src/llama-dna.cpp:453-463`).

## compile:

```
# inside /mnt/c/Users/tobia/Desktop/tobifun/sandbox/forks/llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j4 --target llama-cli
```

## Known restrictions

The following restrictions are currently enforced. Each has a corresponding TODO to lift it.

| # | Restriction | TODO |
|---|-------------|------|
| 0 ✅ | **Silent DNA skip** — DNA previously skipped layers without any visible message when requirements weren't met (log level was WARN/INFO, suppressed by the CLI's default ERROR verbosity). | ✅ Fixed: skip reasons now logged at `LLAMA_LOG_ERROR` level in `llama-dna.cpp`, `llama-graph.cpp`, and `llama-context.cpp`, so they appear even at default CLI verbosity. |
| 1 ✅ | **CPU backend only** — DNA ops (`op_route`, `op_down`) have no GPU/HIP/SYCL backend implementations. | This is fine and accepted |
| 2 | **Dense SwiGLU FFNs only** — MoE / shared-expert / other activation variants are skipped (`type_op == LLM_FFN_SILU && type_gate == LLM_FFN_PAR`). | TODO: extend support to MoE layers and other activation functions. |
| 3 ✅ | ~~**No biases** — `up_b`, `gate_b`, `down_b` must all be null.~~ | ✅ Fixed: biases are copied at `get_layer` time (`llama_dna_read_dense`, shape-checked `n_ff`/`n_embd`) and applied in dense order in `op_route` (`(W·x + b)·s`) and in `llama_dna_finalize_bias` (`c0 = down_s·(W_down@mu + down_b)`, `down_t` pre-scaled by `down_s`). |
| 4 ✅ | ~~**No per-tensor scales** — `up_s`, `gate_s`, `down_s` must all be null; scaled low-precision types (NVFP4/F8_E4M3) rejected.~~ | ✅ Fixed: scalar scales `{1}` are read by `llama_dna_read_scale` and applied in the same spots as the biases (see row 3). NVFP4/F8_E4M3 weights already have `to_float` + CPU `vec_dot`, so they pass once the post-matmul multiply is reproduced. |
| 5 ✅ | **No LoRA** — `has_lora(up/gate/down)` must be false. | This is fine and accepted |
| 6 ✅ | **Weights in plain host memory only** — default is mmap and no repack | currently also needs --fit off |
| 7 ✅ | ~~**F32 activations only** — `cur->type == GGML_TYPE_F32` required.~~ | ✅ Fixed: non-F32 inputs are `ggml_cast` to F32 in `build_ffn` before the DNA ops (both calibration-observe and predict paths), reusing ggml's cast kernels for all GGUF quant types; the kernels keep the F32 assumption. |
| 8 | **Batch size limit** — ubatches larger than `LLAMA_DNA_MAX_BATCH` (default 32) fall back to dense FFN. | TODO: implement batched route/down for large batches to avoid the dense fallback. |
| 9 ✅ | ~~**Not compatible with `no_alloc` mode**~~ — Previously DNA was disabled when `model.no_alloc()` was true, but that flag is misleading: `--fit` (default on) does an internal `no_alloc=true` probe load, and the real inference model always has `no_alloc=false`. The guard has been removed; per-layer weight readability (`llama_dna_row_is_readable`) already rejects genuinely metadata-only loads. | ✅ Fixed: whole-model `no_alloc` gate removed from `llama-context.cpp` (see "The `no_alloc` / `--fit` footgun (root cause)" above). |
| 10 ✅ | **Stats print only at context destruction** — previous: `LLAMA_DNA_VERBOSE=1` only printed at destruction, no per-turn stats. | ✅ Fixed: server-side per-turn `LLAMA_LOG_INFO` in `server_slot::print_timings()` (`tools/server/server-context.cpp`) via `llama_dna::get_stats()` (`src/llama-dna.cpp`) and `llama_get_dna()` (`include/llama.h`, `src/llama.cpp`, `src/llama-context.cpp`). Shows on/off per layer, calibration `done/total` with phase breakdown, and prediction min/avg/max active fraction. Visible with `-lv 3` (INFO). |
| 11 ✅ | ~~**`--no-repack` referenced in the env table does not exist**~~ — The flag now exists (`common/arg.cpp:2395` `-nr`/`--no-repack`); it gates the CPU repack extra-buffer (`use_extra_bufts`, default on). DNA forces `--no-repack` on (`common/arg.cpp:917` post-parse: `params.no_extra_bufts = true`, so `use_extra_bufts=false` down to `make_cpu_buft_list`, `src/llama-model.cpp:1262`), and throws if the user supplied an incompatible explicit override. | ✅ Fixed: `--dna` defaults `--load-mode=mmap` and `--no-repack` via parse-time mutation in `common/arg.cpp` (before `common_model_params_to_llama`, which also asserts the invariants). DNA no longer depends on the user remembering `--no-repack`. `--fit` still correctly forces `no_alloc=true` for probe measurement only; the `no_alloc` workaround (`--fit off`) is orthogonal. |
| 12 ✅ | **mmap mode** | Works today via `CPU_Mapped` buffer (`ggml/src/ggml-backend.cpp:2958`), which DNA whitelists (`src/llama-dna.cpp:284`), so `w->data` reads valid bytes via the mapping. Remaining gap is sparse/lazy I/O: `lazy_mode` (`TENSOR_READ_LAZY`, `common/arg.cpp:2681`, `src/llama-model-loader.cpp:1183`, requires mmap) and per-neuron unmapping (`unmap_weight`, `src/llama-model-loader.cpp:1620`) are not implemented for DNA, so silent rows still cost a page fault per touch and a cold neuron's 4K page is only skipped if never touched. | TODO: per-neuron read path / `madvise(DONTNEED)` for silent rows; today DNA reads `w->data` rows directly and will fault any touched page. |

Per layer the runtime state holds the packed gate (1 + 16/B bpw), the transposed f16 copy of W_down (2 bpw) and the per-neuron `med`, `band`, `mu` plus `c0`. Calibration messages are logged at info level (`--verbose` in the tools).
