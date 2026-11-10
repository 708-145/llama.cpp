# DNA - Dynamic Neuron Activation

## Key idea

For each layer's MLP, use an approximated gate projection to decide which neurons are active. Compute the active neurons' gate/up/down in full precision, and replace silent neurons with their calibrated bias contribution. Compute the down projection only for active neurons (delta form) on the transposed matrix, with blocking and skipping.

## Core principles

1. **Per-neuron bias.** A neuron's gate pre-activation idles at a neuron-specific median (typically 0.7–2.1 std from 0). down_input median is near 0 because the symmetric up projection hides the offset. Bias holds 31–55% of down_input energy — ignoring it is not an option.

2. **Two-sided dead zone on gate deviation.** Silence is defined by distance from the per-neuron bias, not by raw gate magnitude. A neuron is silent when |z| < threshold where z = (g - median) / std (or a cheap proxy). Both large-positive and large-negative deviations are active. One-sided rules (only z > 0) fail end-to-end.

3. **Selection by distance-from-bias.** Select active neurons by |s - b| (deviation of SiLU(gate) from bias contribution), not by raw |gate| or |gate_approx|. Distance-from-bias selection + bias fill beats magnitude selection + zero fill by 15–22x in KL.

4. **Bias fill (mu[i], c0).** For silent neurons, h[i] = silu(g[i]) * u[i] is replaced by its calibration mean mu[i]. Precompute c0 = Wdown @ mu once per layer. Down projection: y = c0 + sum_active Wdown[:, i] * (h[i] - mu[i]). The delta form is exact: silent neurons contribute mu[i] (folded into c0); active neurons contribute their true h[i].

5. **Gate estimate is a router only.** The approximate gate (XOR sign scheme) is used only to decide silent vs active. It is never reused as the actual gate value — active neurons always get full-precision gate/up.

## Implementation

### Offline calibration (per layer, once)

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

1. Offline on cached activations: estimator error, quadrant stats (exp4: wrong-sign + large-magnitude = dangerous misses), active% vs T, bias quality (|mu[i]| distribution, |c0| / |y|). If |c0|/|y| is not negligible, the bias is load-bearing.
2. Per-layer MLP output cos-sim of DNA vs dense against active%. Goal: > 0.99 at <= 50% active before touching llama.cpp. If the sign-only estimator cannot reach it, add 2-bit x or B = 8 first.
3. End-to-end top-1 match and KL on a text set (exp13/exp14 style), all layers at once. Per-layer errors compound (exp14: 0.61 hidden cos-sim -> 48% top-1 change; exp13: KL inverts top-1 ranking at low hot fractions, so report both). Report hidden cos-sim, top-1 match, KL(full||hybrid).
4. Only then write llama.cpp CPU kernels: popcount gate op, masked/index-list gate+up op, axpy-style down op with pre-added c0.

