# DNA - Dynamic Neuron Activation

## Key idea

- for each layers MLP use a approximated gate projection to determine active neurons
- compute the active neurons full gate and up precision and compute SwiGLU et al
  - fill in learned bias term for silent neurons (skippable if done implicit in down projection)
- compute down projection only for active neurons on transposed matrix
  - add bias terms for silent neurons
  - use blocking and skipping

## details

### approximate gate projection

- use only sign from original gate
- use simple XOR scheme for compute


### determine active neurons

- use dead zone: smaller and larger values are considered active
- optional: use sign of gate approximation for +- bias correction


### full precision partial gate and up projection

- compute only for active neurons
- CPU friendly
- optional: compute in GPU shaders


### transposed biased down projection

- compute full residual contribution vector for each active neuron
  - stripes: pointers to N (16) columns and an N wide activation vector -> 1 residual vector
  - add vectors for all stripes (add per thread, then sum those)
- residual contribution of inactive neurons
  - depends on gate sign?
  - if no, then 
    - remove idle background from down weights -> active neurons compute the delta to inactive state
    - add an all-inactive bias residual vector



## Implementation draft (v0, untested)

Status: design proposal derived from the vllm/pred1.md experiments. No number below has been measured for this exact scheme.

Evidence used from pred1.md:
- exp23/24: threshold routing on an approximate gate with an accurate per-block scale beats fixed top-k (match 0.736 @T=0.5, 0.818 @T=0.20 with ~88% hot). So the dead zone must be narrow.
- exp21/22: a coarsely scaled approximate gate (E8M0, power of two) over-activates cold channels. Use a fine scale (fp16 or E5M3).
- exp20: do NOT replace up/down of silent neurons with a cheap approximation. Errors multiply in SwiGLU.
- exp4: wrong-sign + large-magnitude channels are the dangerous misses (~1.3% of channel-tokens).
- exp11: the previous token's approximate gate is a free, good prior (decode only).

### 1. Offline packing (per layer, once)

For each gate row i (length K = n_embd), in blocks of B = 32 (B = 8 is better for quality, see exp19):
- `Sw[i]`: K bits, bit = (w >= 0), stored as K/32 uint32 per row.
- `aw[i][b]`: per-block scale, fp16 = mean(|w|) over the block (E5M3 is acceptable). Mean-abs is the L2-optimal scale for a sign code. Do not use block max (exp17: it inflates small weights).

Cost: 1 + 16/B bits per weight (1.5 bpw at B=32, 3 bpw at B=8). The full-precision gate/up/down stay in memory unchanged.

### 2. Online gate estimate (per token, per layer)

Quantize the input x (after the FFN norm) once per token:
- `Sx`: K bits, bit = (x >= 0)
- `ax[b]`: mean(|x|) per block of B

Per row i, per block b:

    agree = B - 2 * popcount(Sw[i][b] XOR Sx[b])    // sum of sign(w)*sign(x), range [-B, B]
    g~[i] = sum_b aw[i][b] * ax[b] * agree

Notes:
- XOR + popcount is one op per 32 weights (AVX2 / AVX-512 VPOPCNT / NEON cnt). Memory traffic is ~1/10 of the f16 gate.
- Sign(x) * sign(w) discards magnitude correlation inside a block. If the error is too high, first try a 2-bit x (sign + "large" flag, two XOR/popcount passes weighted 1 and 2) or B = 8, before changing anything else.
- Fixed point is fine: aw*ax as int16 per block, accumulate in int32.

### 3. Dead zone (silent set)

Per layer and token, `m = mean_i |g~[i]|` (one extra reduction), or a calibrated per-layer constant if the reduction is too costly.

    silent[i] = |g~[i]| < T * m        // a band around zero
    active[i] = !silent[i]             // large positive AND large negative are active

Why a band around zero and not "negative = silent":
- the estimate is least reliable for large negative values (the exp4 "dangerous" quadrant), and those neurons are cheap to keep.
- the exp24 sweep rewards a small T.

Starting values (tune per layer by sweep, report active% and hidden cos-sim):
- Sweep T in {0.25, 0.5, 0.75, 1.0} and look for the knee. Expect ~80-90% active at T = 0.25 (exp24), which saves little. Real savings need the estimate to be sharper than in exp24.
- Floor: keep at least 2% of neurons active per layer (cheap insurance against dead layers; exp24 says unneeded).
- Optional prior (decode): `active[i] |= prior_active[i]` from the previous token's g~ (exp11). One bitmask OR.

### 4. Active-neuron computation

For i in active (bitmask converted to an index list in blocks of 16/32 rows, so skipping is block-granular):
- `g[i]` = full precision dot(Wgate[i], x). The sign estimate is NOT reused.
- `u[i]` = full precision dot(Wup[i], x)
- `h[i]` = silu(g[i]) * u[i]
- `d[i]` = h[i] - mu[i]    // delta against the silent baseline, see section 5

### 5. Bias handling for silent neurons

Silent neurons have h[i] ~ 0 but not exactly. Replace their contribution by its calibration mean.

Calibration (offline, ~10k tokens of representative text, real layer inputs):
- `mu[i] = E[h[i] | silent]`, collected with the same estimator and T as inference.
- `c0 = Wdown @ mu` (n_embd vector) = the all-silent residual bias, precomputed once per layer.

Inference:

    y = c0 + sum_{i in active} Wdown[:, i] * (h[i] - mu[i])

- A truly silent neuron contributes mu[i] (correct on average, folded into c0).
- An active neuron contributes its full h[i], because Wdown[:,i]*mu[i] is already in c0, so the delta form is exact.
- Down projection runs on the transposed matrix (column i contiguous): axpy onto the output, stripes of 16 columns, per-thread partial residual vectors, summed at the end.
- Per-token bias cost is one vector add per layer.

Expected effect: silent means |g| small, so h ~ g/2 * u has a largely random sign and mu[i] is probably close to 0 for most neurons. Measure |c0| / |y|. If tiny, drop the bias entirely. It matters mainly for channels with systematic offsets (massive-activation dimensions).

Optional sign-conditioned bias (only if the measurement shows a gain):
- `mu_p[i]`, `mu_n[i]` = mean h[i] when silent with g~ > 0 / g~ < 0.
- c0 uses mu_n. For silent neurons with g~ > 0 add Wdown[:,i]*(mu_p[i]-mu_n[i]). Restrict this to a precomputed list of "biased" neurons (top ~1%), otherwise it costs as much as an active neuron.

### 6. Validation plan (in order)

1. Offline in Python on cached activations: estimator error, quadrant stats (as exp4: right/wrong sign x large/small), active% vs T.
2. Per-layer MLP output cos-sim of DNA vs dense against active%. Goal: > 0.99 at <= 50% active before touching llama.cpp. If the sign-only estimator cannot reach it, add the 2-bit x or B = 8 first.
3. End-to-end top-1 match and KL on a text set (exp14 style), all layers at once. Per-layer errors compound (exp14: 0.61 hidden cos-sim -> 48% top-1 change).
4. Only then write llama.cpp CPU kernels: a popcount gate op, a masked/index-list gate+up op (MUL_MAT_ID-like), an axpy-style down op.

### 7. Open risks

- Sparsity in SwiGLU models is much lower than in ReLU models. Exp24 needed ~88% hot for good quality. With the extra estimator pass (~1.5-3 bpw read), DNA probably only wins if active% is below roughly 50-60%. This is the main unknown.
- Prefill batches activate the union of neurons over the batch, so DNA mainly helps single-token decode.
- Gather efficiency of skipped gate/up rows should be checked on the CPU target.

