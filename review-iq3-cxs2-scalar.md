# Review: Scalar IQ3_CXS2 Implementations

Two scalar reference implementations in [DNA_performance.md](file:///home/tobiulla/sandbox/forks/llama.cpp/DNA_performance.md):

1. **Indirect-row GEMV (MR=4)** — sparse gate/up dot products ([lines 537-631](file:///home/tobiulla/sandbox/forks/llama.cpp/DNA_performance.md#L537-L631))
2. **Transposed reference down tile** — sparse down projection accumulation ([lines 384-512](file:///home/tobiulla/sandbox/forks/llama.cpp/DNA_performance.md#L384-L512))

---

## Format recap

| Property | Value |
|---|---|
| Block size | 64 bytes (1 cache line) |
| Weights per block | 160 (10 subblocks x 16) |
| Bits per weight | 3.2 (512 bits / 160 weights) |
| Subblock payload | 16-bit sign mask + 32-bit 2-bit indices = 6 bytes |
| Scale selectors | 10 x 3-bit selectors packed in bytes 60..63 (bits 0..29) |
| Magnitude codebook | {1, 3, 9, 27}, sign-symmetric |
| Total used bits | 10 x (16 + 32 + 3) = 510; 2 reserved bits |

---

## Correctness: Indirect-row GEMV (MR=4)

### Bit extraction — correct

- **2-bit indices**: `(indices >> (2*i)) & 3u` for `i` in 0..15 → shifts 0, 2, ..., 30. Extracts all 16 two-bit fields from the 32-bit word. ✓
- **Sign bits**: `(signs >> i) & 1u` for `i` in 0..15 → all 16 bits from the 16-bit mask. ✓
- **Scale selectors**: `(selectors >> (3*subblock)) & 7u` for `subblock` in 0..9 → shifts 0, 3, ..., 27. Uses bits 0..29 of the selector word, leaving bits 30..31 reserved. ✓
- **Payload addressing**: `block->bytes + subblock * 6` for subblock 0..9 → byte offsets 0, 6, 12, ..., 54. Last payload ends at byte 60. ✓ (bytes 0..59 are payload, 60..63 are selectors)

### Q8_0 alignment — correct

Each 16-weight IQ3_CXS2 subblock maps to exactly one half of a 32-element Q8_0 block:

| Subblock | input_offset | q8_block | q8_offset |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 16 | 0 | 16 |
| 2 | 32 | 1 | 0 |
| 3 | 48 | 1 | 16 |
| ... | ... | ... | ... |
| 9 | 144 | 4 | 16 |

One IQ3_CXS2 superblock (160 weights) spans exactly 5 Q8_0 blocks. The assertion `q8_offset == 0 || q8_offset == 16` is always true. Each pair of consecutive subblocks shares one `activation_scale = x_scales[q8_block]`, which is correct — Q8_0 has one scale per 32 elements. ✓

### Activation LUT — correct

```c
activation_lut[i][k] = q * magnitude_lut[k]   // int8 x int32 → int32
```

Built once per 16-weight subblock, shared across up to 4 rows.

- Range: `|q| <= 127`, `max(magnitude_lut) = 27` → max `|term| = 3,429`
- 16 accumulated terms per row: max `|dot| = 16 * 3,429 = 54,864` — well within int32 range. ✓
- LUT cost: 64 int multiplies amortized over MR rows.

### Scale decomposition — correct

```
accum[r] += (float)dot * activation_scale * row_scales[r] * block_scale_lut[scale_index]
```

This correctly factors the full dot product as:
- **Integer part**: `Σ_i  q8[i] * magnitude[index_i] * sign_i` (accumulated losslessly in int32)
- **Float scale**: `q8_block_scale * superblock_row_scale * block_scale` (one FP multiply+FMA per subblock per row)

The deferred float conversion is a good design choice: it avoids per-element FP rounding and reduces FP pipeline pressure. ✓

### Output scatter and tail handling — correct

- Tail groups with `row_count < 4`: only `accum[0..row_count-1]` is computed and stored. The extra `accum` slots are initialized but unused. ✓
- Output: `output[rows[group_start + r].output_row] = accum[r]` — correctly scattered. ✓

> [!NOTE]
> **No bugs found.** The GEMV implementation is correct for the stated format and Q8_0 input conventions.

---

## Correctness: Transposed reference down tile

### Accumulation structure — correct

The neuron-major loop processes each active neuron's tile contribution:

```
sum[output_pos] += activation * superblock_scale * block_scale * magnitude * sign
```

The `dna_iq3_cxs2_accumulate_subblock` helper pre-multiplies `superblock_activation * block_scale` into a 4-entry `scaled_lut`, then applies sign and accumulates. Same bit extraction as the GEMV — verified above. ✓

### Block indexing — correct

```c
first_block = tile_number * tile_cache_lines
block_index = first_block + line        // line ∈ [0, tile_cache_lines)
block = active[n].blocks + block_index  // neuron-major indexing
```

Each neuron's `blocks` array is indexed by absolute block number. `superblock_scales` uses the same index. This is consistent with the stated neuron-major, tile-major storage. ✓

### Output offset — correct

```c
output_offset = line * 160 + subblock * 16
```

For `tile_cache_lines` lines and 10 subblocks per line, this covers positions `0` through `tile_weights - 1` contiguously and without overlap. ✓

### BF16 conversion — correct

- `dna_bf16_to_f32(0x0000)` = `+0.0f`. ✓
- `dna_bf16_to_f32(0x8000)` = `-0.0f`, and `-0.0f == 0.0f` per IEEE 754 — negative zero is correctly skipped. ✓
- `dna_f32_to_bf16` rounds to nearest even at the BF16 boundary: `bits += 0x7fff + ((bits >> 16) & 1)`. Standard technique. ✓

> [!NOTE]
> **No bugs found.** The down-tile implementation is correct for the stated format and block addressing conventions.

---

## Performance properties

### Indirect-row GEMV (MR=4)

#### Work per subblock per row

| Operation | Count | Domain |
|---|---:|---|
| Activation LUT build (`q * mag[k]`) | 64 / MR = **16** amortized | int32 |
| Index extract + LUT lookup | 16 | int32 |
| Sign-conditional negate + accumulate | 16 | int32 |
| Scale multiply + FMA into accum | 3 | float |
| **Total per row per subblock** | **~35** | mixed |

#### Activation reuse from MR grouping

The activation LUT (256 bytes) is built once per subblock and consumed by up to 4 rows. This amortizes 64 integer multiplies across MR rows. At MR=1, the LUT would be pure overhead (64 multiplies for 16 lookups). At MR=4, it saves `3 * 64 = 192` multiplies per subblock relative to rebuilding per row.

#### Weight access pattern

Each row group loads 4 independent 64-byte blocks per superblock. Within the superblock, the 10 subblock iterations re-read each block's payload bytes at different offsets — this is sequential access within an already-loaded cache line, so the cost is register/decode, not memory.

Across row groups, the blocks pointed to by `rows[]` may be arbitrarily scattered in the original weight matrix. **Weight locality between rows within a group is not guaranteed.** For 4 non-adjacent rows, the 4 cache-line loads per superblock may each miss independently.

#### Scaling behavior

- **Linear in `active_count`**: O(active_count * n_inputs / 160 * 10) subblock operations.
- **Tail cost**: if `active_count % 4 != 0`, the last group underutilizes the LUT amortization (but wastes at most 3 LUT builds).
- **No SIMD utilization**: the 16-element inner loops are serial on scalar. A SIMD implementation could process 8 or 16 elements per cycle.
- **Bottleneck**: on scalar, the inner 16-iteration loop with branch (`? -term : term`) and dependent accumulation is likely the critical path. At ~2 cycles/iteration (branch + add), one subblock per row takes ~32 cycles, or ~320 cycles per superblock per row.

#### Bytes read per output element

For one row spanning `n_inputs` elements:
- Weight: `(n_inputs / 160) * 64` bytes = `0.4 * n_inputs` bytes
- Activation: `n_inputs` bytes (int8) + `(n_inputs / 32) * 4` bytes (scales) = `1.125 * n_inputs` bytes (shared across MR rows)
- Amortized activation per row: `1.125 * n_inputs / MR`
- **Total per row**: `~0.68 * n_inputs` bytes at MR=4

#### Compute intensity

~35 ops per 16 weights → ~2.2 ops/weight. At 0.4 bytes of weight data per weight:
- **~5.5 ops/byte** (weight data only) — compute bound on scalar, memory bound only if weights stream from DRAM.

---

### Transposed reference down tile

#### Work per subblock per neuron

| Operation | Count | Domain |
|---|---:|---|
| Scaled LUT build (`block_activation * mag[k]`) | 5 (1 mul for block_activation + 4 for LUT) | float |
| Index extract + LUT lookup | 16 | float |
| Sign-conditional negate | 16 | float |
| Accumulate into sum | 16 | float |
| **Total per subblock per neuron** | **~37** | float |

#### Memory access pattern

**Output tile (sum_scratch):** `tile_weights * 4` bytes of F32. For `tile_cache_lines = 1`: 640 bytes (10 cache lines). This fits comfortably in L1 and is revisited for every active neuron.

**Weight data:** For each active neuron, `tile_cache_lines * 64` bytes are read. These are the neuron's weight blocks for this tile. Different neurons' blocks are at different addresses — **no spatial locality between neurons**.

**Neuron traversal pattern:** The code visits each neuron's entire tile before moving to the next neuron. Within one neuron's tile, the cache lines are contiguous (neuron-major storage). But across neurons, each `active[n].blocks + block_index` is a different base address.

#### Comparison with the GEMV

| Property | GEMV (MR=4) | Down tile |
|---|---|---|
| Shared data | Activation LUT (int32) | Output accumulator (float) |
| Reuse axis | Same activation, different rows | Same output positions, different neurons |
| Inner accumulation | Integer dot → one FP multiply | FP multiply-add per element |
| Inner loop branches | 16 sign branches per row per subblock | 16 sign branches per neuron per subblock |
| FP operations per 16 weights | 3 (scale + FMA) | ~37 (all in FP) |
| Weight locality within group | 4 scattered row blocks | 1 contiguous neuron block per line |
| Parallelism | Rows within MR group are independent | Output tiles are independent |

#### Key difference: all-FP vs. deferred-FP

The down tile does **all arithmetic in FP**: each weight is decoded to a float, multiplied by the activation, and accumulated. The GEMV defers FP to one operation per subblock per row by using an integer dot product. This means:

- The **GEMV** has lower FP pressure and better precision characteristics (lossless integer accumulation within each subblock).
- The **down tile** has higher FP throughput demand but avoids the integer LUT construction overhead.

For scalar code, the down tile's 37 FP ops per subblock per neuron will be the bottleneck. For SIMD, the down tile's output-position parallelism maps naturally to vector lanes (the AVX2 sketch in the document exploits exactly this).

#### Thread scaling

Both implementations are embarrassingly parallel across their respective partition axis:
- **GEMV**: rows can be partitioned across threads (each thread gets a subset of `rows[]`).
- **Down tile**: tiles are assigned to threads. Each thread owns disjoint output positions, so no cross-thread reduction is needed. ✓

The down tile's thread model is cleaner: tile boundaries are cache-line aligned (160 BF16 values = 320 bytes = 5 cache lines), so there is no false sharing.

---

---

## Correctness: Quantization reference

Added in [lines 642-933](file:///home/tobiulla/sandbox/forks/llama.cpp/DNA_performance.md#L642-L933).

### Scale hierarchy — correct

The quantizer implements the same 4-level decomposition consumed by both dequantization references:

```text
w_hat[i] = superblock_scale * block_scale_lut[selector] * magnitude[index] * sign
```

| Level | Scope | Determined by |
|---|---|---|
| `block_scale_lut[8]` | Per-matrix | Caller (not in this function) |
| `superblock_scale` | Per 160-weight line | Search over 9 candidates |
| `selector` (3-bit) | Per 16-weight subblock | Exhaustive over 8 entries |
| `index` (2-bit) + `sign` (1-bit) | Per weight | Nearest-neighbor in {1, 3, 9, 27} |

### Per-weight quantization — correct

```c
const float scaled = abs_w * inv_scale;
// nearest-neighbor over {1, 3, 9, 27}
```

- Sign is extracted first (`w < 0.0f`), then magnitude is quantized independently. This is correct for a sign-symmetric codebook. ✓
- The search over 4 magnitude entries is exhaustive. ✓
- Error is computed in the original domain (`abs_w - reconstructed`), not the scaled domain — this avoids scale-dependent error weighting. ✓

### Superblock scale search — correct

Initial estimate: `max_abs / (27.0f * median_block_scale)`. This sets the scale so the largest codebook entry (27) times the median block scale reproduces the peak weight. 9 candidates at ±4% steps around this estimate, covering ±16%. ✓

The median computation (insertion sort of 8 elements, average of elements 3 and 4) is correct. ✓

### Block packing — correct

- Payload: `dna_store_le16(payload, signs)` at `sub * 6`, then `dna_store_le32(payload + 2, indices)` — matches the byte layout `[signs_lo, signs_hi, idx0, idx1, idx2, idx3]`. ✓
- Selectors: `selectors |= r.scale_selector << (3 * sub)` for sub 0..9 — bits 0..29. ✓
- `dna_store_le32(block->bytes + 60, selectors)` — bytes 60..63. ✓
- Block is zeroed first (`memset`), so reserved bits 30..31 are zero. ✓

**Round-trip**: the packing matches the dequantization readers exactly (same byte offsets, same bit shifts, same endianness). ✓

### Quantizer search cost

Per superblock (160 weights):
- 9 superblock candidates × 10 subblocks × 8 selectors × 16 weights × 4 magnitudes = **46,080 comparisons**
- Plus the final encoding pass (10 × 8 × 16 × 4 = 5,120 comparisons)
- Total: ~51,200 comparisons per 160 weights — fine for offline use.

### Edge cases

- **All-zero weights**: `max_abs = 0` → `base_scale = 0` → all candidates are 0 or negative (skipped) → `best_scale = 0` → `effective_scale = 0` → `dna_iq3_cxs2_quantize_subblock` returns all-zero signs/indices with error = 0. ✓
- **Zero median_block_scale**: `base_scale = 0` → same as all-zero weights. The function still produces valid output, but with maximum quantization error. The caller should not pass a degenerate block_scale_lut. ✓
- **Negative block_scale_lut entries**: Not guarded against. The quantizer assumes all 8 block scale values are positive. If any are negative or zero, `effective_scale` could be negative, producing incorrect sign assignments. The document says the LUT comes from E5M3 values, which should be positive. No bug, but an implicit precondition. ✓

> [!NOTE]
> **No bugs found.** The quantization function correctly implements bottom-up search over the IQ3_CXS2 scale hierarchy and produces blocks byte-compatible with the dequantization references.

---

## Summary

All three implementations are **correct** as scalar reference code for the stated IQ3_CXS2 format. No bugs in bit extraction, addressing, scale decomposition, accumulation, or quantizer search.

The GEMV's integer-domain LUT trick with deferred FP conversion is the more interesting design — it reduces FP pressure and provides exact integer accumulation. The down tile's all-FP path is simpler but will be the bottleneck on scalar. Both are designed for straightforward SIMD lifting: the GEMV's row independence and the down tile's output-position parallelism each map to vector lanes.

The quantizer uses a reasonable search strategy (9 × 8 exhaustive) suitable for offline use. The per-matrix `block_scale_lut` determination is intentionally left to the caller.
