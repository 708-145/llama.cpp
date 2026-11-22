# DNA performance: current issues

This note describes performance costs visible in the current CPU
implementation. It distinguishes observed behavior from likely bottlenecks
inferred from the operator structure; the latter require profiling before they
should be treated as measured results.

## Summary

DNA can save gate/up and down-projection work when many neurons are silent, but its current operators are row-oriented custom CPU kernels. They still evaluate the predictor for every neuron and process tokens one at a time.

The fraction of neurons marked active is not itself a speedup measurement. Large batches use the dense FFN by default, and DNA also adds calibration, weight-packing, and persistent-memory costs.

The current results in [DNA.md](DNA.md) report perplexity, predictor calibration correlation and active fractions, plus setup and whole-run times for smaller predictor quants. They do not provide matched dense-versus-DNA tokens/s, end-to-end latency, or peak-RSS measurements. No general speedup claim can be made from those results alone.

## Current costs and limitations

### 1. Large dense-FFN batches fall back to dense execution

`llama_dna_params::max_batch` defaults to 32. In `build_ffn`, a predicting dense/shared FFN is switched back to the ordinary dense path when its token dimension exceeds this value.

This is a performance guard, not an apparent shape/correctness limit in the custom route/down operators. `LLAMA_DNA_MAX_BATCH` can raise the guard; existing perplexity recipes use 512.

This particularly limits prompt prefill: batches above the configured cap get no DNA skipping in those FFNs, even after calibration. MoE DNA has a separate path and does not use this `max_batch` check.

References: [`llama_dna_params`](src/llama-dna.h), [`build_ffn`](src/llama-graph.cpp), and the batch-limit notes and recipes in [DNA.md](DNA.md).

### 2. Route work remains proportional to every neuron and token

For each token, `op_route` quantizes the input for the predictor and gate/up dot-product types, then estimates the gate value for every FFN neuron. It then skips the true gate/up dot products for neurons classified as silent.  
Active neurons still require separate row dot products for gate and up.

The loops are token-major and call row-vector kernels; they do not turn a token batch into a batched matrix multiply.

Thus, increasing sparsity avoids some gate/up reads and dot products, but does not remove the all-neuron predictor scan. At high active fractions, the remaining gate/up work may also leave little room to offset predictor overhead.

Reference: [`op_route`](src/llama-dna.cpp).

### 3. Down projection uses tiled sparse IQ3_CXS2

`op_down` now scans the delta once per token to build an active-neuron list, then
assigns disjoint 160-output tiles to threads. Each active neuron contributes its
64-byte IQ3_CXS2 block for the tile through an AVX2 kernel when supported, with
a scalar fallback. The kernel initializes from `c0` and writes each completed
tile once. Routed MoE down projection uses the same quantized blocks and tile
kernel, applying router weights to both the bias and active contributions.

The persistent FP16 transpose has been removed. Calibration uses a temporary
FP16 transpose as staging, then quantizes the scaled rows to IQ3_CXS2; the
staging allocation is released before prediction. The retained storage is
64 bytes per 160 weights plus eight E5M3 scales per neuron. Quantization thus
preserves the former FP16 staging precision, and the final output tile
accumulates in F32.

References: [`op_down`](src/llama-dna.cpp), and the scalar/AVX2 tile kernels in
[`dna-iq3-cxs2.c`](src/dna-iq3-cxs2.c).

### 4. Intermediate and persistent memory grow with model/batch size

- The route output (`delta`) contains `n_ff * n_tokens` F32 values, or `4 * n_ff * n_tokens` bytes per layer before allocator reuse. For example, `n_ff = 11008` and `n_tokens = 512` is about 21.5 MiB for one such tensor.
- After bias calibration, each layer retains IQ3_CXS2 down blocks and eight E5M3 scales per neuron, in addition to the model's original down weights.
- Each enabled layer retains its packed predictor gate (`pred_q`) and per-neuron calibration/prediction data. MoE layers hold corresponding state per expert.

The delta tensor is batch-dependent; the transposed weights and predictor are model/layer-dependent. Batching or removing the batch guard should therefore be measured for peak memory as well as speed.

References: the `delta` allocation in [`build_ffn`](src/llama-graph.cpp), and `down_blocks`/`down_scales`/`pred_q` storage in [`llama-dna.cpp`](src/llama-dna.cpp).

### 5. Setup and calibration delay the point at which skipping begins

When a layer is enabled, `llama_dna_pack_gate` reads/dequantizes its gate rows and quantizes them into the predictor format. This is a one-time per-layer setup cost and adds predictor storage.

- [ ] Option to store the computed predictors in sidecar GGUF. Keep existing calibrate + predict mode. 
- [ ] Option to load saved predictors and skip the calibrate phase.

Calibration runs the ordinary dense FFN and adds observers for two phases, configured by `LLAMA_DNA_CALIB`. Prediction savings begin only after a layer completes both phases. For short prompts or large calibration settings, DNA may spend much or all of the workload calibrating. The Gemma results in [DNA.md](DNA.md) include a four-chunk, `n_calib=4096` run where no layer reached PREDICT.

The reusable per-layer state is:

- `med_est[n_ff]`, the predictor-output center for each neuron's dead zone;
- `band[n_ff]`, its dead-zone half-width;
- `mu[n_ff]`, the mean silent-neuron FFN contribution;
- `c0[n_embd]`, the corresponding down-projection bias;
- `down_blocks[n_ff][ceil(n_embd / 160)]` and `down_scales[n_ff][8]`, the
  quantized down weights and E5M3 scale tables.

Store these arrays as F32 except for the packed down data and byte scales. Let
`B = ceil(n_embd / 160)`. The calibration state takes
`64 * n_ff * B + 8 * n_ff + 4 * (3 * n_ff + n_embd)` bytes, excluding small
metadata and tensor alignment. Save only after the layer reaches PREDICT;
in-progress calibration would also require its phase counters, accumulators,
and sampling reservoirs.

For example, `n_ff=14336`, `n_embd=4096` gives `B=26`: packed down state is
23,969,792 bytes (22.86 MiB), and `med_est`/`band`/`mu`/`c0` add 188,416 bytes
(184 KiB), for about **23.04 MiB per layer**. This is the reusable calibration
state; it does not include the source model weights.

For a cache that also skips predictor setup, include `pred_q`, which is
regenerated from the gate weights today. Its size is
`n_ff * ggml_row_size(pred_type, n_embd)`. At IQ4_XS and the example dimensions
that is 31,195,136 bytes (29.75 MiB), bringing the total to about **52.79 MiB
per layer**. It is optional if startup is allowed to rebuild the predictor.

Use names under a reserved prefix, for example
`llama.dna.v1.layer.<layer>.med_est`, `.band`, `.mu`, `.c0`, `.down_blocks`,
`.down_scales`, and optionally `.pred_q`. For routed experts, add an
`.expert.<id>` component; use `.shared` for a shared expert. Keep tensors
typed/shaped for direct validation on load: F32 vectors for the four float
arrays, I8 byte tensors for the packed blocks/scales, and the predictor's GGML
quant type for `pred_q`. The byte tensors can have shapes
`[64, B, n_ff]` and `[8, n_ff]`, respectively.

Sidecar metadata should include a format/version identifier, base-model
identity, layer/expert identity and dimensions, predictor type, and the
calibration settings that affect routing (`thresh`, `zmode`, `min_active`).
Record `n_calib` as provenance. On load, reject mismatched versions, model
identity, shapes, or relevant settings; a missing/invalid layer can fall back
to normal calibration. Check every prefixed name for collisions before writing
or merging. Write a complete sidecar to a temporary file and atomically
replace the prior one so interruption cannot leave a partial cache.

GGUF is indexed rather than append-only: merging means rewriting the GGUF
metadata/tensor directory and copying both the original and DNA tensor data,
not concatenating the sidecar bytes. Keep the sidecar as the safe initial
format; a merged model also requires confirming the model loader ignores the
extra reserved-prefix tensors and preserves the base-model identity used to
validate the cache.

Each run's recovered PPL for chunks 1–8 is recorded in [DNA.md](DNA.md). Those values are segment estimates derived from cumulative PPL, not independent evaluator runs for each chunk.

References: [`llama_dna_pack_gate`](src/llama-dna.cpp), [`op_observe`](src/llama-dna.cpp), and the calibration results in [DNA.md](DNA.md).

## TODO

The open items below consolidate the TODOs described throughout this document.

### Quality and comparison

- [x] Run matched dense-versus-DNA measurements for every predictor using the same model, corpus, chunking, calibration, threads, context, and batch settings.
- [x] Report PPL over an isolated prediction-on subset using the same equal-sized chunk formula for every run.
- [ ] Add a DNA lifecycle test as regression test for all end2end code updates.
- [ ] Add regression tests for the operator variants: IQ3_CXS2 in gate/up vs down mode. Scalar code vs all SIMD variants.

The matched quality sweep can be rerun from WSL with
[`scripts/dna-ppl-sweep.sh`](scripts/dna-ppl-sweep.sh):

```bash
bash scripts/dna-ppl-sweep.sh MODEL.gguf WIKITEXT.txt [OUTPUT_DIR]
```

It runs dense and all five quantized predictors sequentially with the settings
recorded in [DNA.md](DNA.md), waits two minutes between runs by default, and
saves a separate full log for each run. `LLAMA_PERPLEXITY` can point to a
non-default binary; `DNA_PPL_COOLDOWN_SECONDS` can adjust the pause. The output
directory must not already exist. The script stops on a failed run or a missing
final PPL estimate.

### Operator implementation

- [x] Implement IQ3_CXS2 quantization
  - [x] example code in md without blockscale compute
  - [x] clarify how to compute the 8 scales per row (neuron)
  - [x] implement the complete quantization code in pure C. If it profits from AVX2 then also add that codepath. No AVX512 path planned for quantization.
  - [x] implement and test in C codebase
- [x] Replace sparse down AXPY work with an efficient sparse down transformation.
  - [x] Implement reference C kernels for the 16-weight subblock down tile accumulation in md.
  - [x] Implement scalar C kernels for the 16-weight subblock down tile accumulation in C codebase.
  - [x] Implement SIMD kernels (e.g. AVX2/AVX-512/NEON) for the 16-weight subblock down tile accumulation in md.
  - [x] Implement AVX2 down tile accumulation in C codebase, with runtime dispatch and scalar fallback.
  - [ ] Implement L1-blocked batching over active neurons (up to 32 neurons per batch) to keep partial sums and weight blocks cache-resident.
  - [ ] Benchmark sparse down tile traversal versus dense down projection across thread counts, batch sizes, and active fractions.
  - Decisions:
    - Block granularity: 64 bytes covering 160 weights (ten 16-weight subblocks), ensuring cache-line alignment.
    - Thread partitioning: by output tiles (each thread owns a disjoint set of output positions, eliminating cross-thread sum reductions).
    - Accumulation precision: internal accumulation and output in F32.
    - Tail dimensions: pad the last block and pass the number of valid 16-weight subblocks to skip padded work.
    - Scale storage: eight E5M3 scale bytes per neuron; each block stores a 3-bit selector per subblock.
    - Default tile size: one 64-byte block, covering 160 output weights; multi-line tiles remain unmeasured.
  - Open Questions:

- [ ] Implement and benchmark sparse indirect-row multi-GEMV for gate/up.
  - [x] Implement C reference multi-row indirect kernel (`ggml_gemv_iq3_cxs2_mr4_indirect_q8_0`) with shared activation LUT in md.
  - [ ] Implement C reference multi-row indirect kernel (`ggml_gemv_iq3_cxs2_mr4_indirect_q8_0`) with shared activation LUT in C codebase.
  - [x] Implement SIMD multi-row indirect kernel (`dna_iq3_cxs2_sparse_down_tile_avx512`) with shared activation LUT in md.
  - [ ] Implement SIMD multi-row indirect kernel (`dna_iq3_cxs2_sparse_down_tile_avx512`) with shared activation LUT in C codebase.
  - [ ] Compare MR=4 against MR=1, MR=2, and MR=8 to find optimal register-pressure vs. activation-reuse balance across CPU architectures.
  - [ ] Benchmark cache behavior under scattered/non-adjacent row index distributions versus contiguous row subsets.
  - Decisions:
    - Multi-row indirect batching: MR=4 as the initial design point, sharing a 16-by-4 activation lookup table across 4 rows.
    - Computation domain: integer-domain dot accumulation (`int8` activation x `magnitude_lut` -> `int32` accumulator) with deferred FP scaling per subblock per row.
    - Output handling: scattered write-back directly into the dense output buffer via row indices.
  - Open Questions:
    - Weight locality penalty: arbitrary sparse neuron indices lack spatial locality in weight memory. At what active fraction does memory scatter negate the activation LUT reuse savings compared to dense GEMV/GEMM? <-- Operation is on full cache lines. Memory scatter will not occur, but cache line eviction due to n-way caches may be an issue.
    - Codebook zero representation: the codebook $\{-27, -9, -3, -1, 1, 3, 9, 27\}$ lacks an exact zero. Does quantization noise on silent/near-zero weights affect gating accuracy without an explicit zero-point or dead-zone threshold? <-- +-1 as zero replacement averages out if the sign is chosen randomly.
    - Per-matrix table selection: how should the 8 matrix-wide `block_scale_lut` values be derived across all rows during model quantization (e.g. k-means/Lloyd-Max clustering vs. fixed geometric intervals)? <-- it's only neuron wide. Each neuron / row gets a set of 8 possible block scales. Encoded as E5M3 to save space. Needs 8B storage per neuron. The 8 bits are cut from the FP16 representation by taking the 5 exponent bits and the first 3 mantissa bits. The scales need no sign. 

### Memory and scheduling

- [x] Release calibration-only state when each layer enters prediction. The statistics buffers are released during finalization, and the source down tensor and copied down bias are released at the transition. Keep predictor weights and derived prediction inputs (`med_est`, `band`, `mu`, `c0`, and sparse down weights).

### Benchmarking

- [ ] Measure load/setup and calibration time separately from steady-state prediction.
- [ ] Measure prompt throughput and decode throughput/latency. Make it into a performance regression test.
- [ ] Measure per-layer active fraction and route/down timings.
- [ ] Change the default batch guard only after sparse/dense crossover evidence supports it.

### Future topics

- [ ] Use sparsity inside matrix-matrix multiplication operations.
- [ ] Use sparsity inside the predictor so inference does not scan every neuron.
- [ ] Parallelize calibration layer scheduling according to available RAM.

- [ ] Reserve GGUF metadata fields and store calibration results directly in GGUF.
- [ ] Reserve GGUF metadata fields and support loading merged calibration tensors from the model GGUF. For the merged format, the gate/up/down tensors will also be in an appropriate format.


## Implementation Plan

### Sparse indirect-row multi-GEMV for gate/up

The route operator identifies a sparse subset of gate/up output rows to
evaluate. Today, the x86 Q4_0 dot kernel handles one row per call, so each row
independently loads the same quantized activation blocks. A multi-row kernel
can process a small group of selected rows together, loading each activation
block once and reusing it for all rows in the group. Depending on the weight
format, it can also share an activation-dependent lookup table. Row-specific
weight blocks, scales, and metadata still need separate loads and processing.

For IQ3_CXS2, the suggested name is
`ggml_gemv_iq3_cxs2_mr4_indirect_q8_0`. `mr4` means up to four rows per call;
`indirect` says rows are supplied by pointers or indices and need not be
adjacent in the original matrix. This is distinct from the existing repacked
`ggml_gemv_q4_0_8x8_q8_0`, which requires the `block_q4_0x8` interleaved
layout. MR=4 is the initial reference batch size, not a measured optimum.

The kernel needs:

- A list of active row pointers (or row indices plus the original row stride),
  and destination row indices if results are scattered back into the dense
  output.
- The activation vector quantized once into the dot-product format, such as
  Q8_0 blocks. Within each row group, each activation block is loaded once and
  reused for all rows.
- One accumulator per selected row, with the same accumulation precision and
  scaling semantics as the existing single-row dot path.
- Format-specific weight block decoding. For IQ3_CXS2, each row has its own
  64-byte blocks and per-subblock scale selectors. For LUT-based formats,
  activation-dependent LUT data can be shared; row-specific scales remain
  separate.

```text
quantize activation x once: x_q8[0 .. K/32)

for group_start = 0; group_start < active_count; group_start += MR:
    m = min(MR, active_count - group_start)
    acc[0 .. m) = 0

    for kblock = 0; kblock < K/32; ++kblock:
        a = load_q8_block_once(x_q8[kblock])
        a_scale = x_q8[kblock].scale
        shared_lut = make_activation_lut(a)   # only for LUT-based formats

        for r = 0; r < m; ++r:
            w = load_weight_block(row_ptr[group_start + r], kblock)
            row_scale = w.scale
            acc[r] += dot_block(w, a, row_scale, a_scale, shared_lut)

    for r = 0; r < m; ++r:
        output[row_index[group_start + r]] = reduce_and_store(acc[r])
```

`dot_block` is a format-specific helper that consumes the already-loaded
activation block; it must not reload that block for each row. For IQ3_CXS2,
the scalar MR=4 example below reads weights from separate ordinary row
addresses. A SIMD implementation could unroll the row loop to overlap
independent weight loads while retaining shared activation data. This is
software-indirect multi-row processing, not a hardware gather instruction or
a physical weight repack.

This design removes repeated activation-block loads within each row group and
can amortize activation-dependent LUT setup across the selected rows. It does
not reduce the number of weight bytes read, and arbitrary row pointers may
have poor weight locality. The existing x86 Q4_0 row kernel requires
`nrc == 1`; the IQ3_CXS2 MR=4 path is a new scalar reference with a tail for
groups smaller than MR. Benchmark it against the row-at-a-time path at matched
active-row counts and row distributions before choosing a production default.

### sparse down projection

The down transformation is a sparse sequence of active-neuron contributions.
For each token, the active list is applied to output tiles of 160 values. Each
neuron stores a neuron-major sequence of 64-byte IQ3_CXS2 blocks, so a tile
loads one block per active neuron and accumulates its decoded values into the
F32 output tile.

A GGML-style block model is more suitable than assuming every result block
is storage-aligned:

- Existing IQ4_XS and IQ3_XXS types use 256-value weight blocks, but they are legacy storage formats and do not satisfy the sparse-down cache layout: IQ4_XS stores 256 weights in a 136-byte block and IQ3_XXS in a 98-byte block.
- The sparse-down type is IQ3_CXS2, implemented in `src/dna-iq3-cxs2.c`:
  - Each 64-byte line encodes ten 16-weight subblocks: a 16-bit sign mask, 32-bit index field, and 3-bit scale selector per subblock.
  - Eight E5M3 scales are stored once per neuron, outside the block payload.
  - The line stores 510 meaningful bits for 160 weights, or 3.2 bpw before the per-neuron scale table.
- A block may cover 16, 32, or more weights depending on its packed format, but its byte size will remain a multiple of a cache line.
- The 16-weight granularity provides SIMD-compatible subblocks. Larger blocks may contain multiple 16-weight subblocks and span multiple cache lines.


For the sparse down projection, one useful model is:

- Split the output vector into a set of tiles. Tile width depends on the used quant type, which is cache line aligned. For IQ3_CXS2 it's 160 wide tiles.
- Split each tile into subblocks aligned to the quantized weight block layout and CPU SIMD width.
- Assign one tile to each thread.
- For a tile, scan only the active neurons.
- For each active neuron, scale and accumulate the weight subblocks of that
  neuron over the tile, including the quantization block scale/decode step.
- Write the completed tile once. Each thread owns disjoint output positions, so
  no cross-thread sum reduction is needed. If the output base is 64-byte
  aligned, a 160-element BF16 tile is 320 bytes, or five cache lines, so tile
  boundaries are cache-line aligned as well.

#### Reference C implementation

The following is format-independent. It assumes:

- one neuron entry contains one BF16 activation and a pointer to the first
  cache-line-aligned down-projection tile block;
- neuron blocks are stored neuron-major and tile-major;
- one tile block contains `tile_weights` weights and occupies
  `tile_cache_lines * 64` bytes;
- `tile_weights` is a multiple of 16;
- the caller passes one tile number and one tile size per thread;
- the caller passes only active neurons, or entries with zero activation are
  skipped;
- packed-weight decoding is supplied by a callback.

The implementation accumulates in F32 and converts only the completed tile
back to BF16. The generic version leaves format decoding to its callback; the
following scalar IQ3_CXS2 example shows one concrete decoder and tile path.

```c
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef uint16_t dna_bf16_t;

typedef struct dna_down_block {
    _Alignas(64) uint8_t bytes[64];
} dna_down_block;

typedef struct dna_active_neuron {
    dna_bf16_t activation;
    const dna_down_block *blocks;
} dna_active_neuron;

typedef void (*dna_decode_subblock_fn)(
    const dna_down_block *block,
    size_t                subblock,
    void                 *format_context,
    float                 decoded[16]);

static inline float dna_bf16_to_f32(dna_bf16_t value) {
    uint32_t bits = (uint32_t) value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline dna_bf16_t dna_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    // Round-to-nearest-even at the BF16 boundary.
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (dna_bf16_t) (bits >> 16);
}

void dna_sparse_down_tile(
    const dna_active_neuron *active,
    size_t                   active_count,
    size_t                   tile_number,
    size_t                   tile_cache_lines,
    size_t                   tile_weights,
    const dna_bf16_t        *c0,
    dna_decode_subblock_fn   decode_subblock,
    void                    *format_context,
    dna_bf16_t              *tile_sum)
{
    if (tile_cache_lines == 0 ||
        tile_weights == 0 ||
        tile_weights % 16 != 0) {
        return;
    }

    const size_t subblocks = tile_weights / 16;
    const size_t tile_offset = tile_number * tile_weights;
    float sum[tile_weights];

    for (size_t i = 0; i < tile_weights; ++i) {
        sum[i] = dna_bf16_to_f32(c0[tile_offset + i]);
    }

    for (size_t n = 0; n < active_count; ++n) {
        const dna_bf16_t activation = active[n].activation;
        if (activation == 0) {
            continue;
        }

        const float activation_f32 = dna_bf16_to_f32(activation);
        const dna_down_block *tile_block =
            active[n].blocks + tile_number * tile_cache_lines;

        for (size_t subblock = 0; subblock < subblocks; ++subblock) {
            float weight[16];
            decode_subblock(
                tile_block,
                subblock,
                format_context,
                weight);

            const size_t weight_offset = subblock * 16;
            for (size_t i = 0; i < 16; ++i) {
                sum[weight_offset + i] += activation_f32 * weight[i];
            }
        }
    }

    for (size_t i = 0; i < tile_weights; ++i) {
        tile_sum[i] = dna_f32_to_bf16(sum[i]);
    }
}
```

#### Scalar IQ3_CXS2 transposed reference CPU path

The bit count defines ten 16-weight groups per 64-byte line, but not the exact
field order. For this reference implementation, use this byte layout:

```text
bytes  0..59: ten 6-byte sign/index payloads
              each payload: 16-bit sign mask, then 32-bit 2-bit indices
bytes 60..63: ten 3-bit scale selectors in bits 0..29; bits 30..31 reserved
```

All fields are little-endian. A sign bit of one negates the selected magnitude.
This stores 10 * (16 + 32 + 3) = 510 bits, leaving two reserved bits in the
64-byte line. The four-entry magnitude LUT is sign-symmetric. The eight
selector scales below are passed in as F32 after converting the matrix's E5M3
scale table; `superblock_scales` supplies the applicable per-neuron,
per-output-line scale. These inputs follow the scale decomposition described
above. If the final format chooses another bit order or scale scope, the
reader/setup changes, while the accumulation loop stays the same.

The code reuses `dna_bf16_to_f32()` and `dna_f32_to_bf16()` from the preceding
reference implementation. It is scalar C: it decodes one 16-weight subblock,
scales its four-entry LUT, and accumulates its 16 contributions into F32 sums.

```c
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DNA_IQ3_CXS2_LINE_BYTES = 64,
    DNA_IQ3_CXS2_WEIGHTS_PER_LINE = 160,
    DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK = 16,
    DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE = 10,
};

typedef struct dna_iq3_cxs2_block {
    _Alignas(64) uint8_t bytes[DNA_IQ3_CXS2_LINE_BYTES];
} dna_iq3_cxs2_block;

_Static_assert(sizeof(dna_iq3_cxs2_block) == DNA_IQ3_CXS2_LINE_BYTES,
               "IQ3_CXS2 block must be exactly one cache line");

typedef struct dna_iq3_cxs2_active_neuron {
    dna_bf16_t activation;
    const dna_iq3_cxs2_block *blocks;
    const float *superblock_scales;
} dna_iq3_cxs2_active_neuron;

typedef struct dna_iq3_cxs2_tables {
    float magnitude_lut[4];
    float block_scale_lut[8];
} dna_iq3_cxs2_tables;

static inline uint16_t dna_load_le16(const uint8_t *p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static inline uint32_t dna_load_le32(const uint8_t *p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) |
           ((uint32_t) p[3] << 24);
}

static void dna_iq3_cxs2_accumulate_subblock(
    const dna_iq3_cxs2_block *block,
    size_t                   subblock,
    float                    superblock_activation,
    float                    block_scale,
    const dna_iq3_cxs2_tables *tables,
    float                    sum[DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK])
{
    assert(subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE);

    const uint8_t *payload = block->bytes + subblock * 6;
    const uint16_t signs = dna_load_le16(payload);
    const uint32_t indices = dna_load_le32(payload + 2);

    const float block_activation =
        superblock_activation * block_scale;

    float scaled_lut[4];
    for (size_t k = 0; k < 4; ++k) {
        scaled_lut[k] = block_activation * tables->magnitude_lut[k];
    }

    for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
        const uint32_t index = (indices >> (2 * i)) & 3u;
        const int negative = (signs >> i) & 1u;
        const float contribution =
            negative ? -scaled_lut[index] : scaled_lut[index];
        sum[i] += contribution;
    }
}

void dna_iq3_cxs2_sparse_down_tile(
    const dna_iq3_cxs2_active_neuron *active,
    size_t                           active_count,
    size_t                           tile_number,
    size_t                           tile_cache_lines,
    const dna_bf16_t                *c0,
    const dna_iq3_cxs2_tables       *tables,
    float                           *sum_scratch,
    dna_bf16_t                     *tile_sum)
{
    assert(tile_cache_lines > 0);

    const size_t tile_weights =
        tile_cache_lines * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t tile_offset = tile_number * tile_weights;
    const size_t first_block = tile_number * tile_cache_lines;

    for (size_t i = 0; i < tile_weights; ++i) {
        sum_scratch[i] = dna_bf16_to_f32(c0[tile_offset + i]);
    }

    for (size_t n = 0; n < active_count; ++n) {
        const float activation = dna_bf16_to_f32(active[n].activation);
        if (activation == 0.0f) {
            continue;
        }

        for (size_t line = 0; line < tile_cache_lines; ++line) {
            const size_t block_index = first_block + line;
            const dna_iq3_cxs2_block *block = active[n].blocks + block_index;
            const float superblock_activation =
                activation * active[n].superblock_scales[block_index];
            const uint32_t selectors = dna_load_le32(block->bytes + 60);

            for (size_t subblock = 0;
                 subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE;
                 ++subblock) {
                const uint32_t scale_index =
                    (selectors >> (3 * subblock)) & 7u;
                const size_t output_offset =
                    line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    subblock * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                dna_iq3_cxs2_accumulate_subblock(
                    block,
                    subblock,
                    superblock_activation,
                    tables->block_scale_lut[scale_index],
                    tables,
                    sum_scratch + output_offset);
            }
        }
    }

    for (size_t i = 0; i < tile_weights; ++i) {
        tile_sum[i] = dna_f32_to_bf16(sum_scratch[i]);
    }
}
```

The caller supplies `sum_scratch` with `tile_cache_lines * 160` F32 elements.
Each active-neuron entry points to its neuron-major sequence of 64-byte
IQ3_CXS2 lines, and its `superblock_scales` array uses the same block indexing.
The output is tile-local; `c0` is indexed by the tile's global output offset.
This reference code assumes valid pointers and dimensions and omits the
E5M3-to-F32 conversion, which should use the format's chosen E5M3 convention.

#### Scalar IQ3_CXS2 indirect-row GEMV (MR=4)

The same IQ3_CXS2 weight lines can be used for sparse gate/up rows without
repacking the rows together. This reference takes an activation vector already
quantized once to Q8_0: `x_q8` holds signed bytes and `x_scales` holds the
corresponding 32-element block scales converted to F32. `n_inputs` must be a
multiple of 160; each 64-byte IQ3_CXS2 line then covers 160 input elements,
which is five Q8_0 blocks.

For each 16-weight subblock, the 16 Q8 values are multiplied by the four
positive IQ3 magnitudes `{1, 3, 9, 27}` once to form a shared activation LUT.
Each of the four selected rows uses its own indices, signs, scale selector,
and superblock scale to consume that LUT. Since each 16-element subblock is
aligned within one 32-element Q8_0 block, it has one shared Q8 scale.

```c
typedef struct dna_iq3_cxs2_sparse_row {
    size_t output_row;
    const dna_iq3_cxs2_block *blocks;
    const float *superblock_scales;
} dna_iq3_cxs2_sparse_row;

void dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
    const int8_t                    *x_q8,
    const float                     *x_scales,
    size_t                           n_inputs,
    const dna_iq3_cxs2_sparse_row   *rows,
    size_t                           active_count,
    const float                      block_scale_lut[8],
    float                           *output)
{
    static const int32_t magnitude_lut[4] = { 1, 3, 9, 27 };
    assert(n_inputs % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);

    const size_t superblock_count =
        n_inputs / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    for (size_t group_start = 0;
         group_start < active_count;
         group_start += 4) {
        const size_t row_count =
            active_count - group_start < 4 ? active_count - group_start : 4;
        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t superblock = 0;
             superblock < superblock_count;
             ++superblock) {
            const dna_iq3_cxs2_block *weight_blocks[4];
            uint32_t selectors[4];
            float row_scales[4];

            for (size_t r = 0; r < row_count; ++r) {
                weight_blocks[r] = rows[group_start + r].blocks + superblock;
                selectors[r] =
                    dna_load_le32(weight_blocks[r]->bytes + 60);
                row_scales[r] =
                    rows[group_start + r].superblock_scales[superblock];
            }

            for (size_t subblock = 0;
                 subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE;
                 ++subblock) {
                const size_t input_offset =
                    superblock * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    subblock * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                const size_t q8_block = input_offset / 32;
                const size_t q8_offset = input_offset % 32;
                assert(q8_offset == 0 || q8_offset == 16);
                const int8_t *activation_block =
                    x_q8 + q8_block * 32 + q8_offset;

                int32_t activation_lut[16][4];
                for (size_t i = 0; i < 16; ++i) {
                    const int32_t q = activation_block[i];
                    for (size_t k = 0; k < 4; ++k) {
                        activation_lut[i][k] = q * magnitude_lut[k];
                    }
                }

                const float activation_scale = x_scales[q8_block];

                for (size_t r = 0; r < row_count; ++r) {
                    const dna_iq3_cxs2_block *block = weight_blocks[r];
                    const uint8_t *payload = block->bytes + subblock * 6;
                    const uint16_t signs = dna_load_le16(payload);
                    const uint32_t indices = dna_load_le32(payload + 2);
                    const uint32_t scale_index =
                        (selectors[r] >> (3 * subblock)) & 7u;

                    int32_t dot = 0;
                    for (size_t i = 0; i < 16; ++i) {
                        const uint32_t index = (indices >> (2 * i)) & 3u;
                        const int32_t term = activation_lut[i][index];
                        dot += ((signs >> i) & 1u) ? -term : term;
                    }

                    const float scale =
                        activation_scale *
                        row_scales[r] *
                        block_scale_lut[scale_index];
                    accum[r] += (float) dot * scale;
                }
            }
        }

        for (size_t r = 0; r < row_count; ++r) {
            output[rows[group_start + r].output_row] = accum[r];
        }
    }
}
```

`rows[]` is the active-row pointer list; its entries may point to arbitrary
matrix rows. `x_q8` and `x_scales` are shared across the four rows, as is each
16-by-4 activation LUT. Weight payloads and selectors remain row-specific.
The output is scattered by `output_row`; any gate/up bias or activation
function is applied by the caller. This is scalar reference code only: the
activation quantization, E5M3 scale conversion, and production dispatch are
outside the function.

#### Scalar IQ3_CXS2 quantization reference

This section describes the complete quantization pipeline for IQ3_CXS2, including:
1. Finding the MSE-optimal continuous scale for each 16-weight subblock via coordinate descent.
2. Clustering subblock scale demands into the 8 per-neuron scales using weighted 1D Lloyd-Max (k-means).
3. Quantizing the 8 scales to 8-bit unsigned E5M3 (5 exponent bits, 3 mantissa bits, bias 15, no sign).
4. Direct subblock encoding (selecting the best 3-bit selector, 2-bit magnitude indices, and 1-bit signs).

There is **no separate superblock scale**: each neuron (row) stores exactly 8 E5M3 scale values (8 bytes total), and each 16-weight subblock selects directly from these 8 values via its 3-bit selector.

```text
w_hat[i] = block_scale_lut[selector] * magnitude[index[i]] * sign[i]
```
where `magnitude in {1, 3, 9, 27}` and `sign in {+1, -1}`.

##### E5M3 Floating-point encoding

The 8 scales are stored in 8 bytes as unsigned E5M3 (5 exponent bits, 3 mantissa bits, bias = 15).

```c
#include <stdint.h>
#include <string.h>
#include <math.h>

// Convert float to unsigned E5M3 (bits: eeeeemmm)
static inline uint8_t dna_f32_to_e5m3(float val) {
    if (val <= 0.0f) {
        return 0;
    }
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15; // Rebinned to bias 15
    if (exp <= 0) {
        // Subnormal or underflow
        return 0;
    }
    if (exp >= 31) {
        // Clamp to max representable non-infinite value (exp=31, mantissa=7)
        return 0xFF;
    }
    uint32_t mant = (bits >> 20) & 0x7; // Top 3 mantissa bits
    // Round to nearest: check bit 19
    if ((bits >> 19) & 1) {
        mant++;
        if (mant > 7) {
            mant = 0;
            exp++;
            if (exp >= 31) return 0xFF;
        }
    }
    return (uint8_t) ((exp << 3) | mant);
}

// Convert unsigned E5M3 to float
static inline float dna_e5m3_to_f32(uint8_t val) {
    if (val == 0) {
        return 0.0f;
    }
    uint32_t exp = (val >> 3) & 0x1F;
    uint32_t mant = val & 0x7;
    uint32_t f32_exp = exp - 15 + 127;
    uint32_t f32_mant = mant << 20;
    uint32_t bits = (f32_exp << 23) | f32_mant;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}
```

##### 1. MSE-Optimal Continuous Scale for a 16-Weight Subblock

For a fixed assignment of weights to magnitude codebook entries $k_i \in \{1, 3, 9, 27\}$, the error $\sum (|w_i| - s \cdot k_i)^2$ is quadratic in $s$. Its exact minimizer is:
$$s = \frac{\sum |w_i| \cdot k_i}{\sum k_i^2}$$
Alternating between nearest-neighbor assignment and scale updating converges monotonically in 2–3 iterations.

```c
static const int32_t dna_iq3_cxs2_magnitudes[4] = { 1, 3, 9, 27 };

static float dna_iq3_cxs2_subblock_mse_scale(const float weights[16]) {
    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float a = fabsf(weights[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        return 0.0f;
    }

    // Initial scale estimate
    float s = max_abs / 27.0f;

    // 3 iterations of coordinate descent
    for (int iter = 0; iter < 3; ++iter) {
        if (s <= 0.0f) break;
        float inv_s = 1.0f / s;
        float sum_wk = 0.0f;
        float sum_k2 = 0.0f;

        for (int i = 0; i < 16; ++i) {
            float abs_w = fabsf(weights[i]);
            float target = abs_w * inv_s;

            // Nearest magnitude in {1, 3, 9, 27}
            int best_k = 1;
            if (target > 18.0f) {
                best_k = 27;
            } else if (target > 6.0f) {
                best_k = 9;
            } else if (target > 2.0f) {
                best_k = 3;
            }

            sum_wk += abs_w * (float) best_k;
            sum_k2 += (float) (best_k * best_k);
        }

        if (sum_k2 > 0.0f) {
            s = sum_wk / sum_k2;
        }
    }
    return s;
}
```

##### 2. 1D Lloyd-Max (k-means) Clustering into 8 Neuron Scales

Given all $M = N/16$ subblock ideal scales, weighted by subblock energy, Lloyd-Max groups them into 8 centroids, which are then quantized to E5M3:

```c
#include <stdlib.h>

static void dna_iq3_cxs2_compute_neuron_scales(
    const float *weights,
    size_t       n_weights,
    uint8_t      out_scales_e5m3[8],
    float        out_scales_f32[8])
{
    const size_t n_subblocks = n_weights / 16;
    float *sub_scales = (float *) malloc(n_subblocks * sizeof(float));
    float *sub_energy = (float *) malloc(n_subblocks * sizeof(float));

    float max_scale = 0.0f;
    float min_scale = 1e30f;
    size_t non_zero_count = 0;

    for (size_t b = 0; b < n_subblocks; ++b) {
        const float *w = weights + b * 16;
        float energy = 0.0f;
        for (int i = 0; i < 16; ++i) energy += w[i] * w[i];
        sub_energy[b] = energy + 1e-8f;

        float s = dna_iq3_cxs2_subblock_mse_scale(w);
        sub_scales[b] = s;
        if (s > 0.0f) {
            if (s > max_scale) max_scale = s;
            if (s < min_scale) min_scale = s;
            non_zero_count++;
        }
    }

    if (non_zero_count == 0) {
        // Completely silent neuron
        for (int k = 0; k < 8; ++k) {
            out_scales_e5m3[k] = 0;
            out_scales_f32[k] = 0.0f;
        }
        free(sub_scales);
        free(sub_energy);
        return;
    }

    // Initialize 8 centroids geometrically across [min_scale, max_scale]
    float centroids[8];
    float ratio = (max_scale > min_scale) ? (max_scale / min_scale) : 1.0f;
    for (int k = 0; k < 8; ++k) {
        centroids[k] = min_scale * powf(ratio, (float) k / 7.0f);
    }

    // 8 Lloyd-Max iterations
    for (int iter = 0; iter < 8; ++iter) {
        float cluster_weighted_sum[8] = { 0 };
        float cluster_weights[8] = { 0 };

        for (size_t b = 0; b < n_subblocks; ++b) {
            float s = sub_scales[b];
            if (s <= 0.0f) continue;

            // Find closest centroid
            int best_c = 0;
            float best_dist = fabsf(s - centroids[0]);
            for (int k = 1; k < 8; ++k) {
                float dist = fabsf(s - centroids[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = k;
                }
            }

            cluster_weighted_sum[best_c] += s * sub_energy[b];
            cluster_weights[best_c] += sub_energy[b];
        }

        // Update centroids
        for (int k = 0; k < 8; ++k) {
            if (cluster_weights[k] > 0.0f) {
                centroids[k] = cluster_weighted_sum[k] / cluster_weights[k];
            }
        }
    }

    // Sort centroids and quantize to E5M3
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            if (centroids[j] < centroids[i]) {
                float t = centroids[i];
                centroids[i] = centroids[j];
                centroids[j] = t;
            }
        }
    }

    for (int k = 0; k < 8; ++k) {
        out_scales_e5m3[k] = dna_f32_to_e5m3(centroids[k]);
        out_scales_f32[k] = dna_e5m3_to_f32(out_scales_e5m3[k]);
    }

    free(sub_scales);
    free(sub_energy);
}
```

##### 3. Complete Quantization Function

```c
static inline void dna_store_le16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
}

static inline void dna_store_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

static float dna_iq3_cxs2_quantize_subblock_exact(
    const float  weights[16],
    float        scale,
    uint16_t    *out_signs,
    uint32_t    *out_indices)
{
    if (scale <= 0.0f) {
        *out_signs = 0;
        *out_indices = 0;
        float err = 0.0f;
        for (int i = 0; i < 16; ++i) err += weights[i] * weights[i];
        return err;
    }

    const float inv_scale = 1.0f / scale;
    uint16_t signs = 0;
    uint32_t indices = 0;
    float total_err = 0.0f;

    for (size_t i = 0; i < 16; ++i) {
        const float w = weights[i];
        const int sign = w < 0.0f ? 1 : 0;
        const float abs_w = sign ? -w : w;
        const float target = abs_w * inv_scale;

        int best_k_idx = 0;
        if (target > 18.0f) {
            best_k_idx = 3; // 27
        } else if (target > 6.0f) {
            best_k_idx = 2; // 9
        } else if (target > 2.0f) {
            best_k_idx = 1; // 3
        } else {
            best_k_idx = 0; // 1
        }

        signs |= (uint16_t) sign << i;
        indices |= (uint32_t) best_k_idx << (2 * i);
        const float reconstructed = scale * (float) dna_iq3_cxs2_magnitudes[best_k_idx];
        const float diff = abs_w - reconstructed;
        total_err += diff * diff;
    }

    *out_signs = signs;
    *out_indices = indices;
    return total_err;
}

void dna_iq3_cxs2_quantize_row(
    const float        *weights,
    size_t              n_weights,
    dna_iq3_cxs2_block *out_blocks,
    uint8_t             out_scales_e5m3[8])
{
    assert(n_weights % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);
    const size_t n_lines = n_weights / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    // Step 1 & 2: Derive the 8 neuron scales
    float scales_f32[8];
    dna_iq3_cxs2_compute_neuron_scales(weights, n_weights, out_scales_e5m3, scales_f32);

    // Step 3: Quantize each 160-weight block
    for (size_t l = 0; l < n_lines; ++l) {
        const float *w_line = weights + l * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
        dna_iq3_cxs2_block *block = &out_blocks[l];
        memset(block->bytes, 0, DNA_IQ3_CXS2_LINE_BYTES);

        uint32_t selectors = 0;

        for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
            const float *w_sub = w_line + sub * 16;

            // Find best scale among the 8 neuron scales
            float best_err = 1e30f;
            uint32_t best_selector = 0;
            uint16_t best_signs = 0;
            uint32_t best_indices = 0;

            for (uint32_t s = 0; s < 8; ++s) {
                uint16_t signs;
                uint32_t indices;
                float err = dna_iq3_cxs2_quantize_subblock_exact(
                    w_sub, scales_f32[s], &signs, &indices);
                if (err < best_err) {
                    best_err = err;
                    best_selector = s;
                    best_signs = signs;
                    best_indices = indices;
                }
            }

            // Pack 6-byte payload
            uint8_t *payload = block->bytes + sub * 6;
            dna_store_le16(payload, best_signs);
            dna_store_le32(payload + 2, best_indices);
            selectors |= (best_selector << (3 * sub));
        }

        // Pack 10 3-bit selectors into bytes 60..63
        dna_store_le32(block->bytes + 60, selectors);
    }
}
```

##### 4. Compilation and Usage

The implementation is located in [`src/dna-iq3-cxs2.h`](src/dna-iq3-cxs2.h) and [`src/dna-iq3-cxs2.c`](src/dna-iq3-cxs2.c).

###### Compile and Run Unit Tests
To compile and run the self-contained round-trip test (verifying E5M3 conversions, SNR, AVX2 bit equivalence, and silence handling):
```bash
gcc -O3 -mavx2 -mfma -I. src/dna-iq3-cxs2.c tests/test-dna-iq3.c -o tests/test-dna-iq3 -lm
./tests/test-dna-iq3
```

###### Compile and Run Quality Benchmark (vs. IQ3_XXS and IQ4_XS)
To compare against existing GGML quantization types (`IQ3_XXS` and `IQ4_XS`) on synthetic MLP weight distributions:
```bash
g++ -O3 -mavx2 -mfma -I. -Iggml/include -Iggml/src \
    tests/test-dna-compare-quants.cpp src/dna-iq3-cxs2.c \
    build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a build/ggml/src/libggml.a \
    -lm -lpthread -fopenmp -o tests/test-dna-compare-quants

./tests/test-dna-compare-quants
```

###### Usage Example in C
```c
#include "src/dna-iq3-cxs2.h"

// Weights array of a neuron (e.g. 11008 weights for LLaMA FFN up/gate row)
const size_t n_weights = 160 * 68; // must be a multiple of 160
const float * neuron_weights = ...;

// Output buffers:
// - blocks: 1 block per 160 weights (64 bytes each, cache-line aligned)
// - scales: 8 bytes total per neuron (E5M3)
dna_iq3_cxs2_block blocks[68];
uint8_t scales_e5m3[8];

// Quantize (uses AVX2 if compiled with -mavx2, or call dna_iq3_cxs2_quantize_row for scalar)
#if defined(__AVX2__)
dna_iq3_cxs2_quantize_row_avx2(neuron_weights, n_weights, blocks, scales_e5m3);
#else
dna_iq3_cxs2_quantize_row(neuron_weights, n_weights, blocks, scales_e5m3);
#endif

// Convert the 8 E5M3 scale bytes to float for inference / GEMV:
float scales_f32[8];
for (int k = 0; k < 8; ++k) {
    scales_f32[k] = dna_e5m3_to_f32(scales_e5m3[k]);
}
```

#### AVX2 implementations for IQ3_CXS2

Below are AVX2 SIMD implementations for both the indirect-row GEMV (MR=4) and the transposed sparse down-tile path. Where AVX-512 features (specifically AVX-512F, AVX-512BW, and AVX-512_VNNI) provide significant performance benefits, those are detailed in the notes.

##### 1. AVX2 indirect-row GEMV (MR=4)

In this kernel, 4 selected rows share the same activation vector quantized into Q8_0 blocks. For each 16-weight subblock:
- The 16 Q8 values are multiplied by `{1, 3, 9, 27}` in 16-bit integer SIMD to form an activation table in vector registers.
- For each row, the 16 indices are unpacked, used to gather or permute terms, signs are applied, and terms are accumulated into `int32` vector accumulators.
- Deferred FP scaling (`accum * activation_scale * row_scale * block_scale`) is performed once per subblock per row.

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

static inline __m128i dna_unpack_2bit_indices_128(uint32_t packed) {
    const __m128i shifts_lo = _mm_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14);
    __m128i packed_v = _mm_set1_epi16((int16_t) packed);
    __m128i idx_lo = _mm_srlv_epi16(packed_v, shifts_lo);
    idx_lo = _mm_and_si128(idx_lo, _mm_set1_epi16(3));

    const __m128i shifts_hi = _mm_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14);
    packed_v = _mm_set1_epi16((int16_t) (packed >> 16));
    __m128i idx_hi = _mm_srlv_epi16(packed_v, shifts_hi);
    idx_hi = _mm_and_si128(idx_hi, _mm_set1_epi16(3));

    return _mm_packus_epi16(idx_lo, idx_hi); // 16 uint8 indices (values 0..3)
}

static inline int32_t dna_hsum_epi32_avx(__m128i v) {
    __m128i h = _mm_hadd_epi32(v, v);
    h = _mm_hadd_epi32(h, h);
    return _mm_cvtsi128_si32(h);
}

void dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx2(
    const int8_t                    *x_q8,
    const float                     *x_scales,
    size_t                           n_inputs,
    const dna_iq3_cxs2_sparse_row   *rows,
    size_t                           active_count,
    const float                      block_scale_lut[8],
    float                           *output)
{
    assert(n_inputs % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);
    const size_t superblock_count = n_inputs / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    for (size_t group_start = 0; group_start < active_count; group_start += 4) {
        const size_t row_count = active_count - group_start < 4 ? active_count - group_start : 4;
        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t superblock = 0; superblock < superblock_count; ++superblock) {
            const dna_iq3_cxs2_block *weight_blocks[4];
            uint32_t selectors[4];
            float row_scales[4];

            for (size_t r = 0; r < row_count; ++r) {
                weight_blocks[r] = rows[group_start + r].blocks + superblock;
                selectors[r] = dna_load_le32(weight_blocks[r]->bytes + 60);
                row_scales[r] = rows[group_start + r].superblock_scales[superblock];
            }

            for (size_t subblock = 0; subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++subblock) {
                const size_t input_offset =
                    superblock * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    subblock * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                const size_t q8_block = input_offset / 32;
                const size_t q8_offset = input_offset % 32;
                const int8_t *activation_block = x_q8 + q8_block * 32 + q8_offset;

                // Load 16 activation bytes (Q8_0) and sign-extend to 16-bit words
                __m128i q_raw = _mm_loadu_si128((const __m128i *) activation_block);
                __m128i q_lo = _mm_srai_epi16(_mm_unpacklo_epi8(q_raw, q_raw), 8);
                __m128i q_hi = _mm_srai_epi16(_mm_unpackhi_epi8(q_raw, q_raw), 8);

                // Compute activation terms for magnitudes {1, 3, 9, 27} in int16
                // mag 1: q
                // mag 3: q * 3
                // mag 9: q * 9
                // mag 27: q * 27
                __m128i t1_lo = q_lo;
                __m128i t3_lo = _mm_mullo_epi16(q_lo, _mm_set1_epi16(3));
                __m128i t9_lo = _mm_mullo_epi16(q_lo, _mm_set1_epi16(9));
                __m128i t27_lo = _mm_mullo_epi16(q_lo, _mm_set1_epi16(27));

                __m128i t1_hi = q_hi;
                __m128i t3_hi = _mm_mullo_epi16(q_hi, _mm_set1_epi16(3));
                __m128i t9_hi = _mm_mullo_epi16(q_hi, _mm_set1_epi16(9));
                __m128i t27_hi = _mm_mullo_epi16(q_hi, _mm_set1_epi16(27));

                const float activation_scale = x_scales[q8_block];

                for (size_t r = 0; r < row_count; ++r) {
                    const dna_iq3_cxs2_block *block = weight_blocks[r];
                    const uint8_t *payload = block->bytes + subblock * 6;
                    const uint16_t signs = dna_load_le16(payload);
                    const uint32_t indices = dna_load_le32(payload + 2);
                    const uint32_t scale_index = (selectors[r] >> (3 * subblock)) & 7u;

                    // Unpack indices: 16 uint8 values in 0..3
                    __m128i idx = dna_unpack_2bit_indices_128(indices);
                    uint8_t idx_arr[16];
                    _mm_storeu_si128((__m128i *) idx_arr, idx);

                    // Reconstruct terms for lower 8 and upper 8 elements
                    int16_t terms[16];
                    int16_t t_lut[4];

                    // Lower 8 elements
                    int16_t arr1[8], arr3[8], arr9[8], arr27[8];
                    _mm_storeu_si128((__m128i *) arr1, t1_lo);
                    _mm_storeu_si128((__m128i *) arr3, t3_lo);
                    _mm_storeu_si128((__m128i *) arr9, t9_lo);
                    _mm_storeu_si128((__m128i *) arr27, t27_lo);
                    for (int i = 0; i < 8; ++i) {
                        t_lut[0] = arr1[i];
                        t_lut[1] = arr3[i];
                        t_lut[2] = arr9[i];
                        t_lut[3] = arr27[i];
                        terms[i] = t_lut[idx_arr[i]];
                    }

                    // Upper 8 elements
                    _mm_storeu_si128((__m128i *) arr1, t1_hi);
                    _mm_storeu_si128((__m128i *) arr3, t3_hi);
                    _mm_storeu_si128((__m128i *) arr9, t9_hi);
                    _mm_storeu_si128((__m128i *) arr27, t27_hi);
                    for (int i = 0; i < 8; ++i) {
                        t_lut[0] = arr1[i];
                        t_lut[1] = arr3[i];
                        t_lut[2] = arr9[i];
                        t_lut[3] = arr27[i];
                        terms[i + 8] = t_lut[idx_arr[i + 8]];
                    }

                    // Load reconstructed 16 terms
                    __m128i v_terms_lo = _mm_loadu_si128((const __m128i *) terms);
                    __m128i v_terms_hi = _mm_loadu_si128((const __m128i *) (terms + 8));

                    // Build sign multipliers (+1 or -1 in 16-bit)
                    int16_t sign_arr[16];
                    for (int i = 0; i < 16; ++i) {
                        sign_arr[i] = ((signs >> i) & 1) ? -1 : 1;
                    }
                    __m128i s_lo = _mm_loadu_si128((const __m128i *) sign_arr);
                    __m128i s_hi = _mm_loadu_si128((const __m128i *) (sign_arr + 8));

                    // Multiply and horizontally add to int32: madd_epi16
                    __m128i dot_lo = _mm_madd_epi16(v_terms_lo, s_lo);
                    __m128i dot_hi = _mm_madd_epi16(v_terms_hi, s_hi);
                    __m128i dot_sum = _mm_add_epi32(dot_lo, dot_hi);
                    int32_t dot = dna_hsum_epi32_avx(dot_sum);

                    const float scale =
                        activation_scale * row_scales[r] * block_scale_lut[scale_index];
                    accum[r] += (float) dot * scale;
                }
            }
        }

        for (size_t r = 0; r < row_count; ++r) {
            output[rows[group_start + r].output_row] = accum[r];
        }
    }
}
```

##### 2. Complete AVX2 transposed sparse down tile

This kernel evaluates active neurons across an output tile (multiple of 160 weights = 1 or more cache lines). For each active neuron and each 16-weight subblock:
- `VPERMPS` (`_mm256_permutevar8x32_ps`) dynamically indexes the 4 scaled magnitudes duplicated across 8 lanes.
- Sign bits are applied via XOR on sign bit 31 (`VPXOR`).
- FMA (`_mm256_fmadd_ps` / `_mm256_add_ps`) updates output scratch sums directly in YMM registers.
- At tile completion, F32 sums are converted to BF16 and written out.

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

static inline __m256i dna_unpack_2bit_indices_256(uint32_t packed, int first_shift) {
    const __m256i shifts = _mm256_setr_epi32(
        first_shift + 0,  first_shift + 2,  first_shift + 4,  first_shift + 6,
        first_shift + 8,  first_shift + 10, first_shift + 12, first_shift + 14);
    const __m256i bits = _mm256_srlv_epi32(_mm256_set1_epi32((int) packed), shifts);
    return _mm256_and_si256(bits, _mm256_set1_epi32(3));
}

static inline __m256i dna_unpack_signs_256(uint16_t packed, int first_shift) {
    const __m256i shifts = _mm256_setr_epi32(
        first_shift + 0, first_shift + 1, first_shift + 2, first_shift + 3,
        first_shift + 4, first_shift + 5, first_shift + 6, first_shift + 7);
    const __m256i bits = _mm256_srlv_epi32(_mm256_set1_epi32((int) packed), shifts);
    return _mm256_and_si256(bits, _mm256_set1_epi32(1));
}

static inline __m256 dna_apply_sign_256(__m256 value, __m256i sign) {
    const __m256i sign_bit = _mm256_slli_epi32(sign, 31);
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(value), sign_bit));
}

void dna_iq3_cxs2_sparse_down_tile_avx2(
    const dna_iq3_cxs2_active_neuron *active,
    size_t                           active_count,
    size_t                           tile_number,
    size_t                           tile_cache_lines,
    const dna_bf16_t                *c0,
    const dna_iq3_cxs2_tables       *tables,
    float                           *sum_scratch,
    dna_bf16_t                      *tile_sum)
{
    assert(tile_cache_lines > 0);
    const size_t tile_weights = tile_cache_lines * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t tile_offset = tile_number * tile_weights;
    const size_t first_block = tile_number * tile_cache_lines;

    // Initialize scratch sums from c0
    for (size_t i = 0; i < tile_weights; ++i) {
        sum_scratch[i] = dna_bf16_to_f32(c0[tile_offset + i]);
    }

    // Duplicate magnitude LUT into both 128-bit halves: [m0, m1, m2, m3, m0, m1, m2, m3]
    const __m256 base_lut = _mm256_setr_ps(
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3],
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3]);

    for (size_t n = 0; n < active_count; ++n) {
        const float activation = dna_bf16_to_f32(active[n].activation);
        if (activation == 0.0f) {
            continue;
        }

        for (size_t line = 0; line < tile_cache_lines; ++line) {
            const size_t block_index = first_block + line;
            const dna_iq3_cxs2_block *block = active[n].blocks + block_index;
            const float superblock_activation =
                activation * active[n].superblock_scales[block_index];
            const uint32_t selectors = dna_load_le32(block->bytes + 60);

            for (size_t subblock = 0; subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++subblock) {
                const uint32_t scale_index = (selectors >> (3 * subblock)) & 7u;
                const float block_scale = tables->block_scale_lut[scale_index];
                const float factor = superblock_activation * block_scale;
                const __m256 block_lut = _mm256_mul_ps(base_lut, _mm256_set1_ps(factor));

                const uint8_t *payload = block->bytes + subblock * 6;
                const uint16_t signs = dna_load_le16(payload);
                const uint32_t indices = dna_load_le32(payload + 2);

                const __m256i idx0 = dna_unpack_2bit_indices_256(indices, 0);
                const __m256i idx1 = dna_unpack_2bit_indices_256(indices, 16);
                const __m256i s0   = dna_unpack_signs_256(signs, 0);
                const __m256i s1   = dna_unpack_signs_256(signs, 8);

                __m256 val0 = _mm256_permutevar8x32_ps(block_lut, idx0);
                __m256 val1 = _mm256_permutevar8x32_ps(block_lut, idx1);

                val0 = dna_apply_sign_256(val0, s0);
                val1 = dna_apply_sign_256(val1, s1);

                const size_t offset =
                    line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    subblock * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;

                __m256 sum0 = _mm256_loadu_ps(sum_scratch + offset);
                __m256 sum1 = _mm256_loadu_ps(sum_scratch + offset + 8);

                sum0 = _mm256_add_ps(sum0, val0);
                sum1 = _mm256_add_ps(sum1, val1);

                _mm256_storeu_ps(sum_scratch + offset, sum0);
                _mm256_storeu_ps(sum_scratch + offset + 8, sum1);
            }
        }
    }

    // Convert F32 results to BF16
    for (size_t i = 0; i < tile_weights; ++i) {
        tile_sum[i] = dna_f32_to_bf16(sum_scratch[i]);
    }
}
```

##### 3. AVX-512 implementations for modern CPUs

The following implementations exploit AVX-512F, AVX-512BW, AVX-512_VNNI, and AVX-512_BF16 for modern x86 processors (such as Intel Sapphire Rapids / Emerald Rapids / Granite Rapids and AMD Zen 4 / Zen 5).

Key architectural advantages utilized:
1. **AVX-512F 32x 512-bit registers (`zmm0`..`zmm31`)**: in sparse down, all 10 subblocks of a 160-weight cache line reside permanently in 10 `zmm` accumulator registers throughout the active-neuron loop, eliminating all memory traffic to scratch arrays during accumulation.
2. **AVX-512BW Opmasks (`__mmask16`)**: the 16-bit sign mask is transferred directly to an opmask register (`_cvtu32_mask16`) and signs are inverted with zero-latency mask blending or subtraction (`_mm512_mask_sub_ps`).
3. **Single-instruction full subblock lookup (`_mm512_permutexvar_ps`)**: all 16 weights are decoded simultaneously from a single 16-lane table in one cycle.
4. **AVX-512_VNNI (`_mm512_dpbusd_epi32` / `_mm256_dpbusd_epi32`)**: in indirect GEMV, dot products of 8-bit activations against quantized weight values are computed with hardware multiply-accumulate into 32-bit registers.
5. **AVX-512_BF16 (`_mm512_cvtneps_pbhf16`)**: final F32 accumulators are converted to BF16 in a single hardware instruction, replacing scalar bit-manipulation loops.

###### A. AVX-512 transposed sparse down tile

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

static inline __m512i dna_unpack_2bit_indices_512(uint32_t packed) {
    // Expand 16 2-bit indices from packed 32-bit word into 16 32-bit lanes
    const __m512i shifts = _mm512_setr_epi32(
        0,  2,  4,  6,  8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30);
    __m512i packed_v = _mm512_set1_epi32((int) packed);
    __m512i bits = _mm512_srlv_epi32(packed_v, shifts);
    return _mm512_and_epi32(bits, _mm512_set1_epi32(3));
}

void dna_iq3_cxs2_sparse_down_tile_avx512(
    const dna_iq3_cxs2_active_neuron *active,
    size_t                           active_count,
    size_t                           tile_number,
    size_t                           tile_cache_lines,
    const dna_bf16_t                *c0,
    const dna_iq3_cxs2_tables       *tables,
    dna_bf16_t                      *tile_sum)
{
    assert(tile_cache_lines > 0);
    const size_t tile_weights = tile_cache_lines * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    const size_t tile_offset = tile_number * tile_weights;
    const size_t first_block = tile_number * tile_cache_lines;

    // Duplicate 4-entry magnitude table across all 16 32-bit lanes of zmm:
    // [m0..m3, m0..m3, m0..m3, m0..m3]
    const __m512 base_lut = _mm512_setr_ps(
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3],
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3],
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3],
        tables->magnitude_lut[0], tables->magnitude_lut[1],
        tables->magnitude_lut[2], tables->magnitude_lut[3]);

    for (size_t line = 0; line < tile_cache_lines; ++line) {
        const size_t block_index = first_block + line;
        const size_t line_weight_offset = tile_offset + line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

        // Load c0 values into 10 ZMM registers (160 F32 values held in registers)
        __m512 acc[DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE];
        for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
            float temp_c0[16];
            for (size_t i = 0; i < 16; ++i) {
                temp_c0[i] = dna_bf16_to_f32(c0[line_weight_offset + sub * 16 + i]);
            }
            acc[sub] = _mm512_loadu_ps(temp_c0);
        }

        // Accumulate across active neurons with zero intermediate memory round-trips
        for (size_t n = 0; n < active_count; ++n) {
            const float activation = dna_bf16_to_f32(active[n].activation);
            if (activation == 0.0f) {
                continue;
            }

            const dna_iq3_cxs2_block *block = active[n].blocks + block_index;
            const float superblock_activation =
                activation * active[n].superblock_scales[block_index];
            const uint32_t selectors = dna_load_le32(block->bytes + 60);

            for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
                const uint32_t scale_index = (selectors >> (3 * sub)) & 7u;
                const float block_scale = tables->block_scale_lut[scale_index];
                const float factor = superblock_activation * block_scale;

                // Scale magnitude LUT for this subblock
                const __m512 block_lut = _mm512_mul_ps(base_lut, _mm512_set1_ps(factor));

                const uint8_t *payload = block->bytes + sub * 6;
                const uint16_t signs = dna_load_le16(payload);
                const uint32_t indices = dna_load_le32(payload + 2);

                // Unpack all 16 indices in one vector
                const __m512i idx = dna_unpack_2bit_indices_512(indices);

                // Decode all 16 weights in one instruction via AVX-512F permute
                __m512 val = _mm512_permutexvar_ps(idx, block_lut);

                // Apply signs via AVX-512BW opmask: negate where sign bit is 1
                const __mmask16 sign_mask = _cvtu32_mask16((uint32_t) signs);
                val = _mm512_mask_sub_ps(val, sign_mask, _mm512_setzero_ps(), val);

                // Accumulate directly into register
                acc[sub] = _mm512_add_ps(acc[sub], val);
            }
        }

        // Convert register accumulators to BF16 and write directly to output
        dna_bf16_t *out_line = tile_sum + line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
        for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
#if defined(__AVX512BF16__)
            // Native AVX-512_BF16 hardware conversion: 16 F32 -> 16 BF16 in 1 instruction
            __m256i bf16_vec = (__m256i) _mm512_cvtneps_pbhf16(acc[sub]);
            _mm256_storeu_si256((__m256i *) (out_line + sub * 16), bf16_vec);
#else
            float f32_vals[16];
            _mm512_storeu_ps(f32_vals, acc[sub]);
            for (size_t i = 0; i < 16; ++i) {
                out_line[sub * 16 + i] = dna_f32_to_bf16(f32_vals[i]);
            }
#endif
        }
    }
}
```

###### B. AVX-512 indirect-row GEMV (MR=4) with VNNI

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

void dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx512(
    const int8_t                    *x_q8,
    const float                     *x_scales,
    size_t                           n_inputs,
    const dna_iq3_cxs2_sparse_row   *rows,
    size_t                           active_count,
    const float                      block_scale_lut[8],
    float                           *output)
{
    assert(n_inputs % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);
    const size_t superblock_count = n_inputs / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    // Magnitude values: {1, 3, 9, 27}
    static const int8_t mag_table[4] = { 1, 3, 9, 27 };

    for (size_t group_start = 0; group_start < active_count; group_start += 4) {
        const size_t row_count = active_count - group_start < 4 ? active_count - group_start : 4;
        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t superblock = 0; superblock < superblock_count; ++superblock) {
            const dna_iq3_cxs2_block *weight_blocks[4];
            uint32_t selectors[4];
            float row_scales[4];

            for (size_t r = 0; r < row_count; ++r) {
                weight_blocks[r] = rows[group_start + r].blocks + superblock;
                selectors[r] = dna_load_le32(weight_blocks[r]->bytes + 60);
                row_scales[r] = rows[group_start + r].superblock_scales[superblock];
            }

            for (size_t subblock = 0; subblock < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++subblock) {
                const size_t input_offset =
                    superblock * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    subblock * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                const size_t q8_block = input_offset / 32;
                const size_t q8_offset = input_offset % 32;
                const int8_t *activation_block = x_q8 + q8_block * 32 + q8_offset;

                // Load 16 activation bytes
                __m128i act_16 = _mm_loadu_si128((const __m128i *) activation_block);
                const float activation_scale = x_scales[q8_block];

                for (size_t r = 0; r < row_count; ++r) {
                    const dna_iq3_cxs2_block *block = weight_blocks[r];
                    const uint8_t *payload = block->bytes + subblock * 6;
                    const uint16_t signs = dna_load_le16(payload);
                    const uint32_t indices = dna_load_le32(payload + 2);
                    const uint32_t scale_index = (selectors[r] >> (3 * subblock)) & 7u;

                    // Decode signed 8-bit weight values into 16 bytes
                    int8_t decoded_weights[16];
                    for (int i = 0; i < 16; ++i) {
                        const uint32_t idx = (indices >> (2 * i)) & 3u;
                        const int8_t mag = mag_table[idx];
                        decoded_weights[i] = ((signs >> i) & 1) ? (int8_t) -mag : mag;
                    }

                    __m128i w_16 = _mm_loadu_si128((const __m128i *) decoded_weights);

#if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
                    // AVX-512_VNNI direct int8 x int8 -> int32 dot product.
                    // _mm_dpbusd_epi32 expects unsigned 1st operand and signed 2nd operand:
                    // sum(u8 * s8). With activation x in [-128, 127], let u_act = (uint8_t)(x + 128).
                    // Then u_act * w = (x + 128) * w = x * w + 128 * w.
                    // Correcting: dot(x, w) = dot(u_act, w) - 128 * sum(w).
                    const __m128i offset_128 = _mm_set1_epi8(-128); // 0x80
                    __m128i u_act = _mm_xor_si128(act_16, offset_128); // x + 128 as uint8
                    __m128i vnni_acc = _mm_dpbusd_epi32(_mm_setzero_si128(), u_act, w_16);

                    // Subtract 128 * sum(w) to restore signed dot product
                    __m128i w_lo = _mm_srai_epi16(_mm_unpacklo_epi8(w_16, w_16), 8);
                    __m128i w_hi = _mm_srai_epi16(_mm_unpackhi_epi8(w_16, w_16), 8);
                    __m128i w_sum16 = _mm_add_epi16(w_lo, w_hi);
                    __m128i w_sum32 = _mm_madd_epi16(w_sum16, _mm_set1_epi16(1));
                    __m128i w_corr = _mm_slli_epi32(w_sum32, 7); // multiply sum(w) by 128
                    __m128i prod_sum = _mm_sub_epi32(vnni_acc, w_corr);
#else
                    // Fallback using madd_epi16 without VNNI
                    __m128i act_lo = _mm_srai_epi16(_mm_unpacklo_epi8(act_16, act_16), 8);
                    __m128i act_hi = _mm_srai_epi16(_mm_unpackhi_epi8(act_16, act_16), 8);
                    __m128i w_lo   = _mm_srai_epi16(_mm_unpacklo_epi8(w_16, w_16), 8);
                    __m128i w_hi   = _mm_srai_epi16(_mm_unpackhi_epi8(w_16, w_16), 8);

                    __m128i prod_lo = _mm_madd_epi16(act_lo, w_lo);
                    __m128i prod_hi = _mm_madd_epi16(act_hi, w_hi);
                    __m128i prod_sum = _mm_add_epi32(prod_lo, prod_hi);
#endif

                    // Horizontal reduction of 4 32-bit lanes
                    __m128i h = _mm_hadd_epi32(prod_sum, prod_sum);
                    h = _mm_hadd_epi32(h, h);
                    int32_t dot = _mm_cvtsi128_si32(h);

                    const float scale =
                        activation_scale * row_scales[r] * block_scale_lut[scale_index];
                    accum[r] += (float) dot * scale;
                }
            }
        }

        for (size_t r = 0; r < row_count; ++r) {
            output[rows[group_start + r].output_row] = accum[r];
        }
    }
}
```

#### Repacking memory layout in llama.cpp

llama.cpp CPU repacking is eager, load-time tensor repacking. It is performed
when a model tensor is assigned to a CPU extra buffer type named `CPU_REPACK`;
it does not happen inside the matrix-multiply loop.

The original GGML matrix layout is row-major in tensor dimensions:

```text
src[row][block_x] = original quantized weight row
tensor->ne[0]     = input/output columns
tensor->ne[1]     = weight rows
tensor->ne[2/3]   = optional matrix/batch dimensions
```

The repack entry point is selected by tensor traits in
`ggml_backend_cpu_repack_buffer_init_tensor()`. The selected implementation
copies the complete tensor through
`ggml_backend_cpu_repack_buffer_set_tensor()` and rewrites it into the
buffer-owning `tensor->data`.

Repacking interleaves a fixed group of weight rows, but the interleave unit is
usually much finer than a complete quantization block. For example, Q4_0 uses
32 weights per GGML block: 16 bytes of quantized nibbles plus a 2-byte scale
(18 bytes total). The 8-row `blck_size_interleave=8` layout does not move the
whole 32-weight block as one unit. It moves eight-row chunks of 8 bytes of
quantized nibbles, corresponding to 16 4-bit weights per row:

```text
original row r / input positions x:
    32 weights  -> 16 quantized bytes
    each 8-byte chunk  -> 16 weights
```

With 8 interleaved rows, the following layout is used:

```text
for 8-byte chunk c:
    8 bytes from row0/c
    8 bytes from row1/c
    ...
    8 bytes from row7/c
```

Those 64 contiguous bytes contain the quantized data for 16 weights from each
of the eight rows. The matching eight F16 scales (16 bytes total) are stored
before the quantized data in the repacked structure. Consequently, these
64-byte groups are not necessarily cache-line aligned: the first group starts
16 bytes after the structure start. The next 64-byte group advances to the
next 16-weight chunk for the same eight rows.

Thus, the 8-row interleaved layout for Q4_0 starts by separating all 32-weight
blocks into fine sub-blocks, and groups the corresponding sub-block of each
row. It is therefore not a layout of complete 256-weight superblocks.

More generally, the row-group memory layout can be pictured as:

```text
input sub-block x:
    metadata/data for row0, x
    metadata/data for row1, x
    ...
    metadata/data for rowN, x

input sub-block x+1:
    metadata/data for row0, x+1
    metadata/data for row1, x+1
    ...
    metadata/data for rowN, x+1
```

For Q4_0 `blck_size_interleave=8`, sub-block `x` is 16 weights per row. A
4-row layout uses the same Q4_0 merge function with either 4-byte or 8-byte
interleave chunks, depending on `INTER_SIZE`. The 8-byte variants cover 16
weights per row; a 4-byte variant groups 8 weights per row at finer
granularity.

Super-block types are structured differently. Q4_K uses 256-value superblocks
and stores scales/mins plus quantized bytes. Its 8-row repack implementation
uses `blck_size_interleave=4` or `8` and interleaves those super-block fields
in format-specific chunks. It does not simply concatenate eight complete
256-weight superblocks.

The exact scale and quantized-byte layout depends on the quantization type.
For Q4_0, `block_q4_0x8` stores eight F16 deltas followed by interleaved
16-byte Q4_0 quantized arrays split into two 8-byte chunks. For Q4_K and
other super-block types, `block_q4_Kx8`, `block_q4_Kx16`, and the other
`block_*xN` structures group per-row scales/mins and quantized data for a fixed
number of weight rows.

The generic Q4_0 loop is:

```text
for each group of 8 weight rows:
    for each input block x:
        dst[x] = merge(
            src[row_group+0][x],
            src[row_group+1][x],
            ...
            src[row_group+7][x])
```

The merge does not remain in original row-major order. Instead,
`make_block_q4_0x8()` separates scale metadata and quantized bytes in the
interleaved structure, and the quantized nibbles are arranged in the
`blck_size_interleave` chunks expected by the architecture-specific dot
product kernel.

The repack source strides are therefore:

```text
one original row:
    nblocks = ne0 / QK
    src[row][block_x] = src[block_x + row * nblocks]
```

The destination strides are:

```text
one interleaved group:
    one merged block for block_x 0
    one merged block for block_x 1
    ...
    one merged block for the complete input dimension
```

This layout is consumed as follows:

- The activation/input matrix is quantized to the matching parameter type in
  workspace, such as Q8_0 or Q8_K.
- GEMM/GEMV threads receive row-aligned chunks of the interleaved weight
  tensor.
- `NB_COLS` determines weight-row chunk alignment.
- `INTER_SIZE` determines how quantized sub-block bytes are interleaved.
- `NB_COLS` and `INTER_SIZE` are layout and kernel parameters, not a promise
  that the original tensor can be addressed row by row afterward.
- A 8-row interleaved tensor is only valid for kernels with a matching
  `tensor_traits` implementation.

The selected layout is architecture-specific. `ggml_repack_get_optimal_repack_type()`
chooses Q4_0 4x4, 4x8, or 8x8 layouts, Q4_K 8x4 or 8x8 layouts, and
corresponding layouts for Q2_K, Q5_K, Q6_K, IQ4_NL, MXFP4, Q8_0, and RISC-V
variants. The choice depends on available CPU instructions and whether the
weight-row dimension satisfies the required alignment.

The dense forward kernels pair the interleaved weight blocks with interleaved
activation blocks. The kernel's dot-product code computes several weight rows
at once for a GEMM, or one or more row planes for a GEMV, instead of
dereferencing each original GGML row separately.

Why this layout helps:

- The original row-major matrix stores `row0/blockX`, `row1/blockX`,
  `row2/blockX`, and so on at different addresses separated by the complete
  input dimension. An interleaved block stores the metadata and quantized bytes
  for `row0/blockX` through `rowN/blockX` close together.
- The kernel processes one activation block and several weight rows for that
  block at the same time. It can reuse one quantized activation block and one
  set of loaded scales in SIMD lanes or vector registers.
- Four or eight weight-row dot products can therefore share inner-loop control,
  activation loads, scale handling, and output accumulators.
- This is primarily beneficial for GEMM or batched GEMV, where the same
  activation plane is reused by several output rows. It is less obviously
  beneficial for a true single-row GEMV; a scalar or ordinary row-wise GEMV
  can lose much of the benefit and may be better without interleaving.
- It improves throughput by reducing indirect addressing, scattered loads,
  activation quantization repetition, and repeated kernel setup. It does not
  guarantee that every weight for one complete neuron is nearby in memory.
  - Sustained read throughput on CPU does not read a full 512b (64-byte) vector
    every cycle. Because x86 lacks native 4-bit dot products, 4-bit nibbles
    must be extracted (`vpand`, `vpsrlw`), sign-extended via shuffle tables
    (`vpshufb`), and accumulated (`vpdpbusd` or `vpmaddubsw`/`vpmaddwd`). In
    cached GEMV (`gemv_q4_b32_8x8_q8_0_lut_avx`), 144 bytes of weights and
    scales require ~60-70 instructions (~15-20 cycles), reading a 512b vector
    equivalent every ~7-10 cycles (~7-10 B/cycle). In batched GEMM, reusing
    288 bytes over multiple activation rows requires ~250-350 cycles (one 512b
    vector every ~60-70 cycles, or ~1 B/cycle). When streaming weights from
    DRAM during token generation, memory bus limits cap throughput further to
    ~1.5-3 B/cycle per core (one 512b vector every ~20-40 cycles).

For a dense matrix product, simply assigning rows to threads in parallel is
still valid and can work, but it is not equivalent to the interleaved layout:

- Each thread would load the activation block independently unless the
  activation cache line happens to be shared and already resident.
- Each thread would follow a different row stride to find the weight block.
- Each thread would accumulate only for its own output row.
- The kernel may need more scalar row-offset calculations and less efficient
  packed dot-product instructions.

The interleaved layout moves the row-grouped weight metadata and quantized
bytes into a contiguous block before computation. It is a dense
block-matrix/inner-product layout, not an allocation of one full weight row
or one cache line to a thread.

Because the destination layout is not the original GGML layout:

- `tensor->extra` holds the selected repack/tensor traits.
- `tensor->data` must not be interpreted using normal GGML row strides.
- `tensor->buffer` is `CPU_REPACK`, not ordinary `CPU` or `CPU_Mapped`.
- The original row-major tensor is no longer directly readable through
  `tensor->data`.
- Repacking requires full-tensor replacement: the current CPU repack
  `set_tensor` implementation asserts that `offset == 0` and
  `size == ggml_nbytes(tensor)`.
- The repacked allocation is host memory, but with a backend-specific layout.

No virtual sparse-down interpretation is needed for this section. It describes
only llama.cpp's existing dense CPU weight repacking mechanism.

## SIMD and repacking for sparse down

### Does the AVX2 sparse-down algorithm need repacking?

No cross-neuron repacking is required for the described SIMD algorithm. Its
lanes represent output positions from one active neuron's 16-weight block:
AVX2 expands the packed two-bit indices, selects magnitudes from the four-entry
LUT, applies the sign bits, and adds the contributions to the corresponding
F32 output sums. The scalar activation and block scales are reused to prepare
that block's LUT.

With the proposed per-neuron block pointers and tile-contiguous blocks, a tile
worker can visit each active neuron's block directly. The block's format must
provide the packed sign/index fields and scale metadata in a defined layout;
that is part of storing the new quantized weights, not a separate CPU row
repack done to make AVX2 possible. Cache-line-aligned block starts can help
tile traversal, but individual six-byte fields for 16 weights need not be
aligned: unaligned loads are valid. Loads must still stay within readable
memory, so a wider load must not overread a block or allocation boundary.

Processing four or eight cache lines together is likewise a scheduling and
unrolling choice, not a requirement to interleave rows. It means issuing and
working through several independent packed-weight regions; it does not mean
one cache line equals one SIMD register.

This differs from llama.cpp's existing dense CPU repacking. Dense repacking
interleaves sub-blocks from several weight rows so their dot products can reuse
the same activation block. In sparse down, each active neuron has its own
scalar activation, while SIMD lanes span that neuron's output positions.
Interleaving neurons in the dense-GEMM style therefore does not directly
provide the same activation-reuse benefit.

A separate tile-major layout that places the same output tile for multiple
neurons near one another could be benchmarked for cache locality when a worker
scans active neurons. It is optional and changes address calculation and
storage order; whether it helps depends on active-neuron ordering and cache
behavior. It should not be conflated with the within-block SIMD decode, and
should be measured before adding its complexity.

### Blocked dataflow for a 160-weight IQ3_CXS2 cache-line block

For the proposed IQ3_CXS2 layout, one 64-byte weight line represents 160
weights for one active neuron and one 160-output region. It contains ten
16-weight subblocks. Each subblock has 48 bits of sign/index data plus a 3-bit scale selector:
ten such records use 510 bits. The scalar reference below uses bytes 0..59
for the ten six-byte sign/index payloads and packs all ten selectors into the
last four bytes, reserving their top two bits. This is a reference layout;
the production format must also specify the E5M3 convention and scale-table
scope.

The worker owns one 160-output region. Its dataflow is:

```text
sum[160] = convert c0 for this output region from BF16 to F32

for each batch of up to 32 active neurons:
    for each neuron n in the batch:
        superblock_activation[n] =
            F32(activation[n]) * applicable_superblock_scale[n]
        weight_line[n] = pointer to this neuron's 64-byte line

    for subblock s = 0..9:             # 16 output weights at a time
        acc[16] = sum[16*s : 16*s+16]  # two AVX2 F32 vectors

        for each neuron n in the batch:
            read the 16-weight record and its 3-bit scale selector
            block_scale = lookup(selector)
            factor = superblock_activation[n] * block_scale
            block_lut = base_lut[4] * factor
            unpack 16 two-bit indices and 16 signs
            lookup magnitudes, apply signs, and add to acc[16]

        sum[16*s : 16*s+16] = acc[16]

convert sum[160] from F32 to BF16 and write the output region
```

`base_lut` is the four-entry F32 magnitude table, loaded once and duplicated
across the AVX2 lanes. The scale selector and sign/index fields are read from
the current neuron's line. The scale table and activation factor are reused
at their defined scope; `block_lut` is specific to the current neuron and
16-weight subblock. The 16 outputs in `acc` occupy two YMM registers; the
complete 160-element sum is a 640-byte F32 scratch tile, not a register tile.

For a batch of 32 neurons, the main live data is approximately:

| Data | Size |
|---|---:|
| 32 packed weight lines | 2,048 B (32 cache lines) |
| 32 F32 superblock activations | 128 B |
| 32 line pointers | 256 B |
| F32 partial sum for 160 outputs | 640 B (10 cache lines) |
| BF16 `c0` source during initialization | 320 B |
| Base LUT and scale metadata | small |

This is about 3.3 KiB before minor temporaries and output storage. The
640-byte partial sum remains in L1 and is updated 16 values at a time. The
2-KiB packed-weight batch is also small enough to fit by L1 capacity on common
CPUs. The subblock loop revisits each neuron's line ten times; after its first
use, those accesses are expected to hit in L1, provided the lines are not
evicted by set conflicts or other accesses. Capacity alone does not guarantee
residency, so the actual line addresses and cache counters should be checked.

The batch loop should enclose the subblock loop so the same 32 weight lines
are revisited while the batch is still hot. The inner neuron loop exposes
independent line addresses for load scheduling; it does not imply 32
simultaneous 48-bit loads or guarantee a particular loads-per-cycle rate.
For more than 32 active neurons, the next batch continues from the F32 partial
sum in L1. After all batches, the worker converts and writes its disjoint
BF16 output region; no cross-thread output reduction is required.

