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

- [ ] Future: use sparsity inside matrix-matrix multiplication as well.

References: [`llama_dna_params`](src/llama-dna.h), [`build_ffn`](src/llama-graph.cpp), and the batch-limit notes and recipes in [DNA.md](DNA.md).

### 2. Route work remains proportional to every neuron and token

For each token, `op_route` quantizes the input for the predictor and gate/up dot-product types, then estimates the gate value for every FFN neuron. It then skips the true gate/up dot products for neurons classified as silent.  
Active neurons still require separate row dot products for gate and up.

The loops are token-major and call row-vector kernels; they do not turn a token batch into a batched matrix multiply.

Thus, increasing sparsity avoids some gate/up reads and dot products, but does not remove the all-neuron predictor scan. At high active fractions, the remaining gate/up work may also leave little room to offset predictor overhead.

- [ ] Future: use sparse packing GEMV optimization
- [ ] Future: use sparsity inside the predictor as well.

Reference: [`op_route`](src/llama-dna.cpp).

### 3. Down projection uses per-neuron AXPY updates

`op_down` initializes each output row from `c0`, scans the full neuron delta for each token, and applies an AXPY for every nonzero delta using the transposed F16 down weights.

This avoids computing silent neurons, but does not use a batched matrix-multiply kernel. With larger token counts, the same active weight rows are revisited for each token. Whether this is slower than dense execution depends on sparsity, model dimensions, cache behavior, and CPU.

- [ ] Need faster down projection in the sparse regime.

Reference: [`op_down`](src/llama-dna.cpp).

### 4. Intermediate and persistent memory grow with model/batch size

- The route output (`delta`) contains `n_ff * n_tokens` F32 values, or `4 * n_ff * n_tokens` bytes per layer before allocator reuse. For example, `n_ff = 11008` and `n_tokens = 512` is about 21.5 MiB for one such tensor.
- After bias calibration, each layer retains `down_t`, an F16 copy with `n_ff * n_embd` elements, in addition to the model's original down weights.
- Each enabled layer retains its packed predictor gate (`pred_q`) and per-neuron calibration/prediction data. MoE layers hold corresponding state per expert.

The delta tensor is batch-dependent; the transposed weights and predictor are model/layer-dependent. Batching or removing the batch guard should therefore be measured for peak memory as well as speed.

- [ ] Remove calibration data after prediction starts, decided per layer.
- [ ] Schedule low-RAM calibration according to available memory: run as many layers in parallel as free RAM allows and start new layers when calibration ends.
- [ ] Replace `down_t` with an efficient sparse down transformation.

References: the `delta` allocation in [`build_ffn`](src/llama-graph.cpp), and `down_t`/`pred_q` storage in [`llama-dna.cpp`](src/llama-dna.cpp).

### 5. Setup and calibration delay the point at which skipping begins

When a layer is enabled, `llama_dna_pack_gate` reads/dequantizes its gate rows and quantizes them into the predictor format. This is a one-time per-layer setup cost and adds predictor storage.

- [ ] Precompute predictors and store them in the GGUF.

Calibration runs the ordinary dense FFN and adds observers for two phases, configured by `LLAMA_DNA_CALIB`. Prediction savings begin only after a layer completes both phases. For short prompts or large calibration settings, DNA may spend much or all of the workload calibrating. The Gemma results in [DNA.md](DNA.md) include a four-chunk, `n_calib=4096` run where no layer reached PREDICT.

- [ ] Save calibration results to a sidecar file.
- [ ] Reserve GGUF metadata fields and store calibration results in the GGUF.

Each run's recovered PPL for chunks 1–8 is recorded in [DNA.md](DNA.md). Those values are segment estimates derived from cumulative PPL, not independent evaluator runs for each chunk.

References: [`llama_dna_pack_gate`](src/llama-dna.cpp), [`op_observe`](src/llama-dna.cpp), and the calibration results in [DNA.md](DNA.md).

## Measurement gaps

The in-tree `LLAMA_DNA_VERBOSE` reporting includes per-layer observe, route, and down timings, but the documented sweeps do not provide matched dense-versus-DNA tokens/s, end-to-end latency, or peak RSS. They also mix different calibration lengths, chunking, and prediction coverage.

Consequently, the net speed benefit and batch-size crossover are not yet established.

Before changing the default guard or claiming a speedup, compare dense and DNA under the same model, CPU/thread count, context, and input with optimized codepaths for sparse execution.

- [ ] Separate load/setup time and calibration time from steady-state prediction.
- [ ] Measure prompt throughput and decode throughput/latency.
- [ ] Measure peak memory, including graph workspace.
- [ ] Record per-layer active fraction and route/down timings.
- [ ] Validate output quality/perplexity under identical calibration and prediction coverage.

## TODO

The open items below consolidate the TODOs described throughout this document.

### Quality and comparison

- [ ] Run matched dense-versus-DNA measurements for every predictor using the same model, corpus, chunking, calibration, threads, context, and batch settings.
- [ ] Report PPL over an isolated prediction-on subset using the same equal-sized chunk formula for every run.
- [ ] Add uncertainty intervals for recovered subset PPLs if the evaluator exposes enough per-chunk data.
- [ ] Establish a measured sparse/dense crossover for active fraction, batch size, and model dimensions.

### Operator implementation

- [ ] Replace sparse down AXPY work with an efficient sparse down transformation or batched sparse kernel.
- [ ] Use sparsity for gate/up GEMV optimization. Virtual packing.

### Memory and scheduling

- [ ] Remove per-layer calibration state immediately after prediction starts.
- [ ] Parallelize calibration layer scheduling according to available RAM.
- [ ] Measure peak RSS and graph workspace with the batch guard raised.
- [ ] Bound or tile the per-layer `delta` workspace for large batches.

### Storage and persistence

- [ ] Precompute DNA predictors and store them in GGUF.
- [ ] Save calibration results in a sidecar file.
- [ ] Reserve GGUF metadata fields and store calibration results directly in GGUF.

### Benchmarking

- [ ] Measure load/setup and calibration time separately from steady-state prediction.
- [ ] Measure decode throughput/latency.
- [ ] Measure per-layer active fraction and route/down timings.
- [ ] Change the default batch guard only after sparse/dense crossover evidence supports it.

### Future topics

- [ ] Use sparsity inside matrix-matrix multiplication operations.
- [ ] Use sparsity inside the predictor so inference does not scan every neuron.


## Implementation Plan

### sparse down projection

"Replace sparse down AXPY work with an efficient sparse down transformation or batched sparse kernel."

The current implementation is a sparse sequence of per-neuron AXPY operations:
for each token, it scans every neuron, and for each active neuron it adds
`delta[i] * Wdown[:, i]` to the output vector. Each neuron column is stored
contiguously in `down_t`, so this layout is already suitable for sparse
column-style updates.

A GGML-style block model is more suitable than assuming every result block
is storage-aligned:

- Existing IQ4_XS and IQ3_XXS types use 256-value weight blocks, but they are legacy storage formats and do not satisfy the sparse-down cache layout: IQ4_XS stores 256 weights in a 136-byte block and IQ3_XXS in a 98-byte block.
- The proposed sparse-down block type will instead fit exactly into `n * 64` bytes and contain a multiple of 16 weights.
 - example IQ3_CXS2:
	* 8 per matrix scale factors (E5M3), not per block, not counted here
	- 16w blocks a 3*16b + 3b scale factor select
	- 2*5*16w = 160w, 510bit (512bit / 160w = 3.2 bpw)
- A block may cover 16, 32, or more weights depending on its packed format, but its byte size will remain a multiple of a cache line.
- The 16-weight granularity provides SIMD-compatible subblocks. Larger blocks may contain multiple 16-weight subblocks and span multiple cache lines.
- The sparse transformation will use these blocks directly for cache-line and thread-sized work partitioning. The final block layout and metadata format still need to be designed.


For the sparse down projection, one useful model is:

- Split the output vector into a set of tiles. Tile width depends on the used quant type, which is cache line aligned. For IQ3_CXS2 it's 160 wide tiles.
- Split each tile into subblocks aligned to the quantized weight block layout and CPU SIMD width.
- Assign one tile to each thread.
- For a tile, scan only the active neurons.
- For each active neuron, scale and accumulate the weight subblocks of that
  neuron over the tile, including the quantization block scale/decode step.
- Write the completed tile once. No cross-thread full-result-vector reduction is needed. The tile sum vectors are not cache aligned, so when threads finish they may have to synchronize. But this will only happen at most ```numthread-1``` times per matrix.

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
back to BF16. It does not implement the IQ3_CXS decoder or matrix-scale
handling; those belong in `decode_subblock`.

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

