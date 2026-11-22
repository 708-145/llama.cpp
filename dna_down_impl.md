# Implementation Plan: Replacing `down_t` with IQ3_CXS2 Sparse Down Transformation

## 1. Problem Statement & Motivation

In current DNA execution:
- At the end of the calibration bias phase, each layer stores `down_t`, an uncompressed transposed copy of `Wdown` in FP16 format ($n_{\text{ff}} \times n_{\text{embd}} \times 2$ bytes). For a model with $n_{\text{ff}} = 14336$ and $n_{\text{embd}} = 4096$, this consumes **117 MB (112 MiB) per layer** of persistent memory.
- In `op_down`, every token evaluates a sparse loop of per-neuron AXPY operations (`llama_dna_axpy_f16`). Each active neuron traverses the thread's output slice, reading contiguous FP16 weights and accumulating.
- AXPY streams weight memory across active neurons without output caching, requiring FP16-to-F32 conversions and repeated memory updates to `y`.

## 2. The Solution: IQ3_CXS2 Tiled Sparse Down Transformation

Instead of storing FP16 weights and running AXPY:
1. **Quantize to IQ3_CXS2 at Calibration**:
   - Each neuron's down-weight row ($n_{\text{embd}}$ weights) is quantized into 64-byte `IQ3_CXS2` blocks (160 weights per block) and 8 bytes of unsigned E5M3 scale factors.
   - Persistent storage drops from 16 bpw to about **3.27 bpw**, including per-neuron scale bytes and the padded tail block. For the example dimensions this is **~22.9 MiB**, about a **4.9× reduction** from 112 MiB.
2. **Output Tile Partitioning in `op_down`**:
   - Output dimension $n_{\text{embd}}$ is partitioned into tiles of size $T = 160$ weights (matching the 64-byte block size).
   - Each thread owns a disjoint subset of tiles (embarrassingly parallel, no atomic operations or cross-thread reductions).
3. **In-Register / L1 Accumulation**:
   - For a given tile, each thread scans only the active neurons.
   - For each active neuron, its single 64-byte block corresponding to that tile is loaded.
   - The subblocks of 16 weights are decoded via AVX2 (`_mm256_permutevar8x32_ps`, `VPXOR`, `_mm256_fmadd_ps`) and accumulated directly into an F32 local scratch tile in L1/registers.
   - When all active neurons have contributed, the completed tile is written once to `y`.

---

## 3. Data Structures & Scale Storage

### 3.1 Storing the 8B Scales per Neuron
For each neuron $i \in [0, n_{\text{ff}}-1]$:
- There are exactly **8 scale factors** stored as 8 unsigned bytes in **E5M3** format (5 exponent bits, 3 mantissa bits, bias 15, no sign).
- Two layout options were evaluated:
  - *Option A (Interleaved)*: Storing 8 bytes alongside each line's block. This adds 8 bytes of padding/overhead per block.
  - *Option B (Dedicated contiguous table - SELECTED)*: A separate flat array `std::vector<uint8_t> down_scales` of size $n_{\text{ff}} \times 8$ bytes (total 112 KiB for $n_{\text{ff}}=14336$).
- The pointer to neuron $i$'s 8-scale table is simply:
  ```cpp
  const uint8_t * neuron_scales_e5m3 = L.down_scales.data() + i * 8;
  ```
- Because 8 bytes fit in a single 64-bit integer (`uint64_t`), loading or copying a neuron's entire scale table is a single 8-byte scalar load.

### 3.2 Neuron-Major Block Layout
- Each neuron covers $n_{\text{embd}}$ weights.
- Total 64-byte blocks per neuron:
  $$n_{\text{blocks\_per\_neuron}} = \lceil n_{\text{embd}} / 160 \rceil$$
- Total blocks per layer: $n_{\text{ff}} \times n_{\text{blocks\_per\_neuron}}$ `dna_iq3_cxs2_block` structs.
- Address of neuron $i$'s block for tile $t$:
  $$\text{block\_ptr}(i, t) = \text{down\_blocks} + i \times n_{\text{blocks\_per\_neuron}} + t$$

```cpp
struct llama_dna_layer {
    // Replaced down_t with IQ3_CXS2:
    std::vector<dna_iq3_cxs2_block> down_blocks; // [n_ff * n_blocks_per_neuron] (64 bytes each)
    std::vector<uint8_t>            down_scales; // [n_ff * 8] (E5M3 scales, 8 bytes per neuron)
    size_t                          down_blocks_per_neuron = 0;
    ...
};
```

The implementation builds a temporary FP16 transposed staging buffer while
reading the model's output-major down rows. It quantizes each staged neuron row
to IQ3_CXS2 and releases the staging buffer before prediction. This preserves
the former sparse path's FP16 input precision without retaining the
`n_ff * n_embd` FP16 copy. The one-time calibration peak therefore includes
both that staging buffer and the new packed storage. The same per-layer storage
and tile kernel are used for routed MoE experts; their down rows are read from
the selected expert slice.

---

## 4. Handling Dimension Misfit ($n_{\text{embd}} \pmod{160} \ne 0$)

Most model embedding dimensions ($n_{\text{embd}} = 4096, 5120, 8192$, etc.) are **not** multiples of 160 ($4096 = 25 \times 160 + 96$).

### 4.1 Subblock Granularity (16-Weight Units)
Every full block contains 10 subblocks of 16 weights ($10 \times 16 = 160$).
Because $n_{\text{embd}}$ is almost always a multiple of 64 or 128 (and thus a multiple of 16):
$$n_{\text{embd}} \pmod{16} == 0$$
Thus, the remainder $n_{\text{embd}} \pmod{160}$ is always an exact multiple of 16 subblocks. For example, $4096 \pmod{160} = 96 = 6 \times 16$.

### 4.2 Storage and Compute Rules
1. **Full 64-byte Storage**:
   The trailing tile block still allocates a full 64-byte cache line so memory accesses are always 64-byte aligned and never trigger segfaults. The unneeded subblocks in the last block are padded with zeros at quantization time.
2. **Dynamic Subblock Count Parameter (`n_subblocks`)**:
   Instead of hardcoding 10 subblocks (160 weights), the kernel accepts `n_subblocks` ($1 \le n_{\text{subblocks}} \le 10$):
   - For all full tiles: `n_subblocks = 10` (process all 10 subblocks).
   - For the trailing tile:
     $$\text{n\_subblocks} = \frac{n_{\text{embd}} - \text{tile\_offset}}{16}$$
   - Any subblocks beyond `n_subblocks` are **skipped completely**, saving computation time while preserving bit-exact results and full cache-line alignment.

---

## 5. Active Neuron List: Maintenance & Passing

### 5.1 The Current Bottleneck
In the current implementation:
- `op_route` writes dense float delta tensor: `delta[t, i] = d_i` or `0.0f`.
- `op_down` iterates through all $i \in [0, n_{\text{ff}}-1]$ and checks `if (d[i] != 0.0f)`.
- When sparsity is high (e.g. 90% silent), 90% of the loop iterations are branch checks over memory.

### 5.2 Dense-to-Sparse Active Index List
To accelerate tile traversal, we maintain a compact list of active neuron indices per token:
```cpp
struct dna_active_neuron {
    int32_t index;       // neuron index i in [0, n_ff)
    float   activation;  // delta[i] value
};
```

#### Where to Build the Active List:
There are two integration strategies:

- **Strategy A (Inside `op_down` per-token preprocessing - Zero Graph Changes)**:
  At the beginning of `op_down`, each worker scans `delta` once per token,
  before looping over its owned tiles, and extracts non-zero entries into its
  reusable thread-local scratch buffer:
  ```cpp
  // At typical 10% active neurons on n_ff=14336: ~1400 active entries
  // 1400 * 8 bytes ≈ 11 KiB (fits entirely into L1 cache!)
  thread_local std::vector<dna_active_neuron> active_list;
  active_list.clear();
  for (int32_t i = 0; i < L.n_ff; ++i) {
      if (d[i] != 0.0f) {
          active_list.push_back({i, d[i]});
      }
  }
  ```

- **Strategy B (Packed in `op_route`)**:
  `op_route` already identifies active neurons when evaluating the predictor band. It can write the count and indices directly into a sidecar scratch buffer attached to the tensor userdata.

*Recommendation*: Start with **Strategy A**. It requires zero changes to GGML graph tensor shapes or `op_route`, and scanning $14336$ floats in L1 takes less than $1.5\,\mu\text{s}$ while speeding up the tile loops by $5\text{--}10\times$.

---

## 6. Core AVX2 Tiled Kernel with Dynamic Subblocks

The scalar and AVX2 kernels are implemented in `src/dna-iq3-cxs2.c` as
`dna_iq3_cxs2_down_tile_accum` and
`dna_iq3_cxs2_down_tile_accum_avx2`. `op_down` runtime-dispatches to AVX2 when
supported and otherwise uses the scalar implementation.

```c
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

// Accumulates active neurons over a 160-element tile with n_subblocks (1..10)
void dna_iq3_cxs2_down_tile_accum_avx2(
    float                           * tile_accum,      // [160] F32 accumulator in L1
    const dna_active_neuron         * active_neurons,  // list of active neurons for this token
    size_t                            active_count,    // number of active neurons
    size_t                            tile_idx,        // tile index t in [0, n_blocks_per_neuron)
    size_t                            n_blocks_stride, // L.down_blocks_per_neuron
    const dna_iq3_cxs2_block        * down_blocks,     // [n_ff * n_blocks_stride]
    const uint8_t                   * down_scales,     // [n_ff * 8] (E5M3)
    size_t                            n_subblocks)     // 10 for full tile, <10 for tail tile
{
    const __m256 base_lut = _mm256_setr_ps(1.0f, 3.0f, 9.0f, 27.0f, 1.0f, 3.0f, 9.0f, 27.0f);

    for (size_t a = 0; a < active_count; ++a) {
        const int32_t i   = active_neurons[a].index;
        const float   act = active_neurons[a].activation;

        // Pointer to this neuron's 64-byte block for this tile
        const dna_iq3_cxs2_block * block = down_blocks + (size_t) i * n_blocks_stride + tile_idx;
        const uint8_t * s_e5m3 = down_scales + (size_t) i * 8;

        // Unpack 8 E5M3 scale factors
        float scales[8];
        for (int k = 0; k < 8; ++k) {
            scales[k] = dna_e5m3_to_f32(s_e5m3[k]);
        }

        const uint32_t selectors = dna_load_le32(block->bytes + 60);

        // Process only valid subblocks (skips tail padding)
        for (size_t sub = 0; sub < n_subblocks; ++sub) {
            const uint32_t s_idx = (selectors >> (3 * sub)) & 7u;
            const float factor = act * scales[s_idx];
            const __m256 block_lut = _mm256_mul_ps(base_lut, _mm256_set1_ps(factor));

            const uint8_t * payload = block->bytes + sub * 6;
            const uint16_t signs   = dna_load_le16(payload);
            const uint32_t indices = dna_load_le32(payload + 2);

            const __m256i idx0 = dna_unpack_2bit_indices_256(indices, 0);
            const __m256i idx1 = dna_unpack_2bit_indices_256(indices, 16);
            const __m256i s0   = dna_unpack_signs_256(signs, 0);
            const __m256i s1   = dna_unpack_signs_256(signs, 8);

            __m256 v0 = _mm256_permutevar8x32_ps(block_lut, idx0);
            __m256 v1 = _mm256_permutevar8x32_ps(block_lut, idx1);
            v0 = dna_apply_sign_256(v0, s0);
            v1 = dna_apply_sign_256(v1, s1);

            const size_t sub_off = sub * 16;
            __m256 sum0 = _mm256_loadu_ps(tile_accum + sub_off);
            __m256 sum1 = _mm256_loadu_ps(tile_accum + sub_off + 8);
            sum0 = _mm256_add_ps(sum0, v0);
            sum1 = _mm256_add_ps(sum1, v1);
            _mm256_storeu_ps(tile_accum + sub_off, sum0);
            _mm256_storeu_ps(tile_accum + sub_off + 8, sum1);
        }
    }
}
```

---

## 7. Updated `llama_dna::op_down` Implementation

```cpp
void llama_dna::op_down(ggml_tensor * dst, int ith, int nth, void * userdata) {
    llama_dna_layer & L = *(llama_dna_layer *) userdata;
    const ggml_tensor * delta = dst->src[0];
    const int64_t n_tokens = delta->ne[1];
    const int64_t t_start = ggml_time_us();

    const size_t n_tiles = L.down_blocks_per_neuron;
    const size_t tiles_per_th = (n_tiles + nth - 1) / nth;
    const size_t tile_0 = std::min(ith * tiles_per_th, n_tiles);
    const size_t tile_1 = std::min(tile_0 + tiles_per_th, n_tiles);

    if (tile_0 >= tile_1) return;

    // Buffer for active neuron list (per token)
    thread_local std::vector<dna_active_neuron> active_list;

    // Scratch tile accumulator in L1
    alignas(64) float tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * d = (const float *) ((const char *) delta->data + t * delta->nb[1]);
        float       * y = (float *) ((char *) dst->data + t * dst->nb[1]);

        // 1. Build compact active neuron list for token t
        active_list.clear();
        for (int32_t i = 0; i < L.n_ff; ++i) {
            if (d[i] != 0.0f) {
                active_list.push_back({i, d[i]});
            }
        }

        // 2. Process tiles owned by this thread
        for (size_t tile = tile_0; tile < tile_1; ++tile) {
            const size_t j_offset = tile * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
            const size_t remaining_weights = L.n_embd - j_offset;
            const size_t j_count = std::min<size_t>(DNA_IQ3_CXS2_WEIGHTS_PER_LINE, remaining_weights);

            // Compute how many 16-weight subblocks to process:
            // 10 for full tiles, ceil(remaining / 16) for tail tile
            const size_t n_subblocks = (j_count + 15) / 16;

            // Initialize tile with c0 bias
            memcpy(tile_accum, L.c0.data() + j_offset, j_count * sizeof(float));
            if (j_count < DNA_IQ3_CXS2_WEIGHTS_PER_LINE) {
                memset(tile_accum + j_count, 0, (DNA_IQ3_CXS2_WEIGHTS_PER_LINE - j_count) * sizeof(float));
            }

            // Accumulate active neurons using AVX2 with dynamic subblocks
            dna_iq3_cxs2_down_tile_accum_avx2(
                tile_accum,
                active_list.data(),
                active_list.size(),
                tile,
                L.down_blocks_per_neuron,
                L.down_blocks.data(),
                L.down_scales.data(),
                n_subblocks);

            // Write finished tile back to output y once
            memcpy(y + j_offset, tile_accum, j_count * sizeof(float));
        }
    }

    if (ith == 0) {
        L.t_down_us += ggml_time_us() - t_start;
    }
}
```

`op_expert_down` uses the same tiling and kernel. It folds each expert's router
weight into that slot's active-neuron values and applies the same weight to
`c0`, then sums the expert contributions into the output tile.

---

## 8. Summary of Benefits

1. **Active List Acceleration**:
   Instead of iterating over $14336$ neurons for every tile, the inner loop executes only for the active neurons (typically $5\text{--}15\% \approx 700\text{--}2100$ iterations), yielding immediate speedups.
2. **Exact Tail Handling without Overhead**:
   Passing `n_subblocks` allows tail tiles (e.g. 96 weights = 6 subblocks) to skip unneeded subblocks with zero compute overhead while retaining cache-line-aligned 64-byte block access.
3. **8-Byte Scale Storage Simplicity**:
   Contiguous table of $8 \times n_{\text{ff}}$ bytes allows fast 64-bit scalar indexing per active neuron with minimal memory footprint (112 KiB total per layer).
