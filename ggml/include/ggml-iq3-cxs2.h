#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DNA_IQ3_CXS2_LINE_BYTES           64
#define DNA_IQ3_CXS2_WEIGHTS_PER_LINE     160
#define DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK 16
#define DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE   10

typedef struct dna_iq3_cxs2_block {
    alignas(64) uint8_t bytes[DNA_IQ3_CXS2_LINE_BYTES];
} dna_iq3_cxs2_block;

typedef struct dna_iq3_cxs2_active_neuron {
    int32_t index;
    float   activation;
} dna_iq3_cxs2_active_neuron;

typedef struct dna_iq3_cxs2_sparse_row {
    size_t                         output_row;
    const dna_iq3_cxs2_block     * blocks;
    const uint8_t                * scales_e5m3;
} dna_iq3_cxs2_sparse_row;

// Convert 32-bit float to unsigned 8-bit E5M3 (5 exponent bits, 3 mantissa bits, bias 15)
uint8_t dna_f32_to_e5m3(float val);

// Convert unsigned 8-bit E5M3 to 32-bit float
float dna_e5m3_to_f32(uint8_t val);

// Find MSE-optimal continuous scale for a 16-weight subblock via coordinate descent
float dna_iq3_cxs2_subblock_mse_scale(const float weights[16]);

// Compute the 8 per-neuron scales from an array of float weights
void dna_iq3_cxs2_compute_neuron_scales(
    const float * weights,
    size_t        n_weights,
    uint8_t       out_scales_e5m3[8],
    float         out_scales_f32[8]);

// Quantize a row (neuron) of float weights into IQ3_CXS2 blocks and 8 E5M3 scales
// n_weights must be a multiple of DNA_IQ3_CXS2_WEIGHTS_PER_LINE (160)
void dna_iq3_cxs2_quantize_row(
    const float        * weights,
    size_t               n_weights,
    dna_iq3_cxs2_block * out_blocks,
    uint8_t              out_scales_e5m3[8]);

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
// AVX2 accelerated quantization of a row
void dna_iq3_cxs2_quantize_row_avx2(
    const float        * weights,
    size_t               n_weights,
    dna_iq3_cxs2_block * out_blocks,
    uint8_t              out_scales_e5m3[8]);
#endif

// Dequantize an IQ3_CXS2 block back into 160 float weights for testing/verification
void dna_iq3_cxs2_dequantize_block(
    const dna_iq3_cxs2_block * block,
    const float                scales_f32[8],
    float                      out_weights[160]);

size_t ggml_iq3_cxs2_row_size(size_t n_weights);
void ggml_quantize_row_iq3_cxs2(const float * weights, void * row, int64_t n_weights);
void ggml_dequantize_row_iq3_cxs2(const void * row, float * weights, int64_t n_weights);

// Sparse indirect gate/up GEMV. x_q8 contains signed Q8 values with one scale
// per 32 inputs; rows may reference arbitrary non-adjacent IQ3_CXS2 rows.
// n_inputs must be padded to a multiple of 160; the final line may contain zeros.
// Only output_row entries are written.
void dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
    const int8_t                    * x_q8,
    const float                     * x_scales,
    size_t                            n_inputs,
    const dna_iq3_cxs2_sparse_row   * rows,
    size_t                            active_count,
    float                           * output);

// AVX2 implementation where supported; otherwise uses the scalar reference.
void dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx2(
    const int8_t                    * x_q8,
    const float                     * x_scales,
    size_t                            n_inputs,
    const dna_iq3_cxs2_sparse_row   * rows,
    size_t                            active_count,
    float                           * output);

// Add active-neuron contributions to one 160-weight output tile.
// The AVX2 variant requires runtime CPU-feature dispatch by the caller.
void dna_iq3_cxs2_down_tile_accum(
    float                           tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE],
    const dna_iq3_cxs2_active_neuron * active_neurons,
    size_t                           active_count,
    size_t                           tile_idx,
    size_t                           n_blocks_stride,
    const dna_iq3_cxs2_block        * down_blocks,
    const uint8_t                   * down_scales,
    size_t                           n_subblocks);

void dna_iq3_cxs2_down_tile_accum_avx2(
    float                           tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE],
    const dna_iq3_cxs2_active_neuron * active_neurons,
    size_t                           active_count,
    size_t                           tile_idx,
    size_t                           n_blocks_stride,
    const dna_iq3_cxs2_block        * down_blocks,
    const uint8_t                   * down_scales,
    size_t                           n_subblocks);

#ifdef __cplusplus
}
#endif
