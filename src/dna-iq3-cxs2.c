#include "dna-iq3-cxs2.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#include <immintrin.h>
#endif

static const int32_t dna_iq3_cxs2_magnitudes[4] = { 1, 3, 9, 27 };

static inline void dna_store_le16(uint8_t * p, uint16_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
}

static inline void dna_store_le32(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t) (v);
    p[1] = (uint8_t) (v >> 8);
    p[2] = (uint8_t) (v >> 16);
    p[3] = (uint8_t) (v >> 24);
}

static inline uint16_t dna_load_le16(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static inline uint32_t dna_load_le32(const uint8_t * p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) |
           ((uint32_t) p[3] << 24);
}

uint8_t dna_f32_to_e5m3(float val) {
    if (val <= 0.0f) {
        return 0;
    }
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    if (exp <= 0) {
        return 0;
    }
    if (exp >= 31) {
        return 0xFF;
    }
    uint32_t mant = (bits >> 20) & 0x7;
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

float dna_e5m3_to_f32(uint8_t val) {
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

float dna_iq3_cxs2_subblock_mse_scale(const float weights[16]) {
    float max_abs = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float a = fabsf(weights[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0f) {
        return 0.0f;
    }

    float s = max_abs / 27.0f;

    for (int iter = 0; iter < 3; ++iter) {
        if (s <= 0.0f) break;
        float inv_s = 1.0f / s;
        float sum_wk = 0.0f;
        float sum_k2 = 0.0f;

        for (int i = 0; i < 16; ++i) {
            float abs_w = fabsf(weights[i]);
            float target = abs_w * inv_s;

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

void dna_iq3_cxs2_compute_neuron_scales(
    const float * weights,
    size_t        n_weights,
    uint8_t       out_scales_e5m3[8],
    float         out_scales_f32[8])
{
    const size_t n_subblocks = n_weights / 16;
    float * sub_scales = (float *) malloc(n_subblocks * sizeof(float));
    float * sub_energy = (float *) malloc(n_subblocks * sizeof(float));

    float max_scale = 0.0f;
    float min_scale = 1e30f;
    size_t non_zero_count = 0;

    for (size_t b = 0; b < n_subblocks; ++b) {
        const float * w = weights + b * 16;
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
        for (int k = 0; k < 8; ++k) {
            out_scales_e5m3[k] = 0;
            out_scales_f32[k] = 0.0f;
        }
        free(sub_scales);
        free(sub_energy);
        return;
    }

    float centroids[8];
    float ratio = (max_scale > min_scale) ? (max_scale / min_scale) : 1.0f;
    for (int k = 0; k < 8; ++k) {
        centroids[k] = min_scale * powf(ratio, (float) k / 7.0f);
    }

    for (int iter = 0; iter < 8; ++iter) {
        float cluster_weighted_sum[8] = { 0 };
        float cluster_weights[8] = { 0 };

        for (size_t b = 0; b < n_subblocks; ++b) {
            float s = sub_scales[b];
            if (s <= 0.0f) continue;

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

        for (int k = 0; k < 8; ++k) {
            if (cluster_weights[k] > 0.0f) {
                centroids[k] = cluster_weighted_sum[k] / cluster_weights[k];
            }
        }
    }

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

static float dna_iq3_cxs2_quantize_subblock_exact(
    const float  weights[16],
    float        scale,
    uint16_t   * out_signs,
    uint32_t   * out_indices)
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
            best_k_idx = 3;
        } else if (target > 6.0f) {
            best_k_idx = 2;
        } else if (target > 2.0f) {
            best_k_idx = 1;
        } else {
            best_k_idx = 0;
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
    const float        * weights,
    size_t               n_weights,
    dna_iq3_cxs2_block * out_blocks,
    uint8_t              out_scales_e5m3[8])
{
    assert(n_weights % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);
    const size_t n_lines = n_weights / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    float scales_f32[8];
    dna_iq3_cxs2_compute_neuron_scales(weights, n_weights, out_scales_e5m3, scales_f32);

    for (size_t l = 0; l < n_lines; ++l) {
        const float * w_line = weights + l * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
        dna_iq3_cxs2_block * block = &out_blocks[l];
        memset(block->bytes, 0, DNA_IQ3_CXS2_LINE_BYTES);

        uint32_t selectors = 0;

        for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
            const float * w_sub = w_line + sub * 16;

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

            uint8_t * payload = block->bytes + sub * 6;
            dna_store_le16(payload, best_signs);
            dna_store_le32(payload + 2, best_indices);
            selectors |= (best_selector << (3 * sub));
        }

        dna_store_le32(block->bytes + 60, selectors);
    }
}

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
#endif
void dna_iq3_cxs2_quantize_row_avx2(
    const float        * weights,
    size_t               n_weights,
    dna_iq3_cxs2_block * out_blocks,
    uint8_t              out_scales_e5m3[8])
{
    // The scale compute is done across all weights
    assert(n_weights % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);
    const size_t n_lines = n_weights / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    float scales_f32[8];
    dna_iq3_cxs2_compute_neuron_scales(weights, n_weights, out_scales_e5m3, scales_f32);

    for (size_t l = 0; l < n_lines; ++l) {
        const float * w_line = weights + l * DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
        dna_iq3_cxs2_block * block = &out_blocks[l];
        memset(block->bytes, 0, DNA_IQ3_CXS2_LINE_BYTES);

        uint32_t selectors = 0;

        for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
            const float * w_sub = w_line + sub * 16;

            // Load 16 weights using two 256-bit AVX registers
            __m256 w0 = _mm256_loadu_ps(w_sub);
            __m256 w1 = _mm256_loadu_ps(w_sub + 8);

            // Compute abs(w) by masking sign bit
            __m256 sign_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 abs_w0 = _mm256_and_ps(w0, sign_mask_ps);
            __m256 abs_w1 = _mm256_and_ps(w1, sign_mask_ps);

            // Extract signs (1 if negative, 0 if positive)
            // bit 31 is set if negative -> movemask on ps gives 8 bits
            int sign_mask0 = _mm256_movemask_ps(w0);
            int sign_mask1 = _mm256_movemask_ps(w1);
            uint16_t sub_signs = (uint16_t) (sign_mask0 | (sign_mask1 << 8));

            float best_err = 1e30f;
            uint32_t best_selector = 0;
            uint32_t best_indices = 0;

            for (uint32_t s = 0; s < 8; ++s) {
                float scale = scales_f32[s];
                if (scale <= 0.0f) continue;

                float inv_scale = 1.0f / scale;
                __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

                __m256 target0 = _mm256_mul_ps(abs_w0, inv_scale_vec);
                __m256 target1 = _mm256_mul_ps(abs_w1, inv_scale_vec);

                // Thresholds: 2.0, 6.0, 18.0
                __m256 th_18 = _mm256_set1_ps(18.0f);
                __m256 th_6  = _mm256_set1_ps(6.0f);
                __m256 th_2  = _mm256_set1_ps(2.0f);

                // Compare > th
                __m256 cmp18_0 = _mm256_cmp_ps(target0, th_18, _CMP_GT_OQ);
                __m256 cmp6_0  = _mm256_cmp_ps(target0, th_6,  _CMP_GT_OQ);
                __m256 cmp2_0  = _mm256_cmp_ps(target0, th_2,  _CMP_GT_OQ);

                __m256 cmp18_1 = _mm256_cmp_ps(target1, th_18, _CMP_GT_OQ);
                __m256 cmp6_1  = _mm256_cmp_ps(target1, th_6,  _CMP_GT_OQ);
                __m256 cmp2_1  = _mm256_cmp_ps(target1, th_2,  _CMP_GT_OQ);

                // Magnitude: 1 + 2*(cmp2) + 6*(cmp6) + 18*(cmp18)
                // In float:
                __m256 mag0 = _mm256_set1_ps(1.0f);
                mag0 = _mm256_add_ps(mag0, _mm256_and_ps(cmp2_0, _mm256_set1_ps(2.0f)));
                mag0 = _mm256_add_ps(mag0, _mm256_and_ps(cmp6_0, _mm256_set1_ps(6.0f)));
                mag0 = _mm256_add_ps(mag0, _mm256_and_ps(cmp18_0, _mm256_set1_ps(18.0f)));

                __m256 mag1 = _mm256_set1_ps(1.0f);
                mag1 = _mm256_add_ps(mag1, _mm256_and_ps(cmp2_1, _mm256_set1_ps(2.0f)));
                mag1 = _mm256_add_ps(mag1, _mm256_and_ps(cmp6_1, _mm256_set1_ps(6.0f)));
                mag1 = _mm256_add_ps(mag1, _mm256_and_ps(cmp18_1, _mm256_set1_ps(18.0f)));

                // Reconstructed: scale * mag
                __m256 scale_vec = _mm256_set1_ps(scale);
                __m256 recon0 = _mm256_mul_ps(mag0, scale_vec);
                __m256 recon1 = _mm256_mul_ps(mag1, scale_vec);

                // Error diff
                __m256 diff0 = _mm256_sub_ps(abs_w0, recon0);
                __m256 diff1 = _mm256_sub_ps(abs_w1, recon1);
                __m256 sq_err = _mm256_add_ps(_mm256_mul_ps(diff0, diff0), _mm256_mul_ps(diff1, diff1));

                // Horizontal sum of sq_err
                __m128 lo = _mm256_castps256_ps128(sq_err);
                __m128 hi = _mm256_extractf128_ps(sq_err, 1);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                float err = _mm_cvtss_f32(sum128);

                if (err < best_err) {
                    best_err = err;
                    best_selector = s;

                    // Compute 2-bit index per lane:
                    // cmp18 -> idx=3
                    // cmp6  -> idx=2
                    // cmp2  -> idx=1
                    // else  -> idx=0
                    // index = (cmp2 & 1) + (cmp6 & 1) + (cmp18 & 1)
                    __m256i i_cmp2_0  = _mm256_and_si256(_mm256_castps_si256(cmp2_0),  _mm256_set1_epi32(1));
                    __m256i i_cmp6_0  = _mm256_and_si256(_mm256_castps_si256(cmp6_0),  _mm256_set1_epi32(1));
                    __m256i i_cmp18_0 = _mm256_and_si256(_mm256_castps_si256(cmp18_0), _mm256_set1_epi32(1));
                    __m256i idx0_vec  = _mm256_add_epi32(i_cmp2_0, _mm256_add_epi32(i_cmp6_0, i_cmp18_0));

                    __m256i i_cmp2_1  = _mm256_and_si256(_mm256_castps_si256(cmp2_1),  _mm256_set1_epi32(1));
                    __m256i i_cmp6_1  = _mm256_and_si256(_mm256_castps_si256(cmp6_1),  _mm256_set1_epi32(1));
                    __m256i i_cmp18_1 = _mm256_and_si256(_mm256_castps_si256(cmp18_1), _mm256_set1_epi32(1));
                    __m256i idx1_vec  = _mm256_add_epi32(i_cmp2_1, _mm256_add_epi32(i_cmp6_1, i_cmp18_1));

                    uint32_t raw_idx0[8], raw_idx1[8];
                    _mm256_storeu_si256((__m256i *) raw_idx0, idx0_vec);
                    _mm256_storeu_si256((__m256i *) raw_idx1, idx1_vec);

                    uint32_t packed_idx = 0;
                    for (int i = 0; i < 8; ++i) {
                        packed_idx |= (raw_idx0[i] & 3u) << (2 * i);
                        packed_idx |= (raw_idx1[i] & 3u) << (2 * (i + 8));
                    }
                    best_indices = packed_idx;
                }
            }

            uint8_t * payload = block->bytes + sub * 6;
            dna_store_le16(payload, sub_signs);
            dna_store_le32(payload + 2, best_indices);
            selectors |= (best_selector << (3 * sub));
        }

        dna_store_le32(block->bytes + 60, selectors);
    }
}
#endif

void dna_iq3_cxs2_dequantize_block(
    const dna_iq3_cxs2_block * block,
    const float                scales_f32[8],
    float                      out_weights[160])
{
    uint32_t selectors = dna_load_le32(block->bytes + 60);

    for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
        const uint8_t * payload = block->bytes + sub * 6;
        uint16_t signs = dna_load_le16(payload);
        uint32_t indices = dna_load_le32(payload + 2);
        uint32_t s_idx = (selectors >> (3 * sub)) & 7u;
        float scale = scales_f32[s_idx];

        for (size_t i = 0; i < 16; ++i) {
            uint32_t idx = (indices >> (2 * i)) & 3u;
            int sign = (signs >> i) & 1;
            float mag = (float) dna_iq3_cxs2_magnitudes[idx];
            float val = scale * mag;
            out_weights[sub * 16 + i] = sign ? -val : val;
        }
    }
}

void dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
    const int8_t                    * x_q8,
    const float                     * x_scales,
    size_t                            n_inputs,
    const dna_iq3_cxs2_sparse_row   * rows,
    size_t                            active_count,
    float                           * output)
{
    assert(n_inputs % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);

    const size_t n_lines = n_inputs / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;
    for (size_t group_start = 0; group_start < active_count; group_start += 4) {
        const size_t row_count =
            active_count - group_start < 4 ? active_count - group_start : 4;
        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t line = 0; line < n_lines; ++line) {
            const dna_iq3_cxs2_block * blocks[4];
            uint32_t selectors[4];
            for (size_t r = 0; r < row_count; ++r) {
                blocks[r] = rows[group_start + r].blocks + line;
                selectors[r] = dna_load_le32(blocks[r]->bytes + 60);
            }

            for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
                const size_t input_offset =
                    line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    sub * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                const size_t q8_block = input_offset / 32;
                const size_t q8_offset = input_offset % 32;
                assert(q8_offset == 0 || q8_offset == 16);
                const int8_t * activation = x_q8 + q8_block * 32 + q8_offset;

                int32_t activation_lut[DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK][4];
                for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
                    const int32_t q = activation[i];
                    for (size_t k = 0; k < 4; ++k) {
                        activation_lut[i][k] = q * dna_iq3_cxs2_magnitudes[k];
                    }
                }

                const float activation_scale = x_scales[q8_block];
                for (size_t r = 0; r < row_count; ++r) {
                    const uint8_t * payload = blocks[r]->bytes + sub * 6;
                    const uint16_t signs = dna_load_le16(payload);
                    const uint32_t indices = dna_load_le32(payload + 2);
                    const uint32_t scale_idx =
                        (selectors[r] >> (3 * sub)) & 7u;

                    int32_t dot = 0;
                    for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
                        const uint32_t index = (indices >> (2 * i)) & 3u;
                        const int32_t term = activation_lut[i][index];
                        dot += ((signs >> i) & 1u) ? -term : term;
                    }

                    const float row_scale = dna_e5m3_to_f32(
                        rows[group_start + r].scales_e5m3[scale_idx]);
                    accum[r] += (float) dot * (activation_scale * row_scale);
                }
            }
        }

        for (size_t r = 0; r < row_count; ++r) {
            output[rows[group_start + r].output_row] = accum[r];
        }
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
#endif
void dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx2(
    const int8_t                    * x_q8,
    const float                     * x_scales,
    size_t                            n_inputs,
    const dna_iq3_cxs2_sparse_row   * rows,
    size_t                            active_count,
    float                           * output)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    assert(n_inputs % DNA_IQ3_CXS2_WEIGHTS_PER_LINE == 0);

    const __m256i shifts_lo =
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i shifts_hi =
        _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i sign_shifts_lo =
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i sign_shifts_hi =
        _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i offsets_lo =
        _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i offsets_hi =
        _mm256_setr_epi32(32, 36, 40, 44, 48, 52, 56, 60);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256i mask1 = _mm256_set1_epi32(1);
    const __m256i one = _mm256_set1_epi32(1);
    const size_t n_lines = n_inputs / DNA_IQ3_CXS2_WEIGHTS_PER_LINE;

    for (size_t group_start = 0; group_start < active_count; group_start += 4) {
        const size_t row_count =
            active_count - group_start < 4 ? active_count - group_start : 4;
        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (size_t line = 0; line < n_lines; ++line) {
            const dna_iq3_cxs2_block * blocks[4];
            uint32_t selectors[4];
            for (size_t r = 0; r < row_count; ++r) {
                blocks[r] = rows[group_start + r].blocks + line;
                selectors[r] = dna_load_le32(blocks[r]->bytes + 60);
            }

            for (size_t sub = 0; sub < DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE; ++sub) {
                const size_t input_offset =
                    line * DNA_IQ3_CXS2_WEIGHTS_PER_LINE +
                    sub * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
                const size_t q8_block = input_offset / 32;
                const size_t q8_offset = input_offset % 32;
                assert(q8_offset == 0 || q8_offset == 16);
                const int8_t * activation = x_q8 + q8_block * 32 + q8_offset;

                int32_t activation_lut[DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK][4];
                for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
                    const int32_t q = activation[i];
                    for (size_t k = 0; k < 4; ++k) {
                        activation_lut[i][k] = q * dna_iq3_cxs2_magnitudes[k];
                    }
                }

                const float activation_scale = x_scales[q8_block];
                for (size_t r = 0; r < row_count; ++r) {
                    const uint8_t * payload = blocks[r]->bytes + sub * 6;
                    const uint16_t signs = dna_load_le16(payload);
                    const uint32_t indices = dna_load_le32(payload + 2);
                    const uint32_t scale_idx =
                        (selectors[r] >> (3 * sub)) & 7u;

                    const __m256i packed_indices =
                        _mm256_set1_epi32((int32_t) indices);
                    const __m256i index_lo = _mm256_and_si256(
                        _mm256_srlv_epi32(packed_indices, shifts_lo), mask3);
                    const __m256i index_hi = _mm256_and_si256(
                        _mm256_srlv_epi32(packed_indices, shifts_hi), mask3);
                    const __m256i gather_lo = _mm256_add_epi32(offsets_lo, index_lo);
                    const __m256i gather_hi = _mm256_add_epi32(offsets_hi, index_hi);
                    __m256i value_lo = _mm256_i32gather_epi32(
                        &activation_lut[0][0], gather_lo, sizeof(int32_t));
                    __m256i value_hi = _mm256_i32gather_epi32(
                        &activation_lut[0][0], gather_hi, sizeof(int32_t));

                    const __m256i packed_signs =
                        _mm256_set1_epi32((int32_t) signs);
                    const __m256i sign_lo = _mm256_and_si256(
                        _mm256_srlv_epi32(packed_signs, sign_shifts_lo), mask1);
                    const __m256i sign_hi = _mm256_and_si256(
                        _mm256_srlv_epi32(packed_signs, sign_shifts_hi), mask1);
                    value_lo = _mm256_mullo_epi32(
                        value_lo, _mm256_sub_epi32(one, _mm256_slli_epi32(sign_lo, 1)));
                    value_hi = _mm256_mullo_epi32(
                        value_hi, _mm256_sub_epi32(one, _mm256_slli_epi32(sign_hi, 1)));

                    const __m256i sum = _mm256_add_epi32(value_lo, value_hi);
                    const __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
                    __m128i reduced = _mm_hadd_epi32(sum128, sum128);
                    reduced = _mm_hadd_epi32(reduced, reduced);
                    const int32_t dot = _mm_cvtsi128_si32(reduced);

                    const float row_scale = dna_e5m3_to_f32(
                        rows[group_start + r].scales_e5m3[scale_idx]);
                    accum[r] += (float) dot * (activation_scale * row_scale);
                }
            }
        }

        for (size_t r = 0; r < row_count; ++r) {
            output[rows[group_start + r].output_row] = accum[r];
        }
    }
#else
    dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
        x_q8, x_scales, n_inputs, rows, active_count, output);
#endif
}

void dna_iq3_cxs2_down_tile_accum(
    float                            tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE],
    const dna_iq3_cxs2_active_neuron * active_neurons,
    size_t                            active_count,
    size_t                            tile_idx,
    size_t                            n_blocks_stride,
    const dna_iq3_cxs2_block         * down_blocks,
    const uint8_t                    * down_scales,
    size_t                            n_subblocks)
{
    assert(n_subblocks <= DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE);

    for (size_t a = 0; a < active_count; ++a) {
        const int32_t neuron = active_neurons[a].index;
        const float activation = active_neurons[a].activation;
        const dna_iq3_cxs2_block * block =
            down_blocks + (size_t) neuron * n_blocks_stride + tile_idx;
        const uint8_t * scales_e5m3 = down_scales + (size_t) neuron * 8;
        float scales[8];

        for (size_t k = 0; k < 8; ++k) {
            scales[k] = dna_e5m3_to_f32(scales_e5m3[k]);
        }

        const uint32_t selectors = dna_load_le32(block->bytes + 60);
        for (size_t sub = 0; sub < n_subblocks; ++sub) {
            const uint8_t * payload = block->bytes + sub * 6;
            const uint16_t signs = dna_load_le16(payload);
            const uint32_t indices = dna_load_le32(payload + 2);
            const uint32_t scale_idx = (selectors >> (3 * sub)) & 7u;
            const float scale = activation * scales[scale_idx];
            const size_t sub_offset = sub * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;

            for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
                const uint32_t index = (indices >> (2 * i)) & 3u;
                const float contribution =
                    scale * (float) dna_iq3_cxs2_magnitudes[index];
                tile_accum[sub_offset + i] +=
                    ((signs >> i) & 1u) ? -contribution : contribution;
            }
        }
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
#endif
void dna_iq3_cxs2_down_tile_accum_avx2(
    float                            tile_accum[DNA_IQ3_CXS2_WEIGHTS_PER_LINE],
    const dna_iq3_cxs2_active_neuron * active_neurons,
    size_t                            active_count,
    size_t                            tile_idx,
    size_t                            n_blocks_stride,
    const dna_iq3_cxs2_block         * down_blocks,
    const uint8_t                    * down_scales,
    size_t                            n_subblocks)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    assert(n_subblocks <= DNA_IQ3_CXS2_SUBBLOCKS_PER_LINE);

    const __m256 base_lut =
        _mm256_setr_ps(1.0f, 3.0f, 9.0f, 27.0f, 1.0f, 3.0f, 9.0f, 27.0f);
    const __m256i index_shifts0 =
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i index_shifts1 =
        _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i sign_shifts0 =
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i sign_shifts1 =
        _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256i mask1 = _mm256_set1_epi32(1);

    for (size_t a = 0; a < active_count; ++a) {
        const int32_t neuron = active_neurons[a].index;
        const float activation = active_neurons[a].activation;
        const dna_iq3_cxs2_block * block =
            down_blocks + (size_t) neuron * n_blocks_stride + tile_idx;
        const uint8_t * scales_e5m3 = down_scales + (size_t) neuron * 8;
        float scales[8];
        for (size_t k = 0; k < 8; ++k) {
            scales[k] = dna_e5m3_to_f32(scales_e5m3[k]);
        }

        const uint32_t selectors = dna_load_le32(block->bytes + 60);
        for (size_t sub = 0; sub < n_subblocks; ++sub) {
            const uint8_t * payload = block->bytes + sub * 6;
            const uint16_t signs = dna_load_le16(payload);
            const uint32_t indices = dna_load_le32(payload + 2);
            const uint32_t scale_idx = (selectors >> (3 * sub)) & 7u;
            const __m256 lut = _mm256_mul_ps(
                base_lut, _mm256_set1_ps(activation * scales[scale_idx]));

            const __m256i packed_indices =
                _mm256_set1_epi32((int32_t) indices);
            const __m256i packed_signs =
                _mm256_set1_epi32((int32_t) signs);
            const __m256i idx0 =
                _mm256_and_si256(_mm256_srlv_epi32(packed_indices, index_shifts0), mask3);
            const __m256i idx1 =
                _mm256_and_si256(_mm256_srlv_epi32(packed_indices, index_shifts1), mask3);
            const __m256i sign0 =
                _mm256_and_si256(_mm256_srlv_epi32(packed_signs, sign_shifts0), mask1);
            const __m256i sign1 =
                _mm256_and_si256(_mm256_srlv_epi32(packed_signs, sign_shifts1), mask1);

            __m256 values0 = _mm256_permutevar8x32_ps(lut, idx0);
            __m256 values1 = _mm256_permutevar8x32_ps(lut, idx1);
            values0 = _mm256_castsi256_ps(_mm256_xor_si256(
                _mm256_castps_si256(values0), _mm256_slli_epi32(sign0, 31)));
            values1 = _mm256_castsi256_ps(_mm256_xor_si256(
                _mm256_castps_si256(values1), _mm256_slli_epi32(sign1, 31)));

            const size_t offset =
                sub * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK;
            _mm256_storeu_ps(tile_accum + offset,
                _mm256_add_ps(_mm256_loadu_ps(tile_accum + offset), values0));
            _mm256_storeu_ps(tile_accum + offset + 8,
                _mm256_add_ps(_mm256_loadu_ps(tile_accum + offset + 8), values1));
        }
    }
#else
    dna_iq3_cxs2_down_tile_accum(
        tile_accum, active_neurons, active_count, tile_idx, n_blocks_stride,
        down_blocks, down_scales, n_subblocks);
#endif
}
