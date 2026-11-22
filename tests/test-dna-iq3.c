#include "../src/dna-iq3-cxs2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

static float rand_normal(float mean, float std) {
    // Box-Muller transform
    float u = (float) rand() / (float) RAND_MAX;
    float v = (float) rand() / (float) RAND_MAX;
    if (u <= 1e-7f) u = 1e-7f;
    return mean + std * sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.1415926535f * v);
}

static void test_down_tile_accum(void) {
    enum {
        N_NEURONS = 3,
        N_WEIGHTS = 176,
        N_BLOCKS = 2,
        PADDED_WEIGHTS = N_BLOCKS * DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
    };

    dna_iq3_cxs2_block blocks[N_NEURONS * N_BLOCKS];
    uint8_t scales_e5m3[N_NEURONS * 8];
    float row[PADDED_WEIGHTS];

    for (size_t n = 0; n < N_NEURONS; ++n) {
        for (size_t i = 0; i < PADDED_WEIGHTS; ++i) {
            row[i] = i < N_WEIGHTS
                ? 0.3f * sinf((float) (i * 13 + n * 17) * 0.071f)
                : 0.0f;
        }
        dna_iq3_cxs2_quantize_row(
            row,
            PADDED_WEIGHTS,
            blocks + n * N_BLOCKS,
            scales_e5m3 + n * 8);
    }

    const dna_iq3_cxs2_active_neuron active[] = {
        { 2, 0.75f },
        { 0, -0.5f },
    };
    const size_t test_tiles[] = { 0, 1 };
    const size_t test_subblocks[] = { 10, 1 };

    for (size_t test = 0; test < 2; ++test) {
        const size_t tile = test_tiles[test];
        const size_t n_subblocks = test_subblocks[test];
        float actual[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];
        float expected[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];

        for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
            actual[i] = (float) ((int) i - 80) * 0.002f;
            expected[i] = actual[i];
        }

        for (size_t a = 0; a < sizeof(active) / sizeof(active[0]); ++a) {
            const size_t neuron = (size_t) active[a].index;
            float scales_f32[8];
            float decoded[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];
            for (size_t k = 0; k < 8; ++k) {
                scales_f32[k] = dna_e5m3_to_f32(scales_e5m3[neuron * 8 + k]);
            }
            dna_iq3_cxs2_dequantize_block(
                blocks + neuron * N_BLOCKS + tile, scales_f32, decoded);
            for (size_t sub = 0; sub < n_subblocks; ++sub) {
                for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK; ++i) {
                    const size_t offset =
                        sub * DNA_IQ3_CXS2_WEIGHTS_PER_SUBBLOCK + i;
                    expected[offset] += active[a].activation * decoded[offset];
                }
            }
        }

        dna_iq3_cxs2_down_tile_accum(
            actual, active, sizeof(active) / sizeof(active[0]), tile, N_BLOCKS,
            blocks, scales_e5m3, n_subblocks);
        for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
            assert(fabsf(actual[i] - expected[i]) < 1e-6f);
        }

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2")) {
            float actual_avx2[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];
            for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
                actual_avx2[i] = (float) ((int) i - 80) * 0.002f;
            }
            dna_iq3_cxs2_down_tile_accum_avx2(
                actual_avx2, active, sizeof(active) / sizeof(active[0]), tile,
                N_BLOCKS, blocks, scales_e5m3, n_subblocks);
            assert(memcmp(actual, actual_avx2, sizeof(actual)) == 0);
            for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
                assert(fabsf(actual_avx2[i] - expected[i]) < 1e-6f);
            }
        }
#endif
    }
}

int main(void) {
    printf("=== Testing IQ3_CXS2 Quantization ===\n");

    // 1. Test E5M3 conversion accuracy & round-trip
    printf("[1] Testing E5M3 conversion...\n");
    float test_scales[] = { 0.001f, 0.05f, 0.125f, 0.5f, 1.0f, 2.5f, 10.0f, 100.0f };
    for (size_t i = 0; i < sizeof(test_scales)/sizeof(test_scales[0]); ++i) {
        float original = test_scales[i];
        uint8_t e5m3 = dna_f32_to_e5m3(original);
        float recovered = dna_e5m3_to_f32(e5m3);
        float rel_err = fabsf(original - recovered) / original;
        printf("    orig: %8.4f -> E5M3: 0x%02X -> recov: %8.4f (rel err: %.2f%%)\n",
               original, e5m3, recovered, rel_err * 100.0f);
        // E5M3 has 3 mantissa bits, so maximum relative quantization step is ~ 1/8 = 12.5%
        assert(rel_err < 0.15f);
    }

    // 2. Test Quantization and Dequantization Round-Trip on synthetic weights
    const size_t n_weights = 160 * 8; // 8 superblocks = 1280 weights (80 subblocks)
    float * weights = (float *) malloc(n_weights * sizeof(float));
    float * dequant_scalar = (float *) malloc(n_weights * sizeof(float));
    dna_iq3_cxs2_block * blocks_scalar = (dna_iq3_cxs2_block *) malloc(8 * sizeof(dna_iq3_cxs2_block));
    uint8_t scales_e5m3_scalar[8];

    // Seed RNG
    srand(42);
    for (size_t i = 0; i < n_weights; ++i) {
        // Typical transformer MLP weights: mean ~0, std ~0.05, with occasional outliers
        float w = rand_normal(0.0f, 0.04f);
        if (i % 50 == 0) w *= 3.0f; // outlier
        weights[i] = w;
    }

    printf("[2] Quantizing %zu weights with scalar path...\n", n_weights);
    dna_iq3_cxs2_quantize_row(weights, n_weights, blocks_scalar, scales_e5m3_scalar);

    printf("    Learned 8 E5M3 scales:\n    ");
    float scales_f32[8];
    for (int k = 0; k < 8; ++k) {
        scales_f32[k] = dna_e5m3_to_f32(scales_e5m3_scalar[k]);
        printf("[%d]=%.5f (0x%02X) ", k, scales_f32[k], scales_e5m3_scalar[k]);
    }
    printf("\n");

    // Dequantize and calculate SNR / RMSE
    float sum_sq_err = 0.0f;
    float sum_sq_orig = 0.0f;
    for (size_t l = 0; l < 8; ++l) {
        dna_iq3_cxs2_dequantize_block(&blocks_scalar[l], scales_f32, dequant_scalar + l * 160);
    }

    for (size_t i = 0; i < n_weights; ++i) {
        float diff = weights[i] - dequant_scalar[i];
        sum_sq_err += diff * diff;
        sum_sq_orig += weights[i] * weights[i];
    }

    float rmse = sqrtf(sum_sq_err / (float) n_weights);
    float snr = 10.0f * log10f(sum_sq_orig / sum_sq_err);
    printf("    Scalar RMSE: %.6f, SNR: %.2f dB\n", rmse, snr);
    assert(snr > 10.0f); // 3-bit quant with non-linear codebook typically achieves 12-18 dB SNR

#if defined(__AVX2__)
    // 3. Test AVX2 path equivalence
    printf("[3] Testing AVX2 quantization path...\n");
    dna_iq3_cxs2_block * blocks_avx2 = (dna_iq3_cxs2_block *) malloc(8 * sizeof(dna_iq3_cxs2_block));
    uint8_t scales_e5m3_avx2[8];

    dna_iq3_cxs2_quantize_row_avx2(weights, n_weights, blocks_avx2, scales_e5m3_avx2);

    // Verify scales match
    for (int k = 0; k < 8; ++k) {
        assert(scales_e5m3_scalar[k] == scales_e5m3_avx2[k]);
    }

    // Verify blocks match bit-for-bit
    int mismatches = 0;
    for (size_t l = 0; l < 8; ++l) {
        if (memcmp(blocks_scalar[l].bytes, blocks_avx2[l].bytes, 64) != 0) {
            mismatches++;
        }
    }
    printf("    AVX2 vs Scalar block match: %s (%d mismatches out of 8 blocks)\n",
           mismatches == 0 ? "PASSED (exact bit match)" : "FAILED", mismatches);
    assert(mismatches == 0);
    free(blocks_avx2);
#endif

    // 4. Test all-zero/silent subblocks
    printf("[4] Testing all-zero silence handling...\n");
    float * zeros = (float *) calloc(160, sizeof(float));
    dna_iq3_cxs2_block zero_block;
    uint8_t zero_scales[8];
    dna_iq3_cxs2_quantize_row(zeros, 160, &zero_block, zero_scales);
    for (int k = 0; k < 8; ++k) {
        assert(zero_scales[k] == 0);
    }
    printf("    Zero handling: PASSED\n");

    printf("[5] Testing IQ3_CXS2 sparse down tile accumulation...\n");
    test_down_tile_accum();
    printf("    Sparse down tile accumulation: PASSED\n");

    free(zeros);
    free(weights);
    free(dequant_scalar);
    free(blocks_scalar);

    printf("=== All IQ3_CXS2 tests passed successfully! ===\n");
    return 0;
}
