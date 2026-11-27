#include "ggml-iq3-cxs2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "ggml-cpu.h"

static double bench_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        abort(); \
    } \
} while (0)

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
static int test_cpu_has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#elif defined(__AVX2__)
static int test_cpu_has_avx2(void) {
    return 1;
}
#endif

static float rand_normal(float mean, float std) {
    // Box-Muller transform
    float u = (float) rand() / (float) RAND_MAX;
    float v = (float) rand() / (float) RAND_MAX;
    if (u <= 1e-7f) u = 1e-7f;
    return mean + std * sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.1415926535f * v);
}

static void test_ggml_iq3_cxs2_row_layout(void) {
    const int64_t n_weights = 1536;
    const size_t row_size = 10 * DNA_IQ3_CXS2_LINE_BYTES + 8;
    float weights[1536];
    float decoded[1536];
    uint8_t row[10 * DNA_IQ3_CXS2_LINE_BYTES + 8];
    uint8_t q8_row[48 * 34];
    float input[1536];
    for (size_t i = 0; i < 1536; ++i) {
        weights[i] = 0.1f * sinf((float) i * 0.031f);
        input[i] = 0.2f * cosf((float) i * 0.017f);
    }

    CHECK(ggml_row_size(GGML_TYPE_IQ3_CXS2, n_weights) == row_size);
    ggml_quantize_row_iq3_cxs2(weights, row, n_weights);
    ggml_dequantize_row_iq3_cxs2(row, decoded, n_weights);
    for (size_t i = 0; i < 1536; ++i) {
        CHECK(isfinite(decoded[i]));
    }

    const struct ggml_type_traits_cpu * cxs2_traits =
        ggml_get_type_traits_cpu(GGML_TYPE_IQ3_CXS2);
    const struct ggml_type_traits_cpu * q8_traits =
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    CHECK(cxs2_traits != NULL && cxs2_traits->vec_dot != NULL);
    CHECK(q8_traits != NULL && q8_traits->from_float != NULL);
    q8_traits->from_float(input, q8_row, n_weights);
    float actual = 0.0f;
    cxs2_traits->vec_dot(
        (int) n_weights, &actual, 0, row, 0, q8_row, 0, 1);
    float expected = 0.0f;
    for (size_t block = 0; block < 48; ++block) {
        ggml_fp16_t d;
        memcpy(&d, q8_row + block * 34, sizeof(d));
        const float scale = ggml_fp16_to_fp32(d);
        const int8_t * qs = (const int8_t *) (q8_row + block * 34 + sizeof(d));
        for (size_t i = 0; i < 32; ++i) {
            expected += decoded[block * 32 + i] * scale * (float) qs[i];
        }
    }
    CHECK(fabsf(actual - expected) < 1e-3f);

    struct ggml_init_params params = {
        .mem_size = 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != NULL);
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_CXS2, n_weights, 2);
    CHECK(tensor->nb[1] == row_size);
    CHECK(ggml_nbytes(tensor) == row_size * 2);
    CHECK(ggml_is_contiguous(tensor));
    CHECK(ggml_is_contiguously_allocated(tensor));
    ggml_free(ctx);
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
            CHECK(fabsf(actual[i] - expected[i]) < 1e-6f);
        }

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (test_cpu_has_avx2()) {
            float actual_avx2[DNA_IQ3_CXS2_WEIGHTS_PER_LINE];
            for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
                actual_avx2[i] = (float) ((int) i - 80) * 0.002f;
            }
            dna_iq3_cxs2_down_tile_accum_avx2(
                actual_avx2, active, sizeof(active) / sizeof(active[0]), tile,
                N_BLOCKS, blocks, scales_e5m3, n_subblocks);
            CHECK(memcmp(actual, actual_avx2, sizeof(actual)) == 0);
            for (size_t i = 0; i < DNA_IQ3_CXS2_WEIGHTS_PER_LINE; ++i) {
                CHECK(fabsf(actual_avx2[i] - expected[i]) < 1e-6f);
            }
        }
#endif
    }
}

static void test_sparse_gate_up_gemv(void) {
    enum {
        N_ROWS = 7,
        N_INPUTS = 320,
        N_LINES = N_INPUTS / DNA_IQ3_CXS2_WEIGHTS_PER_LINE,
        N_Q8_BLOCKS = N_INPUTS / 32,
        N_OUTPUTS = 8,
        ACTIVE_COUNT = 5,
    };

    float weights[N_ROWS * N_INPUTS];
    dna_iq3_cxs2_block blocks[N_ROWS * N_LINES];
    uint8_t scales_e5m3[N_ROWS * 8];
    int8_t x_q8[N_INPUTS];
    float x_scales[N_Q8_BLOCKS];
    float x_f32[N_INPUTS];
    dna_iq3_cxs2_sparse_row rows[ACTIVE_COUNT];
    float expected[N_OUTPUTS];
    float actual[N_OUTPUTS];

    for (size_t i = 0; i < N_INPUTS; ++i) {
        x_q8[i] = (int8_t) ((int) ((i * 17 + i / 7) % 101) - 50);
        x_f32[i] = (float) x_q8[i] * 0.125f;
    }
    for (size_t i = 0; i < N_Q8_BLOCKS; ++i) {
        x_scales[i] = 0.125f;
    }

    for (size_t row = 0; row < N_ROWS; ++row) {
        for (size_t i = 0; i < N_INPUTS; ++i) {
            weights[row * N_INPUTS + i] =
                ((row * 13 + i * 7) % 2) ? -3.375f : 3.375f;
        }
        dna_iq3_cxs2_quantize_row(
            weights + row * N_INPUTS,
            N_INPUTS,
            blocks + row * N_LINES,
            scales_e5m3 + row * 8);

        for (size_t k = 0; k < 8; ++k) {
            CHECK(dna_e5m3_to_f32(scales_e5m3[row * 8 + k]) == 0.125f);
        }
    }

    const size_t source_rows[ACTIVE_COUNT] = { 4, 0, 6, 2, 1 };
    const size_t output_rows[ACTIVE_COUNT] = { 5, 1, 7, 2, 0 };
    for (size_t r = 0; r < ACTIVE_COUNT; ++r) {
        rows[r].output_row = output_rows[r];
        rows[r].blocks = blocks + source_rows[r] * N_LINES;
        rows[r].scales_e5m3 = scales_e5m3 + source_rows[r] * 8;
    }

    for (size_t i = 0; i < N_OUTPUTS; ++i) {
        expected[i] = 123.0f;
        actual[i] = 123.0f;
    }

    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    CHECK(traits != NULL && traits->vec_dot != NULL);
    for (size_t r = 0; r < ACTIVE_COUNT; ++r) {
        float dot = 0.0f;
        traits->vec_dot(
            N_INPUTS,
            &dot, 0,
            (const char *) (weights + source_rows[r] * N_INPUTS), 0,
            (const char *) x_f32, 0,
            1);
        expected[output_rows[r]] = dot;
    }

    dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
        x_q8, x_scales, N_INPUTS, rows, ACTIVE_COUNT, actual);
    for (size_t r = 0; r < ACTIVE_COUNT; ++r) {
        const size_t out = output_rows[r];
        CHECK(fabsf(actual[out] - expected[out]) < 1e-3f);
    }
    CHECK(actual[3] == 123.0f);
    CHECK(actual[4] == 123.0f);
    CHECK(actual[6] == 123.0f);

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (test_cpu_has_avx2()) {
        float actual_avx2[N_OUTPUTS];
        for (size_t i = 0; i < N_OUTPUTS; ++i) {
            actual_avx2[i] = 123.0f;
        }
        dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx2(
            x_q8, x_scales, N_INPUTS, rows, ACTIVE_COUNT, actual_avx2);
        for (size_t r = 0; r < ACTIVE_COUNT; ++r) {
            const size_t out = output_rows[r];
            CHECK(fabsf(actual[out] - actual_avx2[out]) < 1e-4f);
            CHECK(fabsf(actual_avx2[out] - expected[out]) < 1e-3f);
        }
        CHECK(actual_avx2[3] == 123.0f);
        CHECK(actual_avx2[4] == 123.0f);
        CHECK(actual_avx2[6] == 123.0f);
    }
#endif
}

static void test_gemv_benchmark(void) {
    // n_ff=4096 neurons, n_embd=4096 input dims (padded to 4160 = 26 * 160)
    enum {
        N_FF        = 4096,
        N_EMBD_RAW  = 4096,
        N_LINES     = (N_EMBD_RAW + DNA_IQ3_CXS2_WEIGHTS_PER_LINE - 1)
                        / DNA_IQ3_CXS2_WEIGHTS_PER_LINE,  // 26
        N_EMBD      = N_LINES * DNA_IQ3_CXS2_WEIGHTS_PER_LINE,  // 4160
        N_Q8_BLOCKS = N_EMBD / 32,
        N_ITERS     = 200,
    };

    // allocate quantized gate/up weight matrix
    dna_iq3_cxs2_block * blocks = (dna_iq3_cxs2_block *)
        malloc((size_t)N_FF * N_LINES * sizeof(dna_iq3_cxs2_block));
    uint8_t * scales = (uint8_t *) malloc((size_t)N_FF * 8);
    float * weights_f32 = (float *) malloc((size_t)N_FF * N_EMBD * sizeof(float));
    float * output = (float *) malloc((size_t)N_FF * sizeof(float));
    int8_t * x_q8 = (int8_t *) malloc((size_t)N_EMBD);
    float * x_scales = (float *) malloc((size_t)N_Q8_BLOCKS * sizeof(float));
    float * x_f32 = (float *) malloc((size_t)N_EMBD * sizeof(float));
    dna_iq3_cxs2_sparse_row * rows = (dna_iq3_cxs2_sparse_row *)
        malloc((size_t)N_FF * sizeof(dna_iq3_cxs2_sparse_row));

    if (!blocks || !scales || !weights_f32 || !output ||
            !x_q8 || !x_scales || !x_f32 || !rows) {
        printf("    benchmark: allocation failed, skipping\n");
        goto bench_done;
    }

    // fill input Q8_0
    for (int i = 0; i < N_EMBD; ++i) {
        x_q8[i] = (int8_t) ((i * 7 + 3) % 127 - 63);
        x_f32[i] = (float) x_q8[i] * 0.0625f;
    }
    for (int b = 0; b < N_Q8_BLOCKS; ++b) {
        x_scales[b] = 0.0625f;
    }

    // quantize all N_FF rows
    for (int r = 0; r < N_FF; ++r) {
        for (int i = 0; i < N_EMBD; ++i) {
            weights_f32[(size_t)r * N_EMBD + i] =
                ((r * 13 + i * 7) % 2) ? -0.08f : 0.08f;
        }
        dna_iq3_cxs2_quantize_row(
            weights_f32 + (size_t)r * N_EMBD,
            N_EMBD,
            blocks + (size_t)r * N_LINES,
            scales + (size_t)r * 8);
    }

    // set up all rows array (identity mapping for benchmark)
    for (int r = 0; r < N_FF; ++r) {
        rows[r].output_row = (size_t)r;
        rows[r].blocks = blocks + (size_t)r * N_LINES;
        rows[r].scales_e5m3 = scales + (size_t)r * 8;
    }

    const struct ggml_type_traits_cpu * f32_traits =
        ggml_get_type_traits_cpu(GGML_TYPE_F32);

    // sparsity fractions to sweep
    const float fractions[]  = { 0.01f, 0.05f, 0.10f, 0.20f, 0.50f };
    const int   n_fractions  = (int)(sizeof(fractions) / sizeof(fractions[0]));

    int has_avx2 = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    has_avx2 = test_cpu_has_avx2();
#endif

    printf("    %6s  %7s  %10s  %10s  %10s\n",
           "active", "rows", "F32-ref ns", "scalar ns", has_avx2 ? "avx2 ns" : "(no avx2)");
    printf("    %-6s  %-7s  %-10s  %-10s  %-10s\n",
           "------", "-------", "----------", "---------", "---------");

    for (int fi = 0; fi < n_fractions; ++fi) {
        int active_count = (int)(fractions[fi] * N_FF + 0.5f);
        if (active_count < 1) active_count = 1;
        if (active_count > N_FF) active_count = N_FF;

        // --- F32 single-row reference ---
        double t0 = bench_now_ns();
        for (int iter = 0; iter < N_ITERS; ++iter) {
            for (int r = 0; r < active_count; ++r) {
                float dot = 0.0f;
                f32_traits->vec_dot(
                    N_EMBD, &dot, 0,
                    (const char *)(weights_f32 + (size_t)r * N_EMBD), 0,
                    (const char *)x_f32, 0, 1);
                output[r] = dot;
            }
        }
        double t_f32 = (bench_now_ns() - t0) / N_ITERS;

        // --- scalar MR=4 ---
        for (int r = 0; r < active_count; ++r) output[r] = 0.0f;
        double t1 = bench_now_ns();
        for (int iter = 0; iter < N_ITERS; ++iter) {
            for (int r = 0; r < active_count; ++r) output[r] = 0.0f;
            dna_iq3_cxs2_gemv_mr4_indirect_q8_0(
                x_q8, x_scales, N_EMBD, rows, (size_t)active_count, output);
        }
        double t_scalar = (bench_now_ns() - t1) / N_ITERS;

        // --- AVX2 MR=4 ---
        double t_avx2 = 0.0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (has_avx2) {
            for (int r = 0; r < active_count; ++r) output[r] = 0.0f;
            double t2 = bench_now_ns();
            for (int iter = 0; iter < N_ITERS; ++iter) {
                for (int r = 0; r < active_count; ++r) output[r] = 0.0f;
                dna_iq3_cxs2_gemv_mr4_indirect_q8_0_avx2(
                    x_q8, x_scales, N_EMBD, rows, (size_t)active_count, output);
            }
            t_avx2 = (bench_now_ns() - t2) / N_ITERS;
        }
#endif

        printf("    %5.0f%%  %7d  %10.0f  %9.0f  %9.0f\n",
               fractions[fi] * 100.0f, active_count,
               t_f32, t_scalar, t_avx2);
    }

bench_done:
    free(rows);
    free(x_f32);
    free(x_scales);
    free(x_q8);
    free(output);
    free(weights_f32);
    free(scales);
    free(blocks);
}

int main(void) {
    ggml_cpu_init();

    printf("=== Testing IQ3_CXS2 Quantization ===\n");
    test_ggml_iq3_cxs2_row_layout();

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
        CHECK(rel_err < 0.15f);
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
    CHECK(snr > 10.0f); // 3-bit quant with non-linear codebook typically achieves 12-18 dB SNR

#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    // 3. Test AVX2 path equivalence
    printf("[3] Testing AVX2 quantization path...\n");
    dna_iq3_cxs2_block * blocks_avx2 = (dna_iq3_cxs2_block *) malloc(8 * sizeof(dna_iq3_cxs2_block));
    uint8_t scales_e5m3_avx2[8];

    if (test_cpu_has_avx2()) {
        dna_iq3_cxs2_quantize_row_avx2(weights, n_weights, blocks_avx2, scales_e5m3_avx2);

        for (int k = 0; k < 8; ++k) {
            CHECK(scales_e5m3_scalar[k] == scales_e5m3_avx2[k]);
        }
        for (size_t l = 0; l < 8; ++l) {
            CHECK(memcmp(blocks_scalar[l].bytes, blocks_avx2[l].bytes, 64) == 0);
        }
        printf("    AVX2 quantization: bit-exact\n");
    } else {
        printf("    AVX2 quantization: skipped (CPU does not support AVX2)\n");
    }
    free(blocks_avx2);
#endif

    // 4. Test all-zero/silent subblocks
    printf("[4] Testing all-zero silence handling...\n");
    float * zeros = (float *) calloc(160, sizeof(float));
    dna_iq3_cxs2_block zero_block;
    uint8_t zero_scales[8];
    dna_iq3_cxs2_quantize_row(zeros, 160, &zero_block, zero_scales);
    for (int k = 0; k < 8; ++k) {
        CHECK(zero_scales[k] == 0);
    }
#if defined(__AVX2__) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (test_cpu_has_avx2()) {
        dna_iq3_cxs2_block zero_block_avx2;
        uint8_t zero_scales_avx2[8];
        dna_iq3_cxs2_quantize_row_avx2(zeros, 160, &zero_block_avx2, zero_scales_avx2);
        CHECK(memcmp(zero_block.bytes, zero_block_avx2.bytes, sizeof(zero_block.bytes)) == 0);
        CHECK(memcmp(zero_scales, zero_scales_avx2, sizeof(zero_scales)) == 0);
    }
#endif
    printf("    Zero handling: PASSED\n");

    printf("[5] Testing IQ3_CXS2 sparse down tile accumulation...\n");
    test_down_tile_accum();
    printf("    Sparse down tile accumulation: PASSED\n");

    printf("[6] Testing IQ3_CXS2 sparse gate/up MR=4 GEMV...\n");
    test_sparse_gate_up_gemv();
    printf("    Sparse gate/up MR=4 GEMV: PASSED\n");

    free(zeros);
    free(weights);
    free(dequant_scalar);
    free(blocks_scalar);

    printf("[7] Benchmarking MR=4 vs single-row F32 GEMV (n_ff=4096, n_embd=4096)...\n");
    test_gemv_benchmark();
    printf("    Benchmark: DONE\n\n");

    printf("=== All IQ3_CXS2 tests passed successfully! ===\n");
    return 0;
}
