#include "ggml-iq3-cxs2.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float rand_normal(float mean, float std) {
    float u = (float) rand() / (float) RAND_MAX;
    float v = (float) rand() / (float) RAND_MAX;
    if (u <= 1e-7f) u = 1e-7f;
    return mean + std * sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.1415926535f * v);
}

typedef struct {
    const char * name;
    float bpw;
    float rmse;
    float snr_db;
} quant_result_t;

static quant_result_t eval_snr(const char * name, float bpw, const float * orig, const float * deq, size_t n) {
    float sq_err = 0.0f;
    float sq_orig = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = orig[i] - deq[i];
        sq_err += d * d;
        sq_orig += orig[i] * orig[i];
    }
    float rmse = sqrtf(sq_err / (float) n);
    float snr = 10.0f * log10f(sq_orig / sq_err);
    quant_result_t r = { name, bpw, rmse, snr };
    return r;
}

int main(void) {
    printf("========================================================================\n");
    printf("   Comparison: IQ3_CXS2 vs. IQ3_XXS vs. IQ4_XS (Synthetic MLP Weights)  \n");
    printf("========================================================================\n");

    // Initialize GGML quant tables
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);
    ggml_quantize_init(GGML_TYPE_IQ4_XS);

    const size_t n = 1280 * 8; // 10,240 weights
    float * x = (float *) malloc(n * sizeof(float));

    // Distribution: Standard Normal (mu=0, sigma=0.04) with typical outliers
    srand(12345);
    for (size_t i = 0; i < n; ++i) {
        float w = rand_normal(0.0f, 0.04f);
        if (i % 64 == 0) w *= 3.0f; // occasional outlier
        x[i] = w;
    }

    // 1. IQ3_CXS2
    dna_iq3_cxs2_block * blocks_cxs2 = (dna_iq3_cxs2_block *) malloc((n / 160) * sizeof(dna_iq3_cxs2_block));
    uint8_t scales_e5m3[8];
    dna_iq3_cxs2_quantize_row(x, n, blocks_cxs2, scales_e5m3);

    float scales_f32[8];
    for (int k = 0; k < 8; ++k) scales_f32[k] = dna_e5m3_to_f32(scales_e5m3[k]);
    float * deq_cxs2 = (float *) malloc(n * sizeof(float));
    for (size_t l = 0; l < n / 160; ++l) {
        dna_iq3_cxs2_dequantize_block(&blocks_cxs2[l], scales_f32, deq_cxs2 + l * 160);
    }
    quant_result_t res_cxs2 = eval_snr("IQ3_CXS2 (Ours)", 3.20f, x, deq_cxs2, n);

    // 2. IQ3_XXS
    block_iq3_xxs * blocks_xxs = (block_iq3_xxs *) malloc((n / 256) * sizeof(block_iq3_xxs));
    quantize_row_iq3_xxs_ref(x, blocks_xxs, n);
    float * deq_xxs = (float *) malloc(n * sizeof(float));
    dequantize_row_iq3_xxs(blocks_xxs, deq_xxs, n);
    quant_result_t res_xxs = eval_snr("IQ3_XXS (ggml)", 3.06f, x, deq_xxs, n);

    // 3. IQ4_XS
    block_iq4_xs * blocks_xs = (block_iq4_xs *) malloc((n / 256) * sizeof(block_iq4_xs));
    quantize_row_iq4_xs_ref(x, blocks_xs, n);
    float * deq_xs = (float *) malloc(n * sizeof(float));
    dequantize_row_iq4_xs(blocks_xs, deq_xs, n);
    quant_result_t res_xs = eval_snr("IQ4_XS  (ggml)", 4.25f, x, deq_xs, n);

    printf("\nBenchmark Results on N = %zu weights:\n\n", n);
    printf("| Format        | Bits / Weight | RMSE      | SNR (dB)  |\n");
    printf("|---------------|---------------|-----------|-----------|\n");
    printf("| %-13s | %10.2f bpw | %9.6f | %6.2f dB |\n", res_xxs.name, res_xxs.bpw, res_xxs.rmse, res_xxs.snr_db);
    printf("| %-13s | %10.2f bpw | %9.6f | %6.2f dB |\n", res_cxs2.name, res_cxs2.bpw, res_cxs2.rmse, res_cxs2.snr_db);
    printf("| %-13s | %10.2f bpw | %9.6f | %6.2f dB |\n", res_xs.name, res_xs.bpw, res_xs.rmse, res_xs.snr_db);
    printf("\n");

    // Clean up
    free(x);
    free(blocks_cxs2);
    free(deq_cxs2);
    free(blocks_xxs);
    free(deq_xxs);
    free(blocks_xs);
    free(deq_xs);
    return 0;
}
