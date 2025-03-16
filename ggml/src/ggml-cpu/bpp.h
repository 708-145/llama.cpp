// g++ addlut_speedtest_mt.c -o addlut_speedtest_mt -O3

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Performs a Look-Up Table (LUT) lookup for a specified range of indices and sums the results into a static result vector.
 * 
 * @param indices_vector The vector containing indices for the LUT lookup.
 * @param random_LUT The Look-Up Table as a vector of floating point numbers.
 * @param start_index The starting index for the range of indices to process.
 * @param end_index The ending index for the range of indices to process.
 * @param thread_sum The vector where the accumulated sum of LUT values will be stored.
 */
void performLUTLookup(const uint16_t* indices_vector, const float* random_LUT, size_t start_index, size_t end_index, float* thread_sum);

void bpp_IQ4NL_F32_vecmul(const block_iq4_nl* iq4nl, float* invec, float* outvec, int vecsize, int from_row, int to_row);
void bpp_IQ4NL_F32_vecmul_simple(const block_iq4_nl* iq4nl, float* invec, float* outvec, int vecsize, int from_row, int to_row);

// Dummy print function declaration
void dummyPrint(void);
