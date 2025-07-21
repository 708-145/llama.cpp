#include "ggml.h"
#include "llama.h"
#include "../src/llama-quant.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <thread>

int main() {
    const int64_t n_rows = 128;
    const int64_t n_cols = 1024;
    const int64_t ne[2] = {n_cols, n_rows};

    std::vector<float> src_data(n_rows * n_cols);
    for (int i = 0; i < n_rows * n_cols; ++i) {
        src_data[i] = i % 256;
    }

    SmarterQuantTensorInfo sq_info;
    sq_info.enabled = true;
    sq_info.compression_types[0] = GGML_TYPE_Q4_0;
    sq_info.compression_types[1] = GGML_TYPE_Q4_1;
    sq_info.compression_types[2] = GGML_TYPE_Q5_0;
    sq_info.compression_types[3] = GGML_TYPE_Q5_1;
    sq_info.column_permutation = nullptr;
    sq_info.n_cols_for_permutation = 0;

    std::vector<uint8_t> dst_data_single(n_rows * n_cols * 2);
    std::vector<uint8_t> dst_data_multi(n_rows * n_cols * 2);

    const int nthread_single = 1;
    const int nthread_multi = std::thread::hardware_concurrency();

    printf("Running with %d thread(s)...\n", nthread_single);
    size_t size_single = llama_tensor_quantize_smarter_blocks(src_data.data(), dst_data_single.data(), ne, sq_info, nullptr, nthread_single);

    printf("Running with %d thread(s)...\n", nthread_multi);
    size_t size_multi = llama_tensor_quantize_smarter_blocks(src_data.data(), dst_data_multi.data(), ne, sq_info, nullptr, nthread_multi);

    printf("Single-threaded size: %zu\n", size_single);
    printf("Multi-threaded size: %zu\n", size_multi);

    assert(size_single == size_multi);
    assert(dst_data_single == dst_data_multi);

    printf("Test passed!\n");

    return 0;
}
