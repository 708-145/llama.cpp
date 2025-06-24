#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "ggml-smarterquant-types.h" // Use the C-compatible definition

// Map from tensor name (std::string) to its SmarterQuantTensorInfo.
// This is the primary data structure holding the parsed smarter quantization configuration.
using SmarterQuantConfig = std::unordered_map<std::string, SmarterQuantTensorInfo>;

// Function to load and parse a smarter quantization JSON configuration file.
// The file should be a JSON object where keys are tensor names and values are
// 2-element arrays:
//   1. An array of 4 integers (ggml_type enums) for the first four 256-column blocks.
//   2. An array of integers for column permutation (can be empty).
// Example:
// {
//   "blk.0.attn_q.weight": [
//     [10, 11, 12, 13],  // compression_types (ggml_type values)
//     [0, 2, 1, 3, ...] // column_permutation
//   ]
// }
// Implemented in llama-quant.cpp.
SmarterQuantConfig load_smarter_quant_config(const std::string & fname);
