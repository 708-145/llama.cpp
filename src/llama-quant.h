#pragma once

#include "ggml.h" // For ggml_type
#include "ggml-smarterquant-types.h" // For SmarterQuantTensorInfo
#include "llama.h"

#include <string>
#include <vector>
#include <map>
#include <unordered_map> // Added for std::unordered_map

// Forward declarations
struct llama_model_kv_override;

// Quantization types. Moved from quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

LLAMA_API size_t llama_tensor_quantize_smarter_blocks(
    const float * src_data,
    void * dst_data,
    const int64_t * ne,
    const SmarterQuantTensorInfo & sq_info,
    const float * imatrix_data,
    int nthread);

// SmarterQuant configuration map: tensor name -> SmarterQuantTensorInfo
using SmarterQuantConfig = std::map<std::string, SmarterQuantTensorInfo>;

// SmartQuant configuration map: tensor name -> ggml_type
using SmartQuantConfig = std::map<std::string, ggml_type>;

// Function to load SmarterQuant configuration from a JSON file
SmarterQuantConfig load_smarter_quant_config(const std::string & fname);

// Function to load SmartQuant configuration from a JSON file
SmartQuantConfig load_smart_quant_config(const std::string & fname);