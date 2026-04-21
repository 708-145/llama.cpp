#pragma once

#include "ggml.h" // For ggml_type
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

// SmartQuant configuration map: tensor name -> ggml_type
using SmartQuantConfig = std::map<std::string, ggml_type>;

// Function to load SmartQuant configuration from a JSON file
SmartQuantConfig load_smart_quant_config(const std::string & fname);