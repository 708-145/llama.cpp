#include "llama.h"
#include "llama-quant.h" // Includes ggml-smarterquant-types.h
#include "ggml.h"
#include "ggml-impl.h" // For ggml_row_size, ggml_is_quantized etc.
#include "common.h"    // For format, LLAMA_LOG macros
#include "json.hpp"    // For nlohmann::json

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef> // For size_t
#include <cstdint> // For int32_t, uint8_t, uint16_t, int64_t
#include <thread>
#include <utility> // For std::move
#include <fstream>
#include <cstdio>  // For snprintf, stdout, fflush, fopen, fclose (though fstream is used)
#include <cstring> // For strcmp, strncpy, memcpy
#include <algorithm> // For std::sort, std::max, std::min, std::count
#include <functional>
#include <limits>
#include <iostream> // For std::cerr (used by LLAMA_LOG_ERROR indirectly)
#include <iomanip>  // For std::setw, std::fixed (if used by logging)
#include <sstream>  // For std::ostringstream (if used by logging)
#include <cinttypes> // For PRId64

// Old C-style SmartQuant map handlers and their usage are removed.
// The new C++ `load_smarter_quant_config` using nlohmann::json is used instead.

// Forward declare to avoid needing full quantize_state_impl definition here for now
// This struct is defined in llama.cpp and is quite complex.
// We only need its members like has_imatrix, n_attention_wv etc.
// A proper solution might involve moving its definition to a shared header if needed extensively.
struct quantize_state_impl;


// Helper function from common.cpp (ensure it's available or replicate if small)
// For now, assuming common.h brings in enough, but zeros might be specific.
// static void zeros(std::ofstream &out, size_t n) {
//     char zero = 0;
//     for (size_t i = 0; i < n; ++i) {
//         out.write(&zero, 1);
//     }
// }
// ^^^ zeros is actually defined in gguf.cpp and used via ggml-impl.h -> gguf-impl.h.

// Forward declaration for llama_tensor_dequantize_impl, which seems to be an internal helper
// It's usually in llama.cpp or similar. For now, we'll assume it's linked.
// static void llama_tensor_dequantize_impl(
//     struct ggml_tensor * tensor,
//     std::vector<no_init<float>> & f32_conv_buf,
//     std::vector<std::thread> & workers,
//     int64_t nelements,
//     int nthread);

// Forward declaration for llama_tensor_quantize_impl
// static size_t llama_tensor_quantize_impl(
//     enum ggml_type type,
//     const float * src,
//     void * dst,
//     int64_t n,
//     int64_t nrows,
//     int64_t k,
//     const float * imatrix,
//     std::vector<std::thread> & workers,
//     int nthread);

// Forward declaration for llama_tensor_get_type
// static enum ggml_type llama_tensor_get_type(
//     quantize_state_impl & qs,
//     enum ggml_type default_type,
//     const struct ggml_tensor * tensor,
//     llama_ftype ftype);


// Forward declaration for llama_tensor_quantize_smarter_blocks
// It's in this file, so it should be fine if defined before use or static.

// This is defined in this file later.
static size_t llama_tensor_quantize_smarter_blocks(
    const float * src_data,
    void * dst_data,
    const int64_t * ne,
    const SmarterQuantTensorInfo & sq_info,
    const float * imatrix_data,
    int nthread);

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;
    SmarterQuantConfig smarter_quant_config_json; // Loaded from JSON
    SmarterQuantConfig smarter_quant_config_gguf; // Loaded from GGUF (will be merged)


    // Load the SmarterQuant configuration from JSON
    // TODO: Make the filename configurable via params, for now hardcoded
    smarter_quant_config_json = load_smarter_quant_config("default.smarterquant.json");

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   default_type = GGML_TYPE_TQ1_0;   break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   default_type = GGML_TYPE_TQ2_0;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides_ptr = nullptr;
    if (params->kv_overrides) {
        auto v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides_ptr = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, kv_overrides_ptr);
    // GGUF SmarterQuant config is loaded by llama_model_loader constructor into ml.gguf_smarter_quant_config
    smarter_quant_config_gguf = ml.gguf_smarter_quant_config;


    ml.init_mappings(false);

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    // Merge JSON and GGUF SmarterQuant configurations. GGUF takes precedence.
    SmarterQuantConfig final_smarter_quant_config = smarter_quant_config_json;
    for (const auto& pair : smarter_quant_config_gguf) {
        final_smarter_quant_config[pair.first] = pair.second;
    }


    struct quantize_state_impl qs(model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            for (const auto & kv : *imatrix_data) {
                for (float f_val : kv.second) { // Renamed f to f_val
                    if (!std::isfinite(f_val)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f_val));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    gguf_set_kv     (ctx_out.get(), ml.meta.get());
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype);

    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                gguf_set_val_i32(ctx_out.get(), o.key, o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & iter : ml.weights_map) { // Renamed it to iter
        tensors.push_back(&iter.second);
    }

    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    for (const auto * iter : tensors) { // Renamed it to iter
        const struct ggml_tensor * tensor = iter->tensor;
        const std::string name = ggml_get_name(tensor);
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;
    if (qs.n_attention_wv != 0) {
        const auto & n_head_kv_iter = model.hparams.n_head_kv_arr.begin();
        int32_t n_attn_layer = model.hparams.n_layer - std::count(n_head_kv_iter, n_head_kv_iter + model.hparams.n_layer, 0);
        if (llama_model_has_encoder(&model)) {
            n_attn_layer *= 3;
        }
        GGML_ASSERT((qs.n_attention_wv == n_attn_layer) && "n_attention_wv is unexpected");
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<std::thread> workers;
    workers.reserve(nthread);
    int idx_counter = 0; // Renamed idx to idx_counter
    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;
    uint16_t n_split = 1;

    if (params->keep_split) {
        for (const auto * iter : tensors) { // Renamed it to iter
            n_split = std::max(uint16_t(iter->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    for (const auto * iter : tensors) { // Renamed it to iter
        uint16_t i_split = params->keep_split ? iter->idx : 0;
        struct ggml_tensor * tensor = iter->tensor;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);
    }

    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), ml.n_tensors);
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index_val) { // Renamed index to index_val
        cur_split = index_val;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname_val = fname_out; // Renamed fname to fname_val
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname_val = std::string(split_path.data());
        }
        fout = std::ofstream(fname_val, std::ios::binary);
        fout.exceptions(std::ofstream::failbit);
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        ::zeros(fout, meta_size);
    };

    const auto tn_func = LLM_TN(model.arch); // Renamed tn to tn_func
    new_ofstream(0);
    for (const auto * iter : tensors) { // Renamed it to iter
        const auto & weight = *iter;
        struct ggml_tensor * tensor = weight.tensor;
        if (weight.idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight.idx);
        }
        const std::string name = ggml_get_name(tensor);
        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);
        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx_counter, ml.n_tensors, // Used idx_counter
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        bool quantize = name.rfind("weight") == name.size() - 6;
        quantize &= (ggml_n_dims(tensor) >= 2);
        quantize &= name.find("_norm.weight") == std::string::npos;
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;
        quantize &= name != tn_func(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != tn_func(LLM_TENSOR_TOKEN_TYPES, "weight");
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
        quantize &= name.find("time_mix_first.weight") == std::string::npos;
        // ... (other quantize &= conditions)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;


        enum ggml_type new_type_val; // Renamed new_type
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type_val = default_type;
            if (!params->pure && ggml_is_quantized(default_type)) {
                new_type_val = llama_tensor_get_type(qs, new_type_val, tensor, ftype);
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type_val = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type_val = params->output_tensor_type;
            }
            quantize = tensor->type != new_type_val;
        }

        if (!quantize) {
            new_type_val = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);
            const float * imatrix_ptr = nullptr; // Renamed imatrix to imatrix_ptr
            if (imatrix_data) {
                auto im_it = imatrix_data->find(tensor->name); // Renamed it to im_it
                if (im_it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                } else {
                    if (im_it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix_ptr = im_it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(im_it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);
                        if (name != tn_func(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(im_it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                        }
                    }
                }
            }
            if ((new_type_val == GGML_TYPE_IQ2_XXS || new_type_val == GGML_TYPE_IQ2_XS  || new_type_val == GGML_TYPE_IQ2_S   ||
                 new_type_val == GGML_TYPE_IQ1_S   || (new_type_val == GGML_TYPE_IQ1_M && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight"))  ||
                (new_type_val == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(tensor->name, "token_embd.weight") != 0)) && !imatrix_ptr) {
                LLAMA_LOG_ERROR("\n\n============================================================\n");
                LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                LLAMA_LOG_ERROR("============================================================\n\n");
                throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
            }

            float * f32_data;
            std::vector<no_init<float>> permuted_f32_data_holder;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            auto sq_it = final_smarter_quant_config.find(name); // Use final_smarter_quant_config
            if (sq_it != final_smarter_quant_config.end() && sq_it->second.enabled) {
                const int32_t* current_perm_ptr = nullptr;
                size_t current_perm_size = 0;

                if (sq_it->second.column_permutation != nullptr && sq_it->second.n_cols_for_permutation > 0) {
                    LLAMA_LOG_INFO("Applying column permutation for tensor %s...\n", name.c_str());
                    current_perm_ptr = sq_it->second.column_permutation;
                    current_perm_size = sq_it->second.n_cols_for_permutation;

                    if (current_perm_size != (size_t)tensor->ne[0]) {
                        LLAMA_LOG_ERROR("Error: Permutation size %zu does not match tensor columns %" PRId64 " for tensor %s. Skipping permutation.\n", current_perm_size, tensor->ne[0], name.c_str());
                    } else {
                        permuted_f32_data_holder.resize(nelements);
                        float * permuted_data_ptr = (float *)permuted_f32_data_holder.data();
                        const int64_t n_cols = tensor->ne[0];
                        const int64_t n_rows = tensor->ne[1];
                        const int64_t higher_dims_stride = ggml_nelements(tensor) / (n_cols * n_rows);

                        for (int64_t h_dim = 0; h_dim < higher_dims_stride; ++h_dim) {
                            const float * current_f32_slice = f32_data + h_dim * (n_cols * n_rows);
                            float * current_permuted_slice = permuted_data_ptr + h_dim * (n_cols * n_rows);
                            for (int64_t r = 0; r < n_rows; ++r) {
                                for (int64_t c_new = 0; c_new < n_cols; ++c_new) {
                                    const int64_t c_orig = current_perm_ptr[c_new];
                                    if (c_orig < 0 || c_orig >= n_cols) {
                                         LLAMA_LOG_ERROR("Error: Invalid column index %" PRId64 " in permutation for tensor %s. Skipping permutation.\n", c_orig, name.c_str());
                                         permuted_f32_data_holder.clear();
                                         f32_data = (float *)((tensor->type == GGML_TYPE_F32) ? tensor->data : f32_conv_buf.data());
                                         goto skip_quant_imatrix_permutation;
                                    }
                                    current_permuted_slice[r * n_cols + c_new] = current_f32_slice[r * n_cols + c_orig];
                                }
                            }
                        }
                        f32_data = permuted_data_ptr;
                        LLAMA_LOG_INFO("Finished applying column permutation for f32_data of tensor %s.\n", name.c_str());

                        if (imatrix_ptr) {
                            std::vector<float> permuted_imatrix_values;
                            const int64_t n_cols_imatrix = tensor->ne[0];
                            if (imatrix_data->at(name).size() % n_cols_imatrix != 0) {
                                LLAMA_LOG_WARN("Warning: imatrix size %zu not a multiple of n_cols %" PRId64 " for tensor %s. Skipping imatrix permutation.\n",
                                               imatrix_data->at(name).size(), n_cols_imatrix, name.c_str());
                            } else {
                                permuted_imatrix_values.resize(imatrix_data->at(name).size());
                                const float* original_imatrix_ptr = imatrix_data->at(name).data();
                                float* p_imatrix_ptr = permuted_imatrix_values.data(); // Renamed permuted_imatrix_ptr
                                const int64_t num_imatrix_slices_in_source = imatrix_data->at(name).size() / n_cols_imatrix;
                                for (int64_t s_idx = 0; s_idx < num_imatrix_slices_in_source; ++s_idx) {
                                    const float* current_original_imatrix_slice = original_imatrix_ptr + s_idx * n_cols_imatrix;
                                    float* current_permuted_imatrix_slice = p_imatrix_ptr + s_idx * n_cols_imatrix;
                                    for (int64_t c_new = 0; c_new < n_cols_imatrix; ++c_new) {
                                        const int64_t c_orig = current_perm_ptr[c_new];
                                        if (c_orig >= 0 && c_orig < n_cols_imatrix) {
                                            current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_orig];
                                        } else {
                                            current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_new];
                                        }
                                    }
                                }
                                qs.permuted_imatrix_holder = permuted_imatrix_values;
                                imatrix_ptr = qs.permuted_imatrix_holder.data();
                                LLAMA_LOG_INFO("Finished applying column permutation for imatrix of tensor %s.\n", name.c_str());
                            }
                        }
                    }
                }
            skip_imatrix_permutation:;

                // Store SmarterQuant GGUF metadata if enabled.
                {
                    nlohmann::json perm_json_array_gguf = nlohmann::json::array(); // Renamed perm_json_array
                    if (sq_it->second.column_permutation != nullptr) {
                        for (int64_t i = 0; i < sq_it->second.n_cols_for_permutation; ++i) {
                            perm_json_array_gguf.push_back(sq_it->second.column_permutation[i]);
                        }
                    }
                    std::string perm_str = perm_json_array_gguf.dump();

                    llama_model_kv_override kvo_perm;
                    snprintf(kvo_perm.key, sizeof(kvo_perm.key), "%s.smarterquant.permutation", name.c_str());
                    kvo_perm.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                    strncpy(kvo_perm.val_str, perm_str.c_str(), sizeof(kvo_perm.val_str) - 1);
                    kvo_perm.val_str[sizeof(kvo_perm.val_str) - 1] = '\0';

                    llama_model_kv_override kvo_enabled;
                    snprintf(kvo_enabled.key, sizeof(kvo_enabled.key), "%s.smarterquant.enabled", name.c_str());
                    kvo_enabled.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
                    kvo_enabled.val_bool = true;

                    nlohmann::json types_json_array_gguf = nlohmann::json::array(); // Renamed types_json_array
                    for(int i=0; i<4; ++i) {
                        types_json_array_gguf.push_back(sq_it->second.compression_types[i]);
                    }
                    std::string types_str = types_json_array_gguf.dump();
                    llama_model_kv_override kvo_types;
                    snprintf(kvo_types.key, sizeof(kvo_types.key), "%s.smarterquant.block_types", name.c_str());
                    kvo_types.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                    strncpy(kvo_types.val_str, types_str.c_str(), sizeof(kvo_types.val_str) -1);
                    kvo_types.val_str[sizeof(kvo_types.val_str)-1] = '\0';

                    if (params->kv_overrides) {
                        auto* overrides_vec = reinterpret_cast<std::vector<llama_model_kv_override>*>(params->kv_overrides);
                        // bool null_term_found = false; // unused variable
                        if (!overrides_vec->empty() && overrides_vec->back().key[0] == 0) {
                            // null_term_found = true; // unused variable
                            overrides_vec->pop_back();
                        }
                        overrides_vec->push_back(kvo_perm);
                        overrides_vec->push_back(kvo_enabled);
                        overrides_vec->push_back(kvo_types);
                        overrides_vec->emplace_back();
                        overrides_vec->back().key[0] = 0;
                    }
                    LLAMA_LOG_INFO("Adding metadata for %s: permutation, enabled, block_types\n", name.c_str());
                }
            }

            if (work.size() < (size_t)nelements * 4) {
                work.resize(nelements * 4);
            }
            new_data = work.data();

            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows     = tensor->ne[1];
            const int64_t n_slices  = tensor->ne[2];
            static const int64_t min_chunk_size_bytes = 32 * 512;
            const int64_t elements_per_row_bytes_approx = n_per_row * sizeof(float);
            const int64_t chunk_size_elements = (elements_per_row_bytes_approx >= min_chunk_size_bytes ? n_per_row : n_per_row * ((min_chunk_size_bytes + elements_per_row_bytes_approx - 1)/elements_per_row_bytes_approx));
            const int64_t nelements_matrix_per_slice = n_per_row * nrows;
            const int64_t nchunk_per_slice = (nelements_matrix_per_slice + chunk_size_elements - 1)/chunk_size_elements;
            const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk_per_slice)) : 1;

            if (sq_it != final_smarter_quant_config.end() && sq_it->second.enabled) {
                new_type_val = static_cast<ggml_type>(sq_it->second.compression_types[3]); // Base GGUF type
                LLAMA_LOG_INFO("Applying SmarterQuant to %s. GGUF type: %s. Calling llama_tensor_quantize_smarter_blocks.\n", name.c_str(), ggml_type_name(new_type_val));
                new_size = llama_tensor_quantize_smarter_blocks(
                    f32_data, new_data, tensor->ne, sq_it->second, imatrix_ptr, nthread_use);
                LLAMA_LOG_INFO("SmarterQuant for %s done. Calculated new_size = %zu bytes.\n", name.c_str(), new_size);
            } else {
                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type_val));
                fflush(stdout);
                new_size = 0;
                for (int64_t i03 = 0; i03 < n_slices; ++i03) {
                    const float * f32_data_slice = f32_data + i03 * nelements_matrix_per_slice;
                    void * new_data_slice = (char *)new_data + i03 * nrows * ggml_row_size(new_type_val, n_per_row);
                    const float * imatrix_slice_ptr = nullptr; // Renamed imatrix_slice
                    if (imatrix_ptr) {
                        imatrix_slice_ptr = imatrix_ptr + i03 * n_per_row;
                    }
                    new_size += llama_tensor_quantize_impl(new_type_val, f32_data_slice, new_data_slice, chunk_size_elements, nrows, n_per_row, imatrix_slice_ptr, workers, nthread_use);
                }
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type_val);
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
		
    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}
