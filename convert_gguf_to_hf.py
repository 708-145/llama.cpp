#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Make sure gguf-py is in the Python path
sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a GGUF model back to a Hugging Face transformer model.")
    parser.add_argument("gguf_model", type=Path, help="Path to the GGUF model file.")
    parser.add_argument("output_dir", type=Path, help="Path to save the Hugging Face model.")
    parser.add_argument("--tensor_map_json", type=Path, help="Optional path to the tensor name map JSON file. Defaults to <gguf_model_name>.json.")
    parser.add_argument("--hf_config_json", type=Path, help="Optional path to a Hugging Face config.json to use as a base. If not provided, a generic config will be attempted.")

    args = parser.parse_args()

    if not args.gguf_model.is_file():
        print(f"Error: GGUF model file not found: {args.gguf_model}")
        sys.exit(1)

    tensor_map_json_path = args.tensor_map_json
    if tensor_map_json_path is None:
        tensor_map_json_path = args.gguf_model.with_suffix(".json")

    if not tensor_map_json_path.is_file():
        print(f"Error: Tensor map JSON file not found: {tensor_map_json_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GGUF model from: {args.gguf_model}")
    gguf_reader = gguf.GGUFReader(args.gguf_model, "r")

    print(f"Loading tensor map from: {tensor_map_json_path}")
    with open(tensor_map_json_path, "r", encoding="utf-8") as f:
        tensor_map: dict[str, str] = json.load(f)

    # --- Basic Model Config Reconstruction ---
    # Try to get architecture from GGUF metadata
    arch_field = gguf_reader.get_field(gguf.Keys.General.ARCHITECTURE)
    model_arch_name = "unknown"
    if arch_field and arch_field.parts and len(arch_field.parts) > 0:
        model_arch_name = bytes(arch_field.parts[-1]).decode("utf-8")
        print(f"Found model architecture in GGUF: {model_arch_name}")
    else:
        print("Warning: Model architecture not found in GGUF metadata.")


    # Try to get model type from GGUF metadata for config.json
    # This is a heuristic and might need adjustment based on common GGUF metadata
    model_type_map = {
        "llama": "LlamaForCausalLM",  # Covers LLaMAForCausalLM, MistralForCausalLM, MixtralForCausalLM, InternLM3ForCausalLM
        "gptneox": "GPTNeoXForCausalLM",
        "gpt2": "GPT2LMHeadModel",
        "phi2": "PhiForCausalLM",  # Covers PhiForCausalLM
        "phi3": "Phi3ForCausalLM",
        "phimoe": "PhiMoEForCausalLM",
        "gemma": "GemmaForCausalLM",
        "gemma2": "Gemma2ForCausalLM",
        "gemma3": "Gemma3ForCausalLM", # Covers Gemma3ForConditionalGeneration
        "starcoder": "GPTBigCodeForCausalLM", # GGUF name is starcoder, HF is GPTBigCodeForCausalLM
        "starcoder2": "Starcoder2ForCausalLM",
        "qwen": "QWenLMHeadModel",
        "qwen2": "Qwen2ForCausalLM",
        "qwen2vl": "Qwen2VLForConditionalGeneration", # Covers Qwen2_5_VLForConditionalGeneration
        "qwen2moe": "Qwen2MoeForCausalLM",
        "bloom": "BloomForCausalLM", # Covers BloomModel
        "mpt": "MPTForCausalLM",
        "orion": "OrionForCausalLM",
        "baichuan": "BaichuanForCausalLM", # Covers BaiChuanForCausalLM
        "xverse": "XverseForCausalLM",
        "falcon": "FalconForCausalLM", # Covers RWForCausalLM
        "refact": "GPTRefactForCausalLM",
        "stablelm": "StableLmForCausalLM", # Covers StableLMEpochForCausalLM, LlavaStableLMEpochForCausalLM
        "deci": "DeciLMForCausalLM",
        "bitnet": "BitnetForCausalLM",
        "grok": "GrokForCausalLM",
        "dbrx": "DbrxForCausalLM",
        "minicpm": "MiniCPMForCausalLM",
        "minicpm3": "MiniCPM3ForCausalLM",
        "wavtokenizer_dec": "WavTokenizerDecModel", # Needs specific class if not a standard HF one
        "plamo": "PlamoForCausalLM",
        "codeshell": "CodeShellForCausalLM",
        "internlm2": "InternLM2ForCausalLM",
        "bert": "BertModel", # Covers BertForMaskedLM, CamembertModel, RobertaModel, XLMRobertaModel, XLMRobertaForSequenceClassification
        "nomic_bert": "NomicBertModel",
        "rwkv6": "Rwkv6ForCausalLM",
        "rwkv6qwen2": "RWKV6Qwen2ForCausalLM",
        "rwkv7": "Rwkv7ForCausalLM", # Covers RWKV7ForCausalLM
        "arwkv7": "RwkvHybridForCausalLM",
        "mamba": "MambaForCausalLM", # Covers MambaLMHeadModel, FalconMambaForCausalLM
        "command_r": "CohereForCausalLM",
        "cohere2": "Cohere2ForCausalLM",
        "olmo": "OlmoForCausalLM", # Covers OLMoForCausalLM
        "olmo2": "Olmo2ForCausalLM",
        "olmoe": "OlmoeForCausalLM",
        "jina_bert_v2": "JinaBertModel", # Covers JinaBertForMaskedLM
        "openelm": "OpenELMForCausalLM",
        "arctic": "ArcticForCausalLM",
        "deepseek": "DeepseekForCausalLM",
        "deepseek2": "DeepseekV2ForCausalLM", # Covers DeepseekV3ForCausalLM
        "plm": "PLMForCausalLM",
        "t5": "T5WithLMHeadModel", # Covers T5ForConditionalGeneration, MT5ForConditionalGeneration, UMT5ForConditionalGeneration
        "t5encoder": "T5EncoderModel",
        "jais": "JAISLMHeadModel",
        "chatglm": "GlmForCausalLM", # Covers ChatGLMModel, ChatGLMForConditionalGeneration
        "nemotron": "NemotronForCausalLM",
        "exaone": "ExaoneForCausalLM",
        "granite": "GraniteForCausalLM",
        "granite_moe": "GraniteMoeForCausalLM",
        "chameleon": "ChameleonForConditionalGeneration", # Covers ChameleonForCausalLM
    }
    hf_model_type = model_type_map.get(model_arch_name.lower(), "AutoModelForCausalLM")
    print(f"Inferred Hugging Face model type: {hf_model_type}")

    if args.hf_config_json and args.hf_config_json.is_file():
        print(f"Loading Hugging Face config from: {args.hf_config_json}")
        config = AutoConfig.from_pretrained(str(args.hf_config_json))
    else:
        print("Attempting to create a generic Hugging Face config.")
        # Create a basic config. This will likely need more fields from GGUF.
        config_data: dict[str, Any] = {"model_type": hf_model_type.lower().replace("forcausallm","").replace("lmheadmodel","")}

        def get_gguf_int_val(key: str, default: int | None = None) -> int | None:
            field = gguf_reader.get_field(key)
            if field and field.parts and len(field.parts) > 0:
                return field.parts[-1].tolist()[0] # Assuming uint32 or similar
            return default

        def get_gguf_float_val(key: str, default: float | None = None) -> float | None:
            field = gguf_reader.get_field(key)
            if field and field.parts and len(field.parts) > 0:
                return field.parts[-1].tolist()[0] # Assuming float32
            return default

        config_data["vocab_size"] = get_gguf_int_val(gguf.Keys.LLAMA.VOCAB_SIZE, 32000) # A common default
        config_data["hidden_size"] = get_gguf_int_val(gguf.Keys.LLAMA.EMBEDDING_LENGTH)
        config_data["intermediate_size"] = get_gguf_int_val(gguf.Keys.LLAMA.FEED_FORWARD_LENGTH)
        config_data["num_hidden_layers"] = get_gguf_int_val(gguf.Keys.LLAMA.BLOCK_COUNT)
        config_data["num_attention_heads"] = get_gguf_int_val(gguf.Keys.LLAMA.HEAD_COUNT)
        if (kv_heads := get_gguf_int_val(gguf.Keys.LLAMA.HEAD_COUNT_KV)) is not None:
            config_data["num_key_value_heads"] = kv_heads

        # Norm EPS
        if (rms_norm_eps := get_gguf_float_val(gguf.Keys.LLAMA.LAYER_NORM_RMS_EPS)) is not None:
            config_data["rms_norm_eps"] = rms_norm_eps
        if (layer_norm_eps := get_gguf_float_val(gguf.Keys.LLAMA.LAYER_NORM_EPS)) is not None:
            config_data["layer_norm_eps"] = layer_norm_eps

        config_data["max_position_embeddings"] = get_gguf_int_val(gguf.Keys.LLAMA.CONTEXT_LENGTH, 2048)

        # RoPE settings
        if (rope_theta := get_gguf_float_val(gguf.Keys.LLAMA.ROPE_FREQ_BASE)) is not None:
            config_data["rope_theta"] = rope_theta
        if (rope_scaling_factor := get_gguf_float_val(gguf.Keys.LLAMA.ROPE_SCALING_FACTOR)) is not None:
            # HF config usually expects a dictionary for rope_scaling
            # This is a basic attempt, might need model-specific handling
            config_data["rope_scaling"] = {"type": "linear", "factor": rope_scaling_factor} # Assuming linear if factor is present
            # Some models (e.g. Qwen2) might use 'yarn' and have 'original_max_position_embeddings'
            if gguf_reader.get_field(gguf.Keys.LLAMA.ROPE_SCALING_TYPE) and \
               bytes(gguf_reader.get_field(gguf.Keys.LLAMA.ROPE_SCALING_TYPE).parts[-1]).decode("utf-8").lower() == "yarn":
                config_data["rope_scaling"]["type"] = "yarn"
                if (orig_max_pos_emb := get_gguf_int_val(gguf.Keys.LLAMA.ROPE_SCALING_ORIG_CTX_LEN)) is not None:
                    config_data["rope_scaling"]["original_max_position_embeddings"] = orig_max_pos_emb

        # MoE parameters
        if (expert_count := get_gguf_int_val(gguf.Keys.LLAMA.EXPERT_COUNT)) is not None:
            config_data["num_local_experts"] = expert_count
        if (experts_used_count := get_gguf_int_val(gguf.Keys.LLAMA.EXPERT_USED_COUNT)) is not None:
            config_data["num_experts_per_tok"] = experts_used_count

        config_data["torch_dtype"] = "float16" # As per requirement
        config_data["architectures"] = [hf_model_type]

        # Remove None values
        config_data = {k: v for k, v in config_data.items() if v is not None}
        config = AutoConfig.from_dict(config_data)

    print("Initializing Hugging Face model with FP16 weights...")
    # Initialize model with meta device to avoid allocating large buffers immediately
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # --- Tensor Reconstruction ---
    state_dict = {}
    tensor_infos = {info.name: info for info in gguf_reader.tensors}

    print("Loading and mapping tensors...")
    for gguf_tensor_name, hf_tensor_name in tensor_map.items():
        if gguf_tensor_name not in tensor_infos:
            print(f"Warning: Tensor '{gguf_tensor_name}' (HF: '{hf_tensor_name}') not found in GGUF file. Skipping.")
            continue

        tensor_info = tensor_infos[gguf_tensor_name]
        print(f"  Processing GGUF tensor: {gguf_tensor_name} -> HF tensor: {hf_tensor_name} | Shape: {tensor_info.shape} | GGUF Dtype: {tensor_info.ggml_type.name}")

        # For simplicity, we assume GGUF stores FP16 or FP32 that can be cast to FP16.
        # Quantized types would require dequantization here.
        if tensor_info.ggml_type not in [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16]:
            print(f"Warning: Tensor '{gguf_tensor_name}' has type {tensor_info.ggml_type.name}, which is not directly F16/F32/BF16. Conversion might be lossy or incorrect. Only F16 is officially supported by this script.")
            # Attempt to load and convert anyway, hoping for the best for simple cases.
            # More robust handling would require dequantization logic from gguf.quants.

        # Load tensor data. gguf_reader.get_tensor loads and dequantizes if necessary for some types.
        data_np = tensor_info.data

        # Ensure data is in a format that can be converted to torch.float16
        if data_np.dtype == np.uint8: # Likely quantized, gguf_reader might not fully dequantize all types
            print(f"Warning: Tensor '{gguf_tensor_name}' is uint8. This indicates a quantized type. Attempting naive cast. Results may be incorrect.")
            # This is a placeholder. Proper dequantization is needed for Q types.
            # For simple Q8_0, this might involve scales. For now, we'll cast and hope.
            data_np = data_np.astype(np.float32) # Cast to float32 first before going to float16

        # Convert to PyTorch tensor and then to FP16
        tensor = torch.from_numpy(data_np).to(torch.float16)

        # Reshape if necessary (GGUF tensors might be stored transposed or with different dimension order)
        # This is a critical part and highly model-dependent.
        # For now, we assume direct mapping of shapes if possible.
        # More sophisticated logic might be needed here based on common GGUF conventions
        # or specific model architecture quirks.
        # TODO: Implement architecture-specific tensor transformations by reversing
        #       the logic in the corresponding convert_hf_to_gguf.py model classes.
        try:
            # It's better to get the expected shape from the uninitialized model's parameters
            # to avoid issues if a tensor is already (incorrectly) on a real device.
            expected_param = hf_model.get_parameter(hf_tensor_name)
            expected_shape = expected_param.shape

            if tensor.shape != expected_shape:
                print(f"  Warning: Shape mismatch for '{hf_tensor_name}'. GGUF: {tensor.shape}, HF: {expected_shape}.")
                # Attempt common transformations
                if tensor.squeeze().shape == expected_shape:
                    print(f"    Attempting to squeeze tensor '{hf_tensor_name}'.")
                    tensor = tensor.squeeze()
                elif tensor.T.shape == expected_shape:
                    print(f"    Attempting to transpose tensor '{hf_tensor_name}'.")
                    tensor = tensor.T
                elif len(tensor.shape) == len(expected_shape):
                    try:
                        print(f"    Attempting to reshape tensor '{hf_tensor_name}'.")
                        tensor = tensor.reshape(expected_shape)
                    except RuntimeError as e_reshape:
                        print(f"    Error reshaping '{hf_tensor_name}': {e_reshape}. Trying transpose then reshape.")
                        try:
                            tensor = tensor.T.reshape(expected_shape)
                            print(f"    Successfully transposed and reshaped '{hf_tensor_name}'.")
                        except RuntimeError as e_t_reshape:
                             print(f"    Error transposing and reshaping '{hf_tensor_name}': {e_t_reshape}. Tensor will likely be incorrect.")
                else:
                    print(f"    Cannot automatically fix shape for '{hf_tensor_name}'. Manual intervention may be needed based on model architecture.")
        except KeyError:
            print(f"  Warning: Tensor '{hf_tensor_name}' not found in initialized HF model's state_dict. This could be an issue with the tensor map or model config.")
        except AttributeError: # For non-parameter tensors if any were to be mapped
             print(f"  Info: '{hf_tensor_name}' is not a parameter in the HF model, skipping shape check.")


        state_dict[hf_tensor_name] = tensor

    # Load the state dict into the model
    # Using assign=True allows loading onto the 'meta' device initially
    try:
        errors = hf_model.load_state_dict(state_dict, assign=True, strict=False)
        if errors.missing_keys:
            print(f"Warning: Missing keys in state_dict: {errors.missing_keys}")
        if errors.unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {errors.unexpected_keys}")
        print("Successfully loaded state_dict into the model.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("This could be due to shape mismatches or missing/unexpected tensors.")
        print("Run with --verbose and check tensor processing messages for clues.")
        sys.exit(1)

    # Materialize the model from 'meta' device to CPU (or other target device)
    # This is where actual memory allocation happens.
    print("Materializing model...")
    try:
        hf_model.to_empty(device="cpu") # First move to empty on target device
        hf_model.load_state_dict(state_dict) # Then load the actual data
    except Exception as e:
        print(f"Error materializing model: {e}")
        sys.exit(1)

    print(f"Saving Hugging Face model to: {args.output_dir}")
    hf_model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    # Save tokenizer files if they exist in the GGUF source (heuristic)
    # GGUF usually stores vocab directly, but HF needs tokenizer.json, tokenizer_config.json etc.
    # This part is tricky as GGUF doesn't store these files directly.
    # We might need to reconstruct them or copy from an original HF model dir if available.
    # For now, we'll skip tokenizer reconstruction as it's complex.
    print("Tokenizer reconstruction is not yet implemented in this script.")
    print("You may need to manually add tokenizer files (tokenizer.json, etc.) to the output directory from a compatible Hugging Face model.")

    print("Conversion complete.")

if __name__ == "__main__":
    main()
