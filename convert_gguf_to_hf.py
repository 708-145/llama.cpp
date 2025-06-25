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
        "llama": "LlamaForCausalLM",
        "gptneox": "GPTNeoXForCausalLM",
        "gpt2": "GPT2LMHeadModel",
        "phi2": "PhiForCausalLM",
        "phi3": "Phi3ForCausalLM",
        "gemma": "GemmaForCausalLM",
        "gemma2": "Gemma2ForCausalLM",
        "starcoder2": "Starcoder2ForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        # Add more mappings as needed
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
        config_data["rms_norm_eps"] = get_gguf_float_val(gguf.Keys.LLAMA.LAYER_NORM_RMS_EPS, 1e-5)
        config_data["max_position_embeddings"] = get_gguf_int_val(gguf.Keys.LLAMA.CONTEXT_LENGTH, 2048)
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
        try:
            expected_shape = hf_model.state_dict(keep_vars=True)[hf_tensor_name].shape
            if tensor.shape != expected_shape:
                print(f"  Warning: Shape mismatch for {hf_tensor_name}. GGUF: {tensor.shape}, HF: {expected_shape}. Attempting reshape/view.")
                # This is a very basic attempt. More complex transpositions might be needed.
                try:
                    tensor = tensor.reshape(expected_shape)
                except RuntimeError as e:
                    print(f"    Error reshaping {hf_tensor_name}: {e}. Tensor will likely be incorrect.")
        except KeyError:
            print(f"  Warning: Tensor {hf_tensor_name} not found in initialized HF model's state_dict. This could be an issue with the tensor map or model config.")


        state_dict[hf_tensor_name] = tensor

    # Load the state dict into the model
    try:
        hf_model.load_state_dict(state_dict, assign=True, strict=False) # strict=False to allow for missing/extra keys for now
        print("Successfully loaded state_dict into the model (strict=False).")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("This could be due to shape mismatches or missing/unexpected tensors.")
        print("Run with --verbose and check tensor processing messages.")
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
