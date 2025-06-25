from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import os

def create_and_save_model(model_dir="hf_test_model_original"):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Using a small, real model for better testing

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Load a small configuration (e.g., Llama-like)
    # For simplicity in testing, we'll use a known small model's config as a base
    # and potentially make it even smaller if needed, though TinyLlama is already quite small.
    try:
        config = AutoConfig.from_pretrained(model_name)
        # Optionally, make it smaller for faster testing if TinyLlama is too big for quick iteration
        # config.num_hidden_layers = 2
        # config.hidden_size = 128
        # config.intermediate_size = 256
        # config.num_attention_heads = 4
        # config.num_key_value_heads = 2 # For GQA

        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        config.save_pretrained(model_dir) # Explicitly save config too

        print(f"Hugging Face model and tokenizer saved to ./{model_dir}")
        print(f"Config used: {config}")

    except Exception as e:
        print(f"Error creating or saving model: {e}")
        # Fallback to a very minimal manual config if TinyLlama fails (e.g. network issues)
        print("Falling back to a dummy manual config if TinyLlama download failed.")
        config_dict = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 64, # Minimal
            "intermediate_size": 128, # Minimal
            "num_hidden_layers": 1, # Minimal
            "num_attention_heads": 2, # Minimal
            "num_key_value_heads": 1, # Minimal GQA
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 256, # Minimal
            "torch_dtype": "float16",
            "architectures": ["LlamaForCausalLM"]
        }
        config = AutoConfig.from_dict(config_dict)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

        # Create a dummy tokenizer.model for sentencepiece
        # and a basic tokenizer_config.json
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "tokenizer.model"), "w") as f:
            # A minimal valid sentencepiece model file is complex to create on the fly.
            # For the purpose of GGUF conversion, often the vocab from config is enough,
            # or a dummy file might pass initial checks if vocab is handled by gguf-py's LlamaHfVocab.
            # This is a placeholder and might cause issues with `convert_hf_to_gguf.py` if it
            # strictly requires a valid SentencePieceProcessor readable model.
            f.write("") # This is NOT a valid SPM model.
            print("Saved a DUMMY tokenizer.model. This might not be sufficient for GGUF conversion's vocab part.")

        tokenizer_config = {
            "model_type": "llama",
            "tokenizer_class": "LlamaTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>", # Often same as unk or a specific pad token
        }
        with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
            import json
            json.dump(tokenizer_config, f)

        # We need a dummy vocab file for LlamaHfVocab to not fail immediately
        # It expects added_tokens.json usually.
        with open(os.path.join(model_dir, "special_tokens_map.json"), "w") as f:
            import json
            json.dump({
                "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"
            }, f)

        model.save_pretrained(model_dir)
        config.save_pretrained(model_dir)
        print(f"Fallback dummy Hugging Face model and tokenizer stubs saved to ./{model_dir}")
        print(f"Config used: {config}")


if __name__ == "__main__":
    create_and_save_model()
