from huggingface_hub import hf_hub_download
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

repo_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
filename = "qwen1.5-0.5b-chat-f16.gguf"
local_model_path = os.path.join("models", filename)

print(f"Attempting to download {filename} from {repo_id} to {local_model_path}...")

try:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="models",
        local_dir_use_symlinks=False, # Make sure it's a full copy
    )
    print(f"Model downloaded to {local_model_path}")
except Exception as e:
    print(f"Failed to download model: {e}")
    # Create an empty file to indicate failure to download this specific model
    # This helps differentiate from other potential issues later.
    with open(os.path.join("models", "DOWNLOAD_FAILED_" + filename), "w") as f:
        f.write(str(e))
    print(f"Marked download failure for {filename}")
