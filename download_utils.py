import os
import urllib.request

# Define the path to the conversion script and its dependencies
GGUF_PY_PATH = "./gguf-py/gguf.py"
GGUF_TENSOR_MAP_PATH = "./gguf-py/gguf_tensor_map.py"
GGUF_PY_DIR = "./gguf-py"

# URLs for downloading the scripts
GGUF_PY_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/gguf-py/gguf.py"
GGUF_TENSOR_MAP_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/gguf-py/gguf_tensor_map.py"

def download_file(url, filepath):
    """Downloads a file from a URL to a local path."""
    if not os.path.exists(os.path.dirname(filepath)) and os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath))
        print(f"Created directory {os.path.dirname(filepath)}")

    print(f"Downloading {filepath} from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Successfully downloaded {filepath}")
        # Check file size
        statinfo = os.stat(filepath)
        print(f"File size: {statinfo.st_size} bytes")
        if statinfo.st_size < 1000: # Check if file size is suspiciously small
             print(f"Warning: Downloaded file {filepath} is very small ({statinfo.st_size} bytes). Please check the URL and content.")
    except Exception as e:
        print(f"Error downloading {filepath}: {e}")
        return False
    return True

if __name__ == "__main__":
    if not download_file(GGUF_PY_URL, GGUF_PY_PATH):
        print(f"Failed to download {GGUF_PY_PATH}")
    if not download_file(GGUF_TENSOR_MAP_URL, GGUF_TENSOR_MAP_PATH):
        print(f"Failed to download {GGUF_TENSOR_MAP_PATH}")

    # Also redownload convert_hf_to_gguf.py for consistency and to ensure it's correct
    CONVERSION_SCRIPT_PATH = "./convert_hf_to_gguf.py"
    CONVERSION_SCRIPT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
    if not download_file(CONVERSION_SCRIPT_URL, CONVERSION_SCRIPT_PATH):
        print(f"Failed to download {CONVERSION_SCRIPT_PATH}")
    else:
        # Make the conversion script executable
        try:
            os.chmod(CONVERSION_SCRIPT_PATH, 0o755)
            print(f"Made {CONVERSION_SCRIPT_PATH} executable.")
        except Exception as e:
            print(f"Error making {CONVERSION_SCRIPT_PATH} executable: {e}")
