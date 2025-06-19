import time
import os
import signal
import sys
from zonos.model import Zonos
from pathlib import Path

# Function to handle signals properly
def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def check_and_download_models():
    """Checks if models are cached and downloads them if necessary."""
    models_to_check = {
        "transformer": "Zyphra/Zonos-v0.1-transformer",
        # "hybrid": "Zyphra/Zonos-v0.1-hybrid" # Uncomment if you want hybrid too
    }
    # Default Hugging Face cache directory
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface/hub"))
    print(f"Using Hugging Face cache directory: {cache_dir}")

    all_models_present = True
    for model_type, model_id in models_to_check.items():
        # Construct expected path based on Hugging Face's caching structure
        # This might need adjustment if HF changes its structure
        model_cache_path = cache_dir / f"models--{model_id.replace('/', '--')}"
        print(f"Checking for {model_type} model at: {model_cache_path}")
        if not model_cache_path.exists() or not any(model_cache_path.iterdir()):
            print(f"{model_type.capitalize()} model not found or cache directory empty. Downloading...")
            all_models_present = False
            try:
                Zonos.from_pretrained(model_id, force_download=True)
                print(f"{model_type.capitalize()} model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading {model_type} model ({model_id}): {e}")
                # Decide if you want to exit or continue without the model
                # sys.exit(1)
        else:
            print(f"{model_type.capitalize()} model found in cache.")

    if all_models_present:
        print("All required Zonos models are present in the cache.")

# --- Main Execution ---
print("Starting Zonos service...")

# Check and download models on startup
check_and_download_models()

print("Zonos service is ready and waiting for commands.")
print("Use 'docker compose exec zonos python3 zonos_generate.py ...' to run TTS.")

# Keep container running
while True:
    try:
        time.sleep(60)  # Just sleep and keep container running
    except KeyboardInterrupt:
        print("\nShutting down...")
        break
