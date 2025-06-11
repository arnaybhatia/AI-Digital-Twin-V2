import time
import os
import signal
import sys
from pathlib import Path

# Function to handle signals properly
def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def check_models():
    """Check if SadTalker models are present."""
    checkpoints_dir = Path("/app/checkpoints")
    gfpgan_dir = Path("/app/gfpgan/weights")
    
    print(f"Checking SadTalker models...")
    
    # Check if model directories exist and have files
    model_dirs = [checkpoints_dir, gfpgan_dir]
    all_present = True
    
    for model_dir in model_dirs:
        if model_dir.exists() and any(model_dir.iterdir()):
            print(f"Models found in: {model_dir}")
        else:
            print(f"Models missing in: {model_dir}")
            all_present = False
    
    if all_present:
        print("All SadTalker models are present.")
    else:
        print("Some models may be missing, but continuing...")
    
    return all_present

# --- Main Execution ---
print("Starting SadTalker service...")

# Check models on startup
check_models()

print("SadTalker service is ready and waiting for commands.")
print("Use 'docker compose exec sadtalker python inference.py ...' to generate talking heads.")

# Keep container running
while True:
    try:
        time.sleep(60)  # Just sleep and keep container running
    except KeyboardInterrupt:
        print("\nShutting down...")
        break
