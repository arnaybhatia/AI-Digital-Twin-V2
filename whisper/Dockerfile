FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/whisper

# Install system dependencies - using python3 instead of python3.9 for compatibility
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Whisper inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Create whisper_transcribe.py
RUN echo 'import argparse\n\
import whisper\n\
import torch\n\
\n\
def transcribe(file, output):\n\
    # Check for GPU availability and use it if available\n\
    device = "cuda" if torch.cuda.is_available() else "cpu"\n\
    print(f"Using device: {device}")\n\
    model = whisper.load_model("medium", device=device)\n\
    result = model.transcribe(file)\n\
    with open(output, "w") as f:\n\
        f.write(result["text"])\n\
\n\
if __name__ == "__main__":\n\
    parser = argparse.ArgumentParser()\n\
    parser.add_argument("--file", required=True)\n\
    parser.add_argument("--output", required=True)\n\
    args = parser.parse_args()\n\
    transcribe(args.file, args.output)' > whisper_transcribe.py

# Set entrypoint to use the virtual environment
ENTRYPOINT ["/opt/venv/bin/python3"]

# Create a simple script that keeps the container running and accepts commands
RUN echo 'import time\n\
import os\n\
import signal\n\
import sys\n\
import threading\n\
import subprocess\n\
import whisper\n\
import torch\n\
\n\
# Function to handle signals properly\n\
def signal_handler(sig, frame):\n\
    print("\\nShutting down gracefully...")\n\
    sys.exit(0)\n\
\n\
# Register signal handlers\n\
signal.signal(signal.SIGTERM, signal_handler)\n\
signal.signal(signal.SIGINT, signal_handler)\n\
\n\
# Pre-load the model to save time during inference\n\
print("Loading Whisper model...")\n\
try:\n\
    device = "cuda" if torch.cuda.is_available() else "cpu"\n\
    model = whisper.load_model("medium", device=device)\n\
    print(f"Model loaded successfully on {device}")\n\
except Exception as e:\n\
    print(f"Error loading model: {str(e)}")\n\
    model = None\n\
\n\
print("Whisper service is ready and waiting for commands.")\n\
print("Use docker exec to run transcription commands.")\n\
\n\
# Keep container running\n\
while True:\n\
    try:\n\
        time.sleep(60)  # Just sleep and keep container running\n\
    except KeyboardInterrupt:\n\
        print("\\nShutting down...")\n\
        break\n\
' > server.py

# Command to run the server
CMD ["server.py"]