FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/kdtalker

# Install system dependencies, add PPA for Python 3.9
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone KDTalker repo
RUN git clone https://github.com/chaolongy/KDTalker.git .

# Create and activate virtual environment using python3.9
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies inside the virtual environment
# Upgrade pip within the venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch matching README recommendation
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install numpy first (needed for many packages)
RUN pip install --no-cache-dir numpy==1.24.4

# Install dependencies one by one for better error isolation
RUN pip install --no-cache-dir onnxruntime-gpu==1.18.0
RUN pip install --no-cache-dir transformers==4.33.0
RUN pip install --no-cache-dir pyyaml==6.0.1
RUN pip install --no-cache-dir scipy==1.10.0
RUN pip install --no-cache-dir imageio==2.34.2
RUN pip install --no-cache-dir lmdb==1.4.1
RUN pip install --no-cache-dir tqdm==4.64.1
RUN pip install --no-cache-dir rich==13.7.1
RUN pip install --no-cache-dir ffmpeg-python==0.2.0
RUN pip install --no-cache-dir protobuf==3.20.2
RUN pip install --no-cache-dir onnx==1.16.1
RUN pip install --no-cache-dir scikit-image==0.24.0
RUN pip install --no-cache-dir scikit-learn==1.3.2
RUN pip install --no-cache-dir albumentations==1.4.10
RUN pip install --no-cache-dir matplotlib==3.7.0
RUN pip install --no-cache-dir imageio-ffmpeg==0.5.1
RUN pip install --no-cache-dir tyro==0.8.5
RUN pip install --no-cache-dir pykalman==0.9.7
RUN pip install --no-cache-dir pillow>=10.2.0
RUN pip install --no-cache-dir pytorch_fid
RUN pip install --no-cache-dir cpbd

RUN pip install --no-cache-dir wandb==0.17.5
RUN pip install --no-cache-dir accelerate==0.23.0
RUN pip install --no-cache-dir basicsr==1.4.2
RUN pip install --no-cache-dir diffusers==0.10.2
RUN pip install --no-cache-dir einops==0.6.0
RUN pip install --no-cache-dir einops_exts==0.0.4
RUN pip install --no-cache-dir hydra-core==1.3.2
RUN pip install --no-cache-dir librosa==0.10.0.post2
# Install lws separately as it's problematic
RUN pip install --no-cache-dir lws==1.2.7 || echo "lws failed to install, continuing anyway"
RUN pip install --no-cache-dir moviepy==1.0.3
RUN pip install --no-cache-dir omegaconf==2.3.0
RUN pip install --no-cache-dir opencv_python_headless>=4.9.0.80
RUN pip install --no-cache-dir pydub==0.25.1
# PyYAML already installed above
RUN pip install --no-cache-dir realesrgan==0.3.0
RUN pip install --no-cache-dir rotary_embedding_torch==0.3.0
RUN pip install --no-cache-dir timm==0.4.12
RUN pip install --no-cache-dir torch_ema==0.3
RUN pip install --no-cache-dir warmup_scheduler==0.3
RUN pip install --no-cache-dir yacs==0.1.8
# numpy already installed above

# Install cmake before dlib
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dlib==19.24.6

# Install huggingface_hub for model weights
RUN pip install --no-cache-dir huggingface_hub

# Create proper Python script for downloading the repository snapshot
RUN echo 'import os\n\
import shutil\n\
import tempfile\n\
from huggingface_hub import snapshot_download\n\
\n\
def download_model_repo():\n\
    print("Downloading the KDTalker repository from Hugging Face...")\n\
    temp_dir = tempfile.mkdtemp()\n\
    try:\n\
        # Download the complete repository snapshot\n\
        repo_path = snapshot_download(\n\
            repo_id="ChaolongYang/KDTalker",\n\
            local_dir=temp_dir,\n\
            local_dir_use_symlinks=False\n\
        )\n\
        print(f"Repository downloaded to {repo_path}")\n\
        \n\
        # Create the necessary directories\n\
        os.makedirs("./ckpts", exist_ok=True)\n\
        os.makedirs("./pretrained_weights", exist_ok=True)\n\
        \n\
        # List all downloaded files for verification\n\
        print("\\nDownloaded files:")\n\
        for root, dirs, files in os.walk(temp_dir):\n\
            for file in files:\n\
                path = os.path.join(root, file)\n\
                rel_path = os.path.relpath(path, temp_dir)\n\
                print(f" - {rel_path}")\n\
                \n\
                # Copy relevant files to the right locations\n\
                if rel_path.startswith("checkpoints/"):\n\
                    dst = os.path.join("./ckpts", os.path.basename(rel_path))\n\
                    print(f"Copying {rel_path} to {dst}")\n\
                    shutil.copy(path, dst)\n\
                elif rel_path.startswith("pretrained_weights/"):\n\
                    dst_dir = os.path.join("./pretrained_weights", os.path.dirname(rel_path.replace("pretrained_weights/", "")))\n\
                    os.makedirs(dst_dir, exist_ok=True)\n\
                    dst = os.path.join(dst_dir, os.path.basename(rel_path))\n\
                    print(f"Copying {rel_path} to {dst}")\n\
                    shutil.copy(path, dst)\n\
    finally:\n\
        # Clean up temp directory\n\
        shutil.rmtree(temp_dir, ignore_errors=True)\n\
    \n\
    # Show the final directory structure\n\
    print("\\nFinal directory structure:")\n\
    for root, dirs, files in os.walk("./"):\n\
        level = root.count(os.sep)\n\
        indent = " " * 4 * level\n\
        print(f"{indent}{os.path.basename(root) or root}/")\n\
        for file in files:\n\
            if file.endswith(".pth") or file.endswith(".onnx"):\n\
                print(f"{indent}    {file}")\n\
\n\
if __name__ == "__main__":\n\
    download_model_repo()\n' > download.py

# Run the download script
RUN python download.py

# Create a simple script that keeps the container running
RUN echo 'import time\n\
import os\n\
import signal\n\
import sys\n\
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
print("KDTalker service is ready and waiting for commands.")\n\
print("Use docker exec to run animation commands.")\n\
\n\
# Keep container running\n\
while True:\n\
    try:\n\
        time.sleep(60)  # Just sleep and keep container running\n\
    except KeyboardInterrupt:\n\
        print("\\nShutting down...")\n\
        break\n\
' > server.py

# Set entrypoint to use python3.9 from the virtual environment
ENTRYPOINT ["/opt/venv/bin/python3.9"]

# Command to run the server
CMD ["server.py"]
