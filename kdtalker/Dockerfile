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

# Create and activate virtual environment
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
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

# Create directories for models
RUN mkdir -p pretrained_weights/insightface/models/buffalo_l && \
    mkdir -p pretrained_weights/liveportrait/base_models && \
    mkdir -p pretrained_weights/liveportrait/retargeting_models && \
    mkdir -p ckpts

# Download models from Hugging Face
COPY download_models.py .
RUN python3 download_models.py

# Create data directory for mounting
RUN mkdir -p /data/temp_audio /data/temp_video

# Set environment variables
ENV PYTHONPATH=/app/kdtalker
ENV CUDA_VISIBLE_DEVICES=0

RUN pip install --no-cache-dir dlib==19.24.6
RUN pip install --no-cache-dir huggingface_hub==0.17.3
RUN pip install --no-cache-dir diffusers==0.17.0
RUN mv ckpts/KDTalker.pth .

# Create a simple server script to keep container running
COPY inference.py /app/kdtalker/inference.py
RUN echo 'import os\n\
import time\n\
import signal\n\
import sys\n\
import subprocess\n\
\n\
def signal_handler(sig, frame):\n\
    print("\\nShutting down...")\n\
    sys.exit(0)\n\
\n\
def run_inference(source_image, driven_audio, output):\n\
    cmd = ["python3", "inference.py",\n\
           "-source_image", source_image,\n\
           "-driven_audio", driven_audio,\n\
           "-output", output]\n\
    try:\n\
        subprocess.run(cmd, check=True)\n\
        return True\n\
    except subprocess.CalledProcessError as e:\n\
        print(f"Error running inference: {e}")\n\
        if hasattr(e, "stderr"):\n\
            print(f"Error output: {e.stderr}")\n\
        return False\n\
\n\
signal.signal(signal.SIGTERM, signal_handler)\n\
signal.signal(signal.SIGINT, signal_handler)\n\
\n\
print("KDTalker service is ready.")\n\
while True:\n\
    try:\n\
        time.sleep(1)\n\
    except KeyboardInterrupt:\n\
        print("\\nShutting down...")\n\
        break\n' > server.py

# Use server.py as the default command
CMD ["python3", "server.py"]