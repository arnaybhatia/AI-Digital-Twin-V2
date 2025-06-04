FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Setting timezone can prevent some tzdata prompts
ENV TZ=UTC

# Install system dependencies
# Updated for Ubuntu 22.04 compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install gdown with --break-system-packages flag
RUN python3 -m pip install --break-system-packages gdown

# Clone the repository (Assuming this is intended, if not, adjust COPY accordingly)
# If you have the MuseTalk code locally and want to use that,
# you should COPY . . instead of git clone.
# For now, keeping the git clone as per your original Dockerfile structure.
RUN git clone https://github.com/TMElyralab/MuseTalk.git . && \
    chmod +x *.sh

# Install PyTorch with CUDA support
RUN pip3 install --break-system-packages torch torchvision torchaudio

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --break-system-packages -r requirements.txt

# Install MMLab packages
RUN python3 -m pip install --break-system-packages --no-cache-dir -U openmim && \
    python3 -m mim install mmengine && \
    python3 -m mim install "mmcv==2.0.1" && \
    python3 -m mim install "mmdet==3.1.0" && \
    python3 -m mim install "mmpose==1.1.0"

# Create models directory
RUN mkdir -p models

# Download model weights using gdown and wget
RUN echo "Downloading model weights..." && \
    # Create directories
    mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose \
    models/face-parse-bisent models/sd-vae models/whisper && \
    \
    # Download MuseTalk weights
    wget -q --show-progress -O models/musetalk/pytorch_model.bin \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin" && \
    wget -q --show-progress -O models/musetalk/musetalk.json \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json" && \
    \
    # Download MuseTalk V1.5 weights
    wget -q --show-progress -O models/musetalkV15/unet.pth \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/unet.pth" && \
    wget -q --show-progress -O models/musetalkV15/musetalk.json \
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json" && \
    \
    # Download VAE weights
    wget -q --show-progress -O models/sd-vae/config.json \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" && \
    wget -q --show-progress -O models/sd-vae/diffusion_pytorch_model.bin \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" && \
    \
    # Download Whisper weights
    wget -q --show-progress -O models/whisper/config.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" && \
    wget -q --show-progress -O models/whisper/pytorch_model.bin \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" && \
    wget -q --show-progress -O models/whisper/preprocessor_config.json \
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" && \
    \
    # Download DWPose weights
    wget -q --show-progress -O models/dwpose/dw-ll_ucoco_384.pth \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" && \
    \
    # Download SyncNet weights
    wget -q --show-progress -O models/syncnet/latentsync_syncnet.pt \
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt" && \
    \
    # Download face parsing weights using gdown (Google Drive)
    # Ensure gdown is installed via apt or pip if this fails. It's added to apt install now.
    gdown "1wF8P-hfWBoQZ-X9Rl0Ak_wMqxzjjElvP" -O models/face-parse-bisent/79999_iter.pth && \
    wget -q --show-progress -O models/face-parse-bisent/resnet18-5c106cde.pth \
    "https://download.pytorch.org/models/resnet18-5c106cde.pth"

# Copy the inference script and make it executable
COPY inference.sh /app/inference.sh
RUN chmod +x /app/inference.sh

# Set FFmpeg path
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Create input and output directories
RUN mkdir -p /app/inputs /app/outputs /app/results

# Create a comprehensive entrypoint script
RUN echo '#!/bin/bash\n\
echo "=== MuseTalk Docker Container Ready! ==="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  1. Normal inference (v1.5): ./inference.sh v1.5 normal"\n\
echo "  2. Real-time inference (v1.5): ./inference.sh v1.5 realtime"\n\
echo "  3. Normal inference (v1.0): ./inference.sh v1.0 normal"\n\
echo "  4. Real-time inference (v1.0): ./inference.sh v1.0 realtime"\n\
echo "  5. Gradio demo: python app.py --use_float16"\n\
echo ""\n\
echo "Direct Python commands:"\n\
echo "  - Normal: python -m scripts.inference --inference_config configs/inference/test.yaml --result_dir results/test --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15"\n\
echo "  - Realtime: python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --result_dir results/realtime --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15 --fps 25"\n\
echo ""\n\
echo "File locations:"\n\
echo "  - Input files: /app/inputs (mounted from host ./inputs)"\n\
echo "  - Output files: /app/outputs (mounted from host ./outputs)"\n\
echo "  - Config files: /app/configs"\n\
echo ""\n\
echo "Note: Update your config files to point to /app/inputs for input files"\n\
echo "=== Ready for inference! ==="\n\
echo ""\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port for Gradio
EXPOSE 7860

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]