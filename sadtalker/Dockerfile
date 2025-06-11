# Use an official NVIDIA CUDA runtime as a parent image.
# This image comes with CUDA 11.3.1 and cuDNN 8, matching the repository's requirements.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system-level dependencies required by the application and its libraries.
# - python3.8 and pip for running the application.
# - git and wget for cloning the repository and downloading models.
# - ffmpeg is required for video processing.
# - libgl1-mesa-glx and libglib2.0-0 are common dependencies for OpenCV and other graphics libraries.
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone the SadTalker repository from GitHub into the working directory.
RUN git clone https://github.com/OpenTalker/SadTalker.git .

# Install Python dependencies using pip.
# First, install the specific versions of torch, torchvision, and torchaudio that are compatible with CUDA 11.3.
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install the rest of the Python packages listed in requirements.txt.
RUN pip install -r requirements.txt

# Download the pre-trained models required by SadTalker.
# The download_models.sh script handles fetching all necessary model files.
RUN bash scripts/download_models.sh

# Download face alignment models required by facexlib, as specified in cog.yaml.
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    wget --output-document "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth" "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" && \
    wget --output-document "/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip" "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"

# Fix Gradio compatibility by installing a compatible version
RUN pip install gradio==3.50.2 TTS

# Fix numba compatibility with numpy 1.23.4 by installing the latest available numba version
RUN pip install numba==0.58.1

# Copy the server script
COPY server.py /app/

# Create a symlink for python command
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Set the entrypoint and command for the container runtime
ENTRYPOINT ["python"]
CMD ["server.py"]