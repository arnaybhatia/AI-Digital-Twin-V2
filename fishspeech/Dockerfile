# Use NVIDIA CUDA runtime as base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set the working directory
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies required for Fish Speech
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    curl \
    portaudio19-dev \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python command
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Install uv (modern Python package manager)
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Clone the Fish Speech repository
RUN git clone https://github.com/fishaudio/fish-speech .

# Sync dependencies using uv with Python 3.12 and install all extras
RUN uv sync --python 3.12 --all-extras

# Activate the virtual environment for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Accept build argument for Hugging Face token
ARG HUGGING_FACE_HUB_TOKEN

# Download the model checkpoints using huggingface-cli
RUN if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then \
        huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN --add-to-git-credential && \
        huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini; \
    else \
        echo "WARNING: HUGGING_FACE_HUB_TOKEN not provided, skipping model download"; \
    fi

# Expose port for Fish Speech API
EXPOSE 8080

# Set the default command to run the API server using the virtual environment
CMD ["/app/.venv/bin/python", "-m", "tools.api_server", \
     "--listen", "0.0.0.0:8080", \
     "--llama-checkpoint-path", "checkpoints/openaudio-s1-mini", \
     "--decoder-checkpoint-path", "checkpoints/openaudio-s1-mini/codec.pth", \
     "--decoder-config-name", "modded_dac_vq"]