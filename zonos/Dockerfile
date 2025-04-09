FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/zonos

# Install system dependencies - using python3 instead of python3.9 for compatibility
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Zonos repo
RUN git clone https://github.com/Zyphra/Zonos.git .

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir huggingface_hub

# Copy the optimized zonos_generate.py script instead of creating an inline one
COPY zonos_generate.py .

# Set entrypoint to use the virtual environment
ENTRYPOINT ["/opt/venv/bin/python3"]

# Command to generate audio
CMD ["zonos_generate.py", "--text", "Hello, this is your AI twin.", "--output", "/data/synthesized.wav", "--speaker_audio", "/data/speaker.wav"]