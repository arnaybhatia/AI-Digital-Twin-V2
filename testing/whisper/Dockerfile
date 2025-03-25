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
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir git+https://github.com/openai/whisper.git

# Create whisper_transcribe.py
RUN echo 'import argparse\n\
import whisper\n\
\n\
def transcribe(file, output):\n\
    model = whisper.load_model("medium")\n\
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

# Command to transcribe audio
CMD ["whisper_transcribe.py", "--file", "/data/input.wav", "--output", "/data/transcript.txt"]