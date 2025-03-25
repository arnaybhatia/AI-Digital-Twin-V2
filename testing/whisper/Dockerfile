FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/whisper

# Install system dependencies - using python3 instead of python3.9 for compatibility
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Whisper
RUN pip3 install --no-cache-dir git+https://github.com/openai/whisper.git

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

# Command to transcribe audio
CMD ["python3", "whisper_transcribe.py", "--file", "/data/input.wav", "--output", "/data/transcript.txt"]