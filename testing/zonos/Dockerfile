FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/zonos

# Install system dependencies - using python3 instead of python3.9 for compatibility
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Zonos repo
RUN git clone https://github.com/Zyphra/Zonos.git .

# Install Python dependencies
RUN pip3 install --no-cache-dir torch torchaudio
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir huggingface_hub

# Download weights from Hugging Face
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    import os; \
    os.makedirs('./checkpoints', exist_ok=True); \
    hf_hub_download(repo_id='Zyphra/Zonos-v0.1-transformer', filename='model.pth', local_dir='./checkpoints')"

# Create zonos_generate.py
RUN echo 'import argparse\n\
import torch\n\
import torchaudio\n\
from zonos.model import Zonos\n\
from zonos.conditioning import make_cond_dict\n\
\n\
def generate_audio(text, output, speaker_audio="/data/speaker.wav"):\n\
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n\
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)\n\
    wav, sampling_rate = torchaudio.load(speaker_audio)\n\
    speaker = model.make_speaker_embedding(wav, sampling_rate)\n\
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")\n\
    conditioning = model.prepare_conditioning(cond_dict)\n\
    codes = model.generate(conditioning)\n\
    wavs = model.autoencoder.decode(codes).cpu()\n\
    torchaudio.save(output, wavs[0], model.autoencoder.sampling_rate)\n\
\n\
if __name__ == "__main__":\n\
    parser = argparse.ArgumentParser()\n\
    parser.add_argument("--text", required=True)\n\
    parser.add_argument("--output", required=True)\n\
    parser.add_argument("--speaker_audio", default="/data/speaker.wav")\n\
    args = parser.parse_args()\n\
    generate_audio(args.text, args.output, args.speaker_audio)' > zonos_generate.py

# Command to generate audio
CMD ["python3", "zonos_generate.py", "--text", "Hello, this is your AI twin.", "--output", "/data/synthesized.wav", "--speaker_audio", "/data/speaker.wav"]