# Use NVIDIA CUDA runtime as base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set the working directory
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies required for Chatterbox
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
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python command
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create virtual environment
RUN python -m venv /app/venv

# Activate virtual environment and upgrade pip
RUN /app/venv/bin/python -m pip install --upgrade pip

# Install PyTorch with CUDA support first
RUN /app/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Chatterbox TTS
RUN /app/venv/bin/pip install chatterbox-tts

# Install additional dependencies for API server
RUN /app/venv/bin/pip install flask requests

# Set PATH to use virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Create a simple API server script
RUN cat > server.py << 'EOF'
import os
import tempfile
import torch
import torchaudio as ta
from flask import Flask, request, jsonify, send_file
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Initialize Chatterbox model
print("Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device=device)
print("Chatterbox model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "device": device,
        "service": "chatterbox-tts"
    })

@app.route('/v1/tts', methods=['POST'])
def generate_speech():
    try:
        print(f"Received request: {request.method} {request.url}")
        print(f"Content-Type: {request.content_type}")
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            print(f"JSON data: {data}")
            text = data.get('text', '') if data else ''
            audio_prompt_path = data.get('audio_prompt_path', '') if data else ''
        else:
            print(f"Form data: {request.form}")
            text = request.form.get('text', '')
            audio_prompt_path = request.form.get('audio_prompt_path', '')
        
        print(f"Text: {text}")
        print(f"Audio prompt path: {audio_prompt_path}")
        
        if not text:
            print("Error: No text provided")
            return jsonify({"error": "No text provided"}), 400
        
        # For compatibility with existing code, look for audio file in data directory
        if not audio_prompt_path:
            # Try to find a default audio file in the data directory
            data_dir = "/app/data"
            print(f"Looking for audio files in: {data_dir}")
            if os.path.exists(data_dir):
                for filename in ["speaker.wav", "trainingaudio.wav"]:
                    potential_path = os.path.join(data_dir, filename)
                    print(f"Checking: {potential_path}")
                    if os.path.exists(potential_path):
                        audio_prompt_path = potential_path
                        print(f"Found audio file: {audio_prompt_path}")
                        break
        
        if not audio_prompt_path or not os.path.exists(audio_prompt_path):
            print(f"Error: Audio prompt file not found: {audio_prompt_path}")
            return jsonify({"error": f"Audio prompt file not found: {audio_prompt_path}"}), 400
        
        print(f"Generating speech for text: {text[:50]}...")
        print(f"Using audio prompt: {audio_prompt_path}")
        
        # Generate audio using Chatterbox
        wav = model.generate(text, audio_prompt_path=audio_prompt_path)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            ta.save(tmp_file.name, wav, model.sr)
            print(f"Generated audio saved to: {tmp_file.name}")
            
            # Read the file and return as response
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            # Return audio data directly
            from flask import Response
            return Response(
                audio_data,
                mimetype='audio/wav',
                headers={'Content-Disposition': 'attachment; filename=generated_speech.wav'}
            )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
EOF

# Expose port for Chatterbox API
EXPOSE 8080

# Set the default command to run the API server
CMD ["python", "server.py"]