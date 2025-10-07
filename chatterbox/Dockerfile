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

# Install Chatterbox TTS
RUN /app/venv/bin/pip install chatterbox-tts==0.1.2

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
from flask import Flask, request, jsonify, send_file, Response
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)

# Set CUDA environment variables before torch initialization
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['NVIDIA_VISIBLE_DEVICES'] = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'all')

# Force CUDA initialization to happen properly
try:
    if torch.cuda.is_available():
        torch.cuda.init()
        device = "cuda"
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
except Exception as e:
    print(f"CUDA initialization failed: {e}")
    device = "cpu"
    print("Falling back to CPU.")

print(f"Using device: {device}")

# Initialize Chatterbox model
print("Loading Chatterbox TTS model...")
try:
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Chatterbox model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load on CPU...")
    device = "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Chatterbox model loaded successfully on CPU!")

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Chatterbox TTS API",
        "version": "1.0",
        "device": device,
        "endpoints": {
            "health": "/health",
            "tts": "/v1/tts (POST)",
            "api_info": "/v1"
        }
    })

@app.route('/v1', methods=['GET'])
def api_info():
    return jsonify({
        "service": "Chatterbox TTS API",
        "version": "1.0",
        "device": device,
        "endpoints": {
            "/v1/tts": {
                "method": "POST",
                "description": "Generate speech from text",
                "parameters": {
                    "text": "Text to convert to speech (required)",
                    "audio_prompt_path": "Path to audio prompt file (optional, defaults to /app/data/speaker.wav)"
                },
                "example": {
                    "text": "Hello world",
                    "audio_prompt_path": "/app/data/speaker.wav"
                }
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "device": device,
        "service": "chatterbox-tts",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

@app.route('/v1/tts', methods=['POST'])
def generate_speech():
    try:
        print(f"Received request: {request.method} {request.url}")
        print(f"Content-Type: {request.content_type}")
        
        # Validate request content type
        if not request.is_json and not request.form:
            return jsonify({
                "error": "Invalid content type. Use application/json or form data.",
                "example": {"text": "Hello world", "audio_prompt_path": "/app/data/speaker.wav"}
            }), 400
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            print(f"JSON data: {data}")
            if not data:
                return jsonify({
                    "error": "Invalid JSON data",
                    "example": {"text": "Hello world", "audio_prompt_path": "/app/data/speaker.wav"}
                }), 400
            text = data.get('text', '')
            audio_prompt_path = data.get('audio_prompt_path', '')
        else:
            print(f"Form data: {request.form}")
            text = request.form.get('text', '')
            audio_prompt_path = request.form.get('audio_prompt_path', '')
        
        print(f"Text: {text}")
        print(f"Audio prompt path: {audio_prompt_path}")
        
        if not text or not text.strip():
            return jsonify({
                "error": "Text parameter is required and cannot be empty",
                "example": {"text": "Hello world", "audio_prompt_path": "/app/data/speaker.wav"}
            }), 400
        
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
            return jsonify({
                "error": f"Audio prompt file not found: {audio_prompt_path}",
                "available_files": [f for f in os.listdir("/app/data") if f.endswith(('.wav', '.mp3', '.flac'))] if os.path.exists("/app/data") else [],
                "suggestion": "Place an audio file (speaker.wav or trainingaudio.wav) in /app/data/ or specify audio_prompt_path"
            }), 400
        
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
            return Response(
                audio_data,
                mimetype='audio/wav',
                headers={'Content-Disposition': 'attachment; filename=generated_speech.wav'}
            )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/v1/tts', methods=['GET'])
def tts_get_info():
    return jsonify({
        "error": "Method not allowed. Use POST to generate speech.",
        "method": "POST",
        "example": {
            "text": "Hello world",
            "audio_prompt_path": "/app/data/speaker.wav"
        }
    }), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

# Expose port for Chatterbox API
EXPOSE 8080

# Set the default command to run the API server
CMD ["python", "server.py"]