# Use NVIDIA CUDA runtime as base image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set the working directory
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies required for Dia
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

# Clone the Dia repository
RUN git clone https://github.com/nari-labs/dia.git .

# Install dia package and dependencies using uv
RUN uv venv && uv pip install -e .
RUN uv pip install git+https://github.com/huggingface/transformers.git

# Activate the virtual environment for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Download the model weights from Hugging Face
RUN python -c "from transformers import AutoProcessor, DiaForConditionalGeneration; \
    processor = AutoProcessor.from_pretrained('nari-labs/Dia-1.6B-0626'); \
    model = DiaForConditionalGeneration.from_pretrained('nari-labs/Dia-1.6B-0626')"

# Create a simple API server script
RUN cat > server.py << 'EOF'
import os
import tempfile
from flask import Flask, request, jsonify, send_file
from transformers import AutoProcessor, DiaForConditionalGeneration
import torch

app = Flask(__name__)

# Initialize model and processor
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

print(f"Loading Dia model on {torch_device}...")
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": torch_device})

@app.route('/generate', methods=['POST'])
def generate_speech():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Format text with speaker tags if not already present
        if not text.startswith('[S1]'):
            text = f"[S1] {text}"
        
        # Process input
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(torch_device)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=3072, 
                guidance_scale=3.0, 
                temperature=1.8, 
                top_p=0.90, 
                top_k=45
            )
        
        # Decode outputs
        outputs = processor.batch_decode(outputs)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            processor.save_audio(outputs, tmp_file.name)
            return send_file(tmp_file.name, as_attachment=True, download_name='generated_speech.wav')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

# Expose port for Dia API
EXPOSE 8080

# Set the default command to run the API server
CMD ["/app/.venv/bin/python", "server.py"]