FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/kdtalker

# Install system dependencies - using python3 instead of python3.9 for compatibility
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone KDTalker repo
RUN git clone https://github.com/chaolongy/KDTalker.git .

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub

# Download weights from Hugging Face
RUN python -c "from huggingface_hub import hf_hub_download; \
    import os; \
    os.makedirs('./ckpts', exist_ok=True); \
    os.makedirs('./pretrained_weights', exist_ok=True); \
    files = ['checkpoints/face_detector.pth', 'checkpoints/audio_extractor.pth', 'checkpoints/KDTalker.pth', \
             'pretrained_weights/insightface/models/buffalo_l/2d106det.onnx', \
             'pretrained_weights/insightface/models/buffalo_l/det_10g.onnx', \
             'pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth', \
             'pretrained_weights/liveportrait/base_models/motion_extractor.pth', \
             'pretrained_weights/liveportrait/base_models/spade_generator.pth', \
             'pretrained_weights/liveportrait/base_models/warping_module.pth', \
             'pretrained_weights/liveportrait/landmark.onnx', \
             'pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth']; \
    for f in files: \
        hf_hub_download(repo_id='ChaolongYang/KDTalker', filename=f, local_dir='./')"

# Set entrypoint to use the virtual environment
ENTRYPOINT ["/opt/venv/bin/python3"]

# Command to run inference
CMD ["inference.py", "--source_image", "/data/source_image.png", "--driven_audio", "/data/input.wav", "--output", "/data/output.mp4"]