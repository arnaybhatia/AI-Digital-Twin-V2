FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app/kdtalker

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone KDTalker repo
RUN git clone https://github.com/chaolongy/KDTalker.git .

# Install Python dependencies
RUN pip3 install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir huggingface_hub

# Download weights from Hugging Face
RUN python3 -c "from huggingface_hub import hf_hub_download; \
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

# Command to run inference
CMD ["python3", "inference.py", "--source_image", "/data/source_image.png", "--driven_audio", "/data/input.wav", "--output", "/data/output.mp4"]