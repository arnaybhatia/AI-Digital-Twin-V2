#!/bin/bash

# Check if version and mode are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 {v1.0|v1.5} {normal|realtime}"
    echo "Example: $0 v1.5 normal"
    exit 1
fi

VERSION=$1
MODE=$2

# Set default values
UNET_MODEL="models/musetalkV15/unet.pth"
UNET_CONFIG="models/musetalkV15/musetalk.json"
INFERENCE_CONFIG="configs/inference/test.yaml"
RESULT_DIR="results/test"
FPS=25

# Update paths based on version
if [ "$VERSION" = "v1.0" ]; then
    UNET_MODEL="models/musetalk/pytorch_model.bin"
    UNET_CONFIG="models/musetalk/musetalk.json"
    INFERENCE_CONFIG="configs/inference/test_v1.yaml"
    RESULT_DIR="results/test_v1"
fi

# Update config based on mode
if [ "$MODE" = "realtime" ]; then
    INFERENCE_CONFIG="configs/inference/realtime.yaml"
    RESULT_DIR="results/realtime"
    
    # Run realtime inference
    python -m scripts.realtime_inference \
        --inference_config "$INFERENCE_CONFIG" \
        --result_dir "$RESULT_DIR" \
        --unet_model_path "$UNET_MODEL" \
        --unet_config "$UNET_CONFIG" \
        --version "$VERSION" \
        --fps $FPS
else
    # Run normal inference
    python -m scripts.inference \
        --inference_config "$INFERENCE_CONFIG" \
        --result_dir "$RESULT_DIR" \
        --unet_model_path "$UNET_MODEL" \
        --unet_config "$UNET_CONFIG" \
        --version "$VERSION"
fi
