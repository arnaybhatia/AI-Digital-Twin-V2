import os
from huggingface_hub import hf_hub_download

def download_model_repo():
    print("Downloading required KDTalker model files from Hugging Face...")
    
    # List of specific files we need
    files_to_download = [
        "pretrained_weights/insightface/models/buffalo_l/2d106det.onnx",
        "pretrained_weights/insightface/models/buffalo_l/det_10g.onnx",
        "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "pretrained_weights/liveportrait/base_models/warping_module.pth",
        "pretrained_weights/liveportrait/landmark.onnx",
        "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
        "ckpts/KDTalker.pth",
        "ckpts/shape_predictor_68_face_landmarks.dat",
        "ckpts/wav2lip.pth"
    ]
    
    # Create necessary directories
    for file_path in files_to_download:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Download each file individually
    for file_path in files_to_download:
        print(f"Downloading {file_path}...")
        try:
            hf_hub_download(
                repo_id="ChaolongYang/KDTalker",
                filename=file_path,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {file_path}")
        except Exception as e:
            print(f"Error downloading {file_path}: {str(e)}")
            raise

    print("\nAll required model files have been downloaded successfully!")

if __name__ == "__main__":
    download_model_repo()
