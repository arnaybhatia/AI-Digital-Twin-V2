import subprocess
import os
import uuid
from typing import Optional

class AvatarGenerator:
    def __init__(self, source_image="./data/source_image.png"):
        self.temp_dir = "./temp_video"
        self.source_image = os.path.abspath(source_image)
        self.data_dir = os.path.dirname(self.source_image)
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Create data directory for Docker if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def generate_avatar_video(self, audio_file, output_file=None):
        """
        Generate a talking avatar video using the KDTalker Docker container
        
        Args:
            audio_file: Path to the audio file to sync with the avatar
            output_file: Optional path for the output video (will generate a unique name if not provided)
        
        Returns:
            Path to the generated video file
        """
        try:
            # Generate a unique filename if not provided
            if output_file is None:
                output_filename = f"kdtalker_output_{uuid.uuid4()}.mp4"
                output_file = os.path.join(self.temp_dir, output_filename)
            
            # Convert paths for Docker volume mapping
            docker_output = f"/data/output_{os.path.basename(output_file)}"
            docker_audio = f"/data/{os.path.basename(audio_file)}"
            docker_image = f"/data/{os.path.basename(self.source_image)}"
            
            # Prepare the Docker command
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{self.data_dir}:/data", 
                "-v", f"{os.path.dirname(audio_file)}:/data/audio",
                "-v", f"{os.path.abspath(self.temp_dir)}:/data/output",
                "kdtalker",
                "python3", "inference.py", 
                "--source_image", docker_image,
                "--driven_audio", docker_audio,
                "--output", docker_output
            ]
            
            # Run the Docker container
            print(f"Generating avatar video with KDTalker")
            result = subprocess.run(cmd, check=True, capture_output=True)
            
            if result.returncode != 0:
                print(f"Error running KDTalker: {result.stderr.decode()}")
                return None
                
            return output_file
                
        except Exception as e:
            print(f"[KDTalker] Error: {e}")
            return None
            
    def cleanup(self):
        """Clean up temporary video files"""
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("kdtalker_output_") and file.endswith(".mp4"):
                    os.remove(os.path.join(self.temp_dir, file))
        except Exception as e:
            print(f"[KDTalker] Cleanup error: {e}")