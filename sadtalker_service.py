import subprocess
import os
import glob
import time
from typing import Optional, Tuple
import uuid

class SadTalkerService:
    def __init__(self, data_dir: str, results_dir: str = None):
        """
        Initialize SadTalker service
        
        Args:
            data_dir: Path to the data directory containing source images
            results_dir: Path to store video results (defaults to results/)
        """
        self.data_dir = os.path.abspath(data_dir)
        self.results_dir = os.path.abspath(results_dir or "results")
        self.docker_image = "wawa9000/sadtalker"
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_latest_audio_file(self, temp_audio_dir: str) -> Optional[str]:
        """
        Get the latest audio file from temp_audio directory
        
        Args:
            temp_audio_dir: Path to temp audio directory
            
        Returns:
            Path to the latest audio file or None if no files found
        """
        audio_pattern = os.path.join(temp_audio_dir, "*.wav")
        audio_files = glob.glob(audio_pattern)
        
        if not audio_files:
            print(f"No audio files found in {temp_audio_dir}")
            return None
            
        # Get the most recently modified file
        latest_file = max(audio_files, key=os.path.getmtime)
        return latest_file
    
    def get_source_image(self) -> Optional[str]:
        """
        Get the source image from data directory
        
        Returns:
            Path to source image or None if not found
        """
        source_image_path = os.path.join(self.data_dir, "source_image.png")
        if os.path.exists(source_image_path):
            return source_image_path
        
        # Try other common image formats
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = os.path.join(self.data_dir, f"source_image{ext}")
            if os.path.exists(image_path):
                return image_path
                
        print(f"No source image found in {self.data_dir}")
        return None
    
    def generate_video(self, 
                      audio_file: str = None, 
                      source_image: str = None,
                      expression_scale: float = 1.0,
                      still: bool = True,
                      output_name: str = None) -> Optional[str]:
        """
        Generate video using SadTalker Docker image
        
        Args:
            audio_file: Path to audio file (if None, uses latest from temp_audio)
            source_image: Path to source image (if None, uses source_image.png)
            expression_scale: Expression scale parameter (default: 1.0)
            still: Whether to use still mode (default: True)
            output_name: Custom output name (if None, generates unique name)
            
        Returns:
            Path to generated video file or None if failed
        """
        try:
            # Get audio file
            if audio_file is None:
                temp_audio_dir = os.path.join(self.data_dir, "temp_audio")
                audio_file = self.get_latest_audio_file(temp_audio_dir)
            
            if not audio_file or not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                return None
                
            # Get source image
            if source_image is None:
                source_image = self.get_source_image()
                
            if not source_image or not os.path.exists(source_image):
                print(f"Source image not found: {source_image}")
                return None
            
            # Generate output filename
            if output_name is None:
                timestamp = int(time.time())
                unique_id = str(uuid.uuid4())[:8]
                output_name = f"sadtalker_output_{timestamp}_{unique_id}.mp4"
            
            output_path = os.path.join(self.results_dir, output_name)
            
            # Prepare Docker command
            # Convert Windows paths to Unix-style for Docker
            host_dir = os.path.dirname(os.path.abspath(audio_file))
            audio_filename = os.path.basename(audio_file)
            image_filename = os.path.basename(source_image)
            
            # Copy source image to the same directory as audio for Docker mount
            temp_image_path = os.path.join(host_dir, image_filename)
            if source_image != temp_image_path:
                import shutil
                shutil.copy2(source_image, temp_image_path)
            
            docker_cmd = [
                "docker", "run", 
                "--gpus", "all",
                "--rm",
                "-v", f"{host_dir}:/host_dir",
                self.docker_image,
                "--driven_audio", f"/host_dir/{audio_filename}",
                "--source_image", f"/host_dir/{image_filename}",
                "--expression_scale", str(expression_scale),
                "--result_dir", "/host_dir"
            ]
            
            if still:
                docker_cmd.append("--still")
            
            print(f"Running SadTalker with command: {' '.join(docker_cmd)}")
            print(f"Audio file: {audio_file}")
            print(f"Source image: {source_image}")
            print(f"Output will be saved to: {host_dir}")
            
            # Run Docker command
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"SadTalker Docker command failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None
            
            print("SadTalker completed successfully!")
            print(f"STDOUT: {result.stdout}")
            
            # Find the generated video file
            # SadTalker typically generates files with specific naming patterns
            video_files = glob.glob(os.path.join(host_dir, "*.mp4"))
            if video_files:
                # Get the most recently created video file
                latest_video = max(video_files, key=os.path.getctime)
                
                # Move to results directory with our naming convention
                final_output_path = os.path.join(self.results_dir, output_name)
                if latest_video != final_output_path:
                    import shutil
                    shutil.move(latest_video, final_output_path)
                
                print(f"Video generated successfully: {final_output_path}")
                return final_output_path
            else:
                print("No video file found after SadTalker execution")
                return None
                
        except subprocess.TimeoutExpired:
            print("SadTalker execution timed out")
            return None
        except Exception as e:
            print(f"Error running SadTalker: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_video_from_latest_audio(self, 
                                       expression_scale: float = 1.0,
                                       still: bool = True) -> Optional[str]:
        """
        Convenience method to generate video using latest audio file and source image
        
        Args:
            expression_scale: Expression scale parameter (default: 1.0)
            still: Whether to use still mode (default: True)
            
        Returns:
            Path to generated video file or None if failed
        """
        return self.generate_video(
            expression_scale=expression_scale,
            still=still
        )

# Example usage
if __name__ == "__main__":
    # Initialize service
    service = SadTalkerService(data_dir="data")
    
    # Generate video from latest audio
    video_path = service.generate_video_from_latest_audio()
    
    if video_path:
        print(f"Success! Video generated at: {video_path}")
    else:
        print("Failed to generate video")
