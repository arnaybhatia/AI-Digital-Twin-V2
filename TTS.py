import subprocess
import os
import time
import threading
from typing import Optional
from playsound import playsound
import uuid

class TextToSpeech:
    def __init__(self, sample_audio="./data/sample_audio.wav"):
        self.temp_dir = "./temp_audio"
        self.sample_audio = os.path.abspath(sample_audio)
        self.data_dir = os.path.dirname(self.sample_audio)
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Create data directory for Docker if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def speak(self, text: str, interrupt_event: Optional[threading.Event] = None):
        try:
            # Generate a unique filename for this TTS request
            output_filename = f"zonos_output_{uuid.uuid4()}.wav"
            output_path = os.path.join(self.temp_dir, output_filename)
            
            # Convert paths for Docker volume mapping
            docker_output = f"/data/{output_filename}"
            docker_sample = "/data/sample_audio.wav"
            
            # Prepare the Docker command
            cmd = [
                "docker", "run", "--rm", 
                "-v", f"{self.data_dir}:/data", 
                "-v", f"{os.path.abspath(self.temp_dir)}:/data/output",
                "zonos",
                "python3", "zonos_generate.py", 
                "--text", text,
                "--output", docker_output,
                "--speaker_audio", docker_sample
            ]
            
            # Run the Docker container if not interrupted
            if not (interrupt_event and interrupt_event.is_set()):
                print(f"Generating speech with Zonos: '{text}'")
                result = subprocess.run(cmd, check=True, capture_output=True)
                
                if result.returncode != 0:
                    print(f"Error running Zonos: {result.stderr.decode()}")
                    return
                
                # Play the generated audio if not interrupted
                if not (interrupt_event and interrupt_event.is_set()):
                    playsound(output_path)
                    
                # Cleanup
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {output_path}: {e}")
            else:
                print("TTS interrupted")
                
        except Exception as e:
            print(f"[Zonos TTS] Error: {e}")

    def cleanup(self):
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("zonos_output_") and file.endswith(".wav"):
                    os.remove(os.path.join(self.temp_dir, file))
        except Exception as e:
            print(f"[Zonos TTS] Cleanup error: {e}")
