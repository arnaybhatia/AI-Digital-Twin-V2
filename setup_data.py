import os
import shutil
import argparse
from PIL import Image
import numpy as np

def create_test_image(output_path, width=512, height=512):
    """Create a simple test image"""
    # Create a gradient image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Create a simple face-like shape
            circle_x = (x - width/2)**2
            circle_y = (y - height/2)**2
            circle_r = np.sqrt(circle_x + circle_y) / (width/3)
            
            if circle_r < 1.0:
                # Face area
                img[y, x] = [220, 180, 160]  # Skin tone
            else:
                # Background
                img[y, x] = [240, 240, 240]  # Light gray
                
    # Add eyes
    eye_size = width // 10
    for eye_x in [width//3, 2*width//3]:
        for x in range(eye_x - eye_size//2, eye_x + eye_size//2):
            for y in range(height//3 - eye_size//2, height//3 + eye_size//2):
                if (x - eye_x)**2 + (y - height//3)**2 < (eye_size//2)**2:
                    img[y, x] = [255, 255, 255]  # White eye
                    # Pupil
                    if (x - eye_x)**2 + (y - height//3)**2 < (eye_size//4)**2:
                        img[y, x] = [80, 80, 160]  # Blue pupil
    
    # Add mouth
    for x in range(width//3, 2*width//3):
        y = int(height * 2/3 + np.sin((x - width/2) / 30) * 10)
        for dy in range(-3, 4):
            if 0 <= y + dy < height:
                img[y + dy, x] = [180, 100, 100]  # Lips
    
    # Convert to PIL Image and save
    image = Image.fromarray(img)
    image.save(output_path)
    print(f"Created test image: {output_path}")

def create_test_audio(output_path):
    """Create a simple test audio file with silence"""
    import wave
    import struct
    
    # Create 1 second of silence (16kHz, 16-bit mono)
    duration = 1  # seconds
    sample_rate = 16000
    num_samples = duration * sample_rate
    
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            # Create some low-level noise to simulate silence with ambient noise
            value = int(32767 * 0.01 * np.random.random())
            packed_value = struct.pack('h', value)
            wf.writeframes(packed_value)
            
    print(f"Created test audio file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Set up data directory structure for AI Digital Twin')
    parser.add_argument('--data-dir', default='data', help='Path to the data directory')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_audio'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_video'), exist_ok=True)
    
    # Create test image
    source_image = os.path.join(data_dir, 'source_image.png')
    if not os.path.exists(source_image):
        create_test_image(source_image)
        
    # Create test audio
    speaker_audio = os.path.join(data_dir, 'speaker.wav')
    if not os.path.exists(speaker_audio):
        try:
            create_test_audio(speaker_audio)
        except ImportError:
            print("Warning: Could not create test audio - wave or numpy module not available")
            with open(speaker_audio, 'wb') as f:
                f.write(b'')  # Create empty file as placeholder
    
    # Create .env file if it doesn't exist
    env_file = os.path.join(base_dir, '.env')
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("# Add your API key here\n")
            f.write("TMPT_API_KEY=your_api_key_here\n")
        print(f"Created template .env file. Please edit {env_file} to add your API key.")
    
    print(f"Data directory setup complete at {data_dir}")
    print("Now you can run 'docker compose build' to build the containers")
    print("Then run 'docker compose up -d' to start the services")
    print("Finally, run 'python main.py' to start the AI Digital Twin")

if __name__ == '__main__':
    main()
