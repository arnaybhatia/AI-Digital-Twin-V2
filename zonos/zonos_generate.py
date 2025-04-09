import argparse
import torch
import torchaudio
import os
import time
import gc
import sys
from typing import Optional

def generate_audio(text: str, output: str, speaker_audio: str = "/data/speaker.wav") -> bool:
    """
    Generate audio using the Zonos model with proper error handling and memory management.
    
    Args:
        text: Text to synthesize
        output: Output path for the generated audio
        speaker_audio: Path to the speaker reference audio
        
    Returns:
        bool: True if generation was successful, False otherwise
    """
    # Print system info
    print(f"Generating audio with text: {text}")
    print(f"Using speaker audio: {speaker_audio}")
    print(f"Output will be saved to: {output}")
    
    # Ensure speaker audio exists
    if not os.path.exists(speaker_audio):
        print(f"ERROR: Speaker audio file not found at {speaker_audio}")
        return False
    
    # Try with lower precision to save memory
    try:
        # Check for GPU and available memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if device.type == "cuda":
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Import Zonos modules only after environment checks to avoid crashes
        try:
            from zonos.model import Zonos
            from zonos.conditioning import make_cond_dict
        except ImportError:
            print("ERROR: Failed to import Zonos modules. Check that the package is properly installed.")
            return False
        
        # Free up memory
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Load model with lower precision
        print("Loading Zonos model...")
        start_time = time.time()
        
        # Set a reasonable chunk size for text to avoid OOM errors
        max_text_length = 150
        if len(text) > max_text_length:
            print(f"Text too long ({len(text)} chars), truncating to {max_text_length} chars")
            text = text[:max_text_length] + "..."
        
        try:
            model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device, torch_dtype=torch.float16)
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            return False
        
        # Process audio with proper error handling
        try:
            print("Loading speaker audio...")
            wav, sampling_rate = torchaudio.load(speaker_audio)
            print(f"Speaker audio loaded, shape: {wav.shape}, sampling rate: {sampling_rate}")
            
            print("Creating speaker embedding...")
            speaker = model.make_speaker_embedding(wav, sampling_rate)
            
            print("Preparing conditioning...")
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
            conditioning = model.prepare_conditioning(cond_dict)
            
            print("Generating audio codes...")
            codes = model.generate(conditioning)
            
            print("Decoding audio...")
            wavs = model.autoencoder.decode(codes).cpu()
            
            # Create output directories if they don't exist
            os.makedirs(os.path.dirname(output), exist_ok=True)
            
            print(f"Saving audio to {output}...")
            torchaudio.save(output, wavs[0], model.autoencoder.sampling_rate)
            
            # Force cleanup
            del model, wav, speaker, cond_dict, conditioning, codes, wavs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            print("Audio generation completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during audio generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech using Zonos TTS model")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output path for the audio file")
    parser.add_argument("--speaker_audio", default="/data/speaker.wav", help="Path to speaker reference audio")
    
    args = parser.parse_args()
    
    success = generate_audio(args.text, args.output, args.speaker_audio)
    sys.exit(0 if success else 1)