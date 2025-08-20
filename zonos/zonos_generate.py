import os
import sys
import argparse
import traceback
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def generate_speech(text, output_path, speaker_audio=None, model_name="hybrid", language="en-us"):
    """Generate speech using Zonos TTS engine with proper voice cloning"""
    print(f"Generating speech for text: '{text}'")
    try:
        # Load the model - hybrid model offers a good balance of speed and quality
        print(f"Loading Zonos {model_name} model on device: {device}")
        model_path = f"Zyphra/Zonos-v0.1-{model_name}"
        # Load model - it should use the cache automatically if available
        model = Zonos.from_pretrained(model_path, device=device)
        print("Model loaded successfully")
        
        # Create the speaker embedding if reference audio provided
        if speaker_audio and os.path.exists(speaker_audio):
            print(f"Using speaker audio: {speaker_audio}")
            try:
                # Load the reference audio and create speaker embedding
                wav, sampling_rate = torchaudio.load(speaker_audio)
                speaker = model.make_speaker_embedding(wav, sampling_rate)
                print(f"Created speaker embedding from {speaker_audio}")
                
                # Generate speech with the speaker embedding
                cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
            except Exception as e:
                print(f"Error using reference audio: {str(e)}")
                print("Falling back to default voice")
                # Generate speech without speaker embedding
                cond_dict = make_cond_dict(text=text, language=language)
        else:
            print("Using default speaker")
            # Generate speech without speaker embedding
            cond_dict = make_cond_dict(text=text, language=language)
        
        # Prepare conditioning and generate audio
        print("Preparing conditioning")
        conditioning = model.prepare_conditioning(cond_dict)
        
        print("Generating audio tokens")
        codes = model.generate(conditioning)
        
        print("Decoding audio tokens to waveform")
        wavs = model.autoencoder.decode(codes).cpu()
        
        # Save the audio file
        print(f"Saving audio to {output_path}")
        torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
        
        # Verify the file exists
        if os.path.exists(output_path):
            print(f"Success! Audio saved to {output_path}")
            return True
        else:
            print(f"Error: File not found at {output_path}")
            return False
            
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        traceback.print_exc()
        
        # As a last resort, try using espeak-ng
        try:
            print("Attempting fallback with espeak-ng...")
            result = os.system(f'espeak-ng "{text}" -w "{output_path}"')
            if result == 0 and os.path.exists(output_path):
                print(f"Fallback speech saved to: {output_path}")
                return True
        except Exception as fallback_error:
            print(f"Fallback also failed: {str(fallback_error)}")
            
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate speech using Zonos')
    parser.add_argument('--text', required=True, help='Text to convert to speech')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--speaker_audio', help='Path to speaker reference audio file')
    parser.add_argument('--model', default='hybrid', choices=['transformer', 'hybrid'], help='Model to use (transformer or hybrid)')
    parser.add_argument('--language', default='en-us', help='BCP-47 language tag like en-us, es-es, fr-fr')
    
    args = parser.parse_args()
    success = generate_speech(args.text, args.output, args.speaker_audio, args.model, args.language)
    if not success:
        sys.exit(1)