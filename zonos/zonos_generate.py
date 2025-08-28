import os
import sys
import argparse
import traceback
import torchaudio
from typing import List

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device


def split_text_into_sentences(text: str, max_tokens: int = 150) -> List[str]:
    """Split text into sentences with a soft max token (word) limit.

    This mirrors the previous logic that lived in app.py so we only pay the model
    load + speaker embedding cost once for a multi‑sentence passage.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    final = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_tokens:
            final.append(s)
        else:
            for i in range(0, len(words), max_tokens):
                chunk = " ".join(words[i : i + max_tokens]).strip()
                if chunk:
                    final.append(chunk)
    if not final:
        final = [text.strip()]
    return final


def generate_speech(
    text: str,
    output_path: str,
    speaker_audio: str = None,
    model_name: str = "hybrid",
    language: str = "en-us",
    split_sentences: bool = True,
) -> bool:
    """Generate speech using Zonos with optional sentence batching.

    Strategy:
      1. Load model once.
      2. Create speaker embedding once.
      3. For each sentence/chunk: prepare conditioning -> generate -> decode.
      4. Concatenate decoded waveforms and write a single output file.
    """
    print(f"Generating speech for text (len={len(text)} chars)")
    try:
        # 1. Load model once
        print(f"Loading Zonos {model_name} model on device: {device}")
        model_path = f"Zyphra/Zonos-v0.1-{model_name}"
        model = Zonos.from_pretrained(model_path, device=device)
        print("Model loaded successfully")

        # 2. Speaker embedding (optional)
        speaker = None
        if speaker_audio and os.path.exists(speaker_audio):
            try:
                print(f"Using speaker audio: {speaker_audio}")
                wav, sr = torchaudio.load(speaker_audio)
                speaker = model.make_speaker_embedding(wav, sr)
                print(f"Created speaker embedding from {speaker_audio}")
            except Exception as e:
                print(f"Warning: failed to create speaker embedding ({e}); using default voice")
        else:
            print("Using default speaker (no reference provided)")

        # 3. Sentence batching
        if split_sentences:
            chunks = split_text_into_sentences(text)
        else:
            chunks = [text]

        print(f"Processing {len(chunks)} chunk(s)")
        decoded_segments = []
        target_sr = None

        for idx, chunk in enumerate(chunks):
            print(f"[Chunk {idx+1}/{len(chunks)}] len={len(chunk)} chars")
            cond_dict = make_cond_dict(text=chunk, speaker=speaker, language=language) if speaker is not None else make_cond_dict(text=chunk, language=language)
            conditioning = model.prepare_conditioning(cond_dict)
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()
            wav_tensor = wavs[0]  # (channels, samples)
            if target_sr is None:
                target_sr = model.autoencoder.sampling_rate
            decoded_segments.append(wav_tensor)

        if not decoded_segments:
            print("No audio segments decoded; aborting.")
            return False

        # 4. Concatenate along time dimension
        import torch

        full_wav = torch.cat(decoded_segments, dim=-1)

        # 5. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving concatenated audio to {output_path}")
        torchaudio.save(output_path, full_wav, target_sr)
        if os.path.exists(output_path):
            print(f"Success! Audio saved to {output_path}")
            return True
        print("Failed to save output file")
        return False

    except Exception as e:
        print(f"Error generating speech: {e}")
        traceback.print_exc()
        # Fallback simple TTS (espeak-ng)
        try:
            print("Attempting fallback with espeak-ng (no voice cloning, no batching)...")
            result = os.system(f'espeak-ng "{text}" -w "{output_path}"')
            if result == 0 and os.path.exists(output_path):
                print(f"Fallback speech saved to: {output_path}")
                return True
        except Exception as fb_err:
            print(f"Fallback also failed: {fb_err}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate speech using Zonos')
    parser.add_argument('--text', required=True, help='Text to convert to speech')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--speaker_audio', help='Path to speaker reference audio file')
    parser.add_argument('--model', default='hybrid', choices=['transformer', 'hybrid'], help='TTS model to use (hybrid recommended for Japanese)')
    parser.add_argument(
        '--language',
        default='en-us',
        choices=['en-us', 'fr-fr', 'de', 'ja', 'ko', 'cmn'],
        help='Supported languages: en-us (English US), fr-fr (French), de (German), ja (Japanese), ko (Korean), cmn (Mandarin Chinese)'
    )
    parser.add_argument('--no_split', action='store_true', help='Disable sentence splitting (generate in one pass)')
    
    args = parser.parse_args()
    success = generate_speech(
        args.text,
        args.output,
        args.speaker_audio,
        args.model,
        args.language,
        split_sentences=not args.no_split,
    )
    if not success:
        sys.exit(1)