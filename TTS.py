from kokoro import KPipeline
import soundfile as sf
from playsound import playsound
import time
import os
import threading
from typing import Optional

class TextToSpeech:
    def __init__(self, model_dir="./models"):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.pipeline = KPipeline(lang_code='a')
        self.temp_dir = "./temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def speak(self, text: str, interrupt_event: Optional[threading.Event] = None):
        try:
            audio_iter = self.pipeline(
                text,
                voice='af_heart',
                speed=1.0,
                split_pattern=r'\n+'
            )

            for i, (graphemes, phonemes, audio_data) in enumerate(audio_iter):
                if interrupt_event and interrupt_event.is_set():
                    print("TTS interrupted")
                    break

                wav_path = os.path.join(self.temp_dir, f"kokoro_output_{i}.wav")

                sf.write(wav_path, audio_data, 24000)

                playsound(wav_path)

                time.sleep(0.2)

                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {wav_path}: {e}")

        except Exception as e:
            print(f"[Kokoro TTS] Error: {e}")

    def cleanup(self):
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("kokoro_output_") and file.endswith(".wav"):
                    os.remove(os.path.join(self.temp_dir, file))
        except Exception as e:
            print(f"[Kokoro TTS] Cleanup error: {e}")
