import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = """Of course. Here is the previous response formatted as plain text in paragraphs.

Based on common discussions in online cycling communities like Reddit, the general agreement is that no lock is 100% theft-proof. The main goal is to make your bike a more difficult and less appealing target than others nearby. The consensus points to a few key types of locks and strategies for securing your bike effectively.

U-locks are consistently recommended as the gold standard for most cyclists. They provide a great balance of security, weight, and price, and are significantly more resistant to cutting with bolt cutters than cable locks. The most frequently praised models come from top brands like Kryptonite and Abus. Kryptonite's "New York" series, including the Fahgettaboudit and the New York Lock M18-WL, is often cited as one of the toughest options available, offering maximum security at the cost of being quite heavy. Similarly, the Abus Granit X-Plus 540 is highly regar"""
AUDIO_PROMPT_PATH = "/home/arn/Organized_Files/Coding_Things/GitHub/AI-Digital-Twin-V2/data/trainingaudio.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)