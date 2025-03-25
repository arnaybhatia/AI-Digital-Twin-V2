import subprocess
import requests
import os

def query_api(text):
    response = requests.get("https://your-api.com", params={"query": text})
    return response.text

def run_whisper(input_audio, output_text):
    cmd = [
        "docker", "run", "--rm", "-v", f"{os.getcwd()}/data:/data", "whisper",
        "python3", "whisper_transcribe.py", "--file", input_audio, "--output", output_text
    ]
    subprocess.run(cmd, check=True)
    with open("./data/transcript.txt") as f:
        return f.read()

def run_zonos(text, output_audio, speaker_audio="/data/speaker.wav"):
    cmd = [
        "docker", "run", "--rm", "-v", f"{os.getcwd()}/data:/data", "zonos",
        "python3", "zonos_generate.py", "--text", text, "--output", output_audio, "--speaker_audio", speaker_audio
    ]
    subprocess.run(cmd, check=True)

def run_kdtalker(audio, image, output_video):
    cmd = [
        "docker", "run", "--rm", "-v", f"{os.getcwd()}/data:/data", "kdtalker",
        "python3", "inference.py", "--source_image", image, "--driven_audio", audio, "--output", output_video
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Input files in ./data/
    input_audio = "/data/user_input.wav"
    transcript_output = "/data/transcript.txt"
    synthesized_audio = "/data/synthesized.wav"
    speaker_audio = "/data/speaker.wav"
    source_image = "/data/source_image.png"
    output_video = "/data/output.mp4"

    # Workflow
    transcript = run_whisper(input_audio, transcript_output)
    api_response = query_api(transcript)
    run_zonos(api_response, synthesized_audio, speaker_audio)
    run_kdtalker(synthesized_audio, source_image, output_video)
    print(f"AI twin video generated at {output_video}")