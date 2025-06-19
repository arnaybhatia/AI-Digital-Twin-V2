import os
import time
import uuid
import shutil
import subprocess
import tempfile

import msgpack
import requests
from dotenv import load_dotenv
import gradio as gr

# ——— Environment ———
API_KEY = None
FISH_API_URL = None

def load_env():
    """Load TMPT_API_KEY and FISH_API_URL from .env."""
    load_dotenv()
    global API_KEY, FISH_API_URL
    API_KEY = os.getenv("TMPT_API_KEY")
    if not API_KEY:
        raise RuntimeError("TMPT_API_KEY not found in .env")
    FISH_API_URL = os.getenv("FISH_API_URL", "http://localhost:8080")


# ——— TMPT API client (old endpoints) ———
BASE_URL = "https://api.tmpt.app/v1"
CLIENT_TOKEN = None
THREAD_ID = None

def make_request(method, endpoint, **kwargs):
    url = BASE_URL + endpoint
    headers = kwargs.pop("headers", {})
    headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    resp = requests.request(method, url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp.json()

def initialize_session():
    """Create client token and thread."""
    global CLIENT_TOKEN, THREAD_ID
    tok = make_request("POST", "/client_token",
                       json={"external_user_id": f"user_{int(time.time())}",
                             "is_reviewable": False})
    CLIENT_TOKEN = tok["client_token_id"]
    th = make_request("POST", "/threads",
                      json={"client_token_id": CLIENT_TOKEN})
    THREAD_ID = th["id"]

def get_response(user_input: str) -> str:
    """Old TMPT system: POST /threads/.../messages then GET /threads/.../reply/..."""
    global CLIENT_TOKEN, THREAD_ID
    if CLIENT_TOKEN is None or THREAD_ID is None:
        initialize_session()

    msg = make_request(
        "POST",
        f"/threads/{THREAD_ID}/messages",
        json={"client_token_id": CLIENT_TOKEN, "text": user_input}
    )
    message_id = msg["id"]

    for _ in range(10):
        try:
            reply = make_request(
                "GET",
                f"/threads/{THREAD_ID}/reply/{message_id}",
                params={"client_token_id": CLIENT_TOKEN, "timeout": 5}
            )
            if reply and "text" in reply:
                return reply["text"]
        except requests.HTTPError:
            pass
        time.sleep(1)

    # fallback: fetch all messages
    msgs = make_request(
        "GET",
        f"/threads/{THREAD_ID}/messages",
        params={"client_token_id": CLIENT_TOKEN}
    )
    for m in reversed(msgs):
        if m.get("speaker") == "agent" and "text" in m:
            return m["text"]
    raise RuntimeError("No reply from TMPT within timeout")


# ——— Voice cloning via Fish-Speech HTTP API ———
def clone_voice_http(text: str, source_wav: str, out_wav: str) -> str:
    """
    Call Fish-Speech /v1/tts endpoint with a reference audio to clone the voice.
    Uses msgpack-encoded request body.
    """
    with open(source_wav, "rb") as f:
        audio_bytes = f.read()

    payload = {
        "text": text,
        "format": "wav",
        "references": [
            {"audio": audio_bytes, "text": text}
        ]
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    resp = requests.post(
        f"{FISH_API_URL}/v1/tts",
        data=packed,
        headers={"Content-Type": "application/msgpack"}
    )
    resp.raise_for_status()
    with open(out_wav, "wb") as out:
        out.write(resp.content)
    return out_wav

def clone_voice_cli(text: str, source_wav: str, out_wav: str) -> str:
    """Fallback CLI with --compile and --half flags."""
    cmd = [
        "fish-speech",
        "--compile", "--half",
        "--input", source_wav,
        "--text", text,
        "--output", out_wav
    ]
    subprocess.run(cmd, check=True)
    return out_wav

def clone_voice(text: str, src_wav: str, out_wav: str) -> str:
    try:
        return clone_voice_http(text, src_wav, out_wav)
    except Exception:
        return clone_voice_cli(text, src_wav, out_wav)


# ——— SadTalker animation + mux ———
def sadtalker_animate(image_path: str, driving_video_path: str,
                      audio_path: str, out_path: str) -> str:
    silent_vid = out_path.replace(".mp4", "_silent.mp4")
    subprocess.run([
        "python", "sadtalker/inference.py",
        "--source_image", image_path,
        "--driving_video", driving_video_path,
        "--result_path", silent_vid
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", silent_vid,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        out_path
    ], check=True)
    try:
        os.remove(silent_vid)
    except OSError:
        pass
    return out_path


# ——— Full pipeline ———
def pipeline(user_text: str, voice_file, image_file, video_file) -> str:
    workdir = tempfile.mkdtemp(prefix="dtwn_")
    uid = uuid.uuid4().hex

    src_wav = os.path.join(workdir, f"{uid}_src.wav")
    img_in = os.path.join(workdir, f"{uid}_img{os.path.splitext(image_file.name)[1]}")
    vid_in = os.path.join(workdir, f"{uid}_drv{os.path.splitext(video_file.name)[1]}")
    shutil.copy(voice_file.name, src_wav)
    shutil.copy(image_file.name, img_in)
    shutil.copy(video_file.name, vid_in)

    assistant_text = get_response(user_text)

    cloned_wav = os.path.join(workdir, f"{uid}_clone.wav")
    clone_voice(assistant_text, src_wav, cloned_wav)

    output_mp4 = os.path.join(workdir, f"{uid}_out.mp4")
    sadtalker_animate(img_in, vid_in, cloned_wav, output_mp4)

    return output_mp4


# ——— Main ———
if __name__ == "__main__":
    load_env()
    initialize_session()

    demo = gr.Blocks()
    with demo:
        gr.Markdown("## AI-Digital-Twin (text → voice-clone → talking-head)")
        txt = gr.Textbox(lines=2,
                         placeholder="Enter your prompt here",
                         label="Input Text")
        with gr.Row():
            voice = gr.File(label="Your voice sample (to clone)",
                            file_types=["audio"])
            img = gr.File(label="Still image (to clone)",
                          file_types=["image"])
            vid = gr.File(label="Driving video (motion)",
                          file_types=["video"])
        btn = gr.Button("Generate Talking Head")
        out = gr.Video(label="Resulting Video")

        btn.click(fn=pipeline,
                  inputs=[txt, voice, img, vid],
                  outputs=out)

    demo.launch(server_name="0.0.0.0", server_port=7860)