import os
import time
import uuid
import shutil
import subprocess
import tempfile
import re
import msgpack
import requests
from dotenv import load_dotenv
import gradio as gr

# ——— Environment ———
API_KEY = None
FISH_API_URL = None

def load_env():
  load_dotenv()
  global API_KEY, FISH_API_URL
  API_KEY = os.getenv("TMPT_API_KEY")
  if not API_KEY:
    raise RuntimeError("TMPT_API_KEY not found in .env")
  FISH_API_URL = os.getenv("FISH_API_URL", "http://localhost:8080")

# ——— TMPT client ———
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
  global CLIENT_TOKEN, THREAD_ID
  tok = make_request("POST", "/client_token",
                     json={"external_user_id": f"user_{int(time.time())}",
                           "is_reviewable": False})
  CLIENT_TOKEN = tok["client_token_id"]
  th = make_request("POST", "/threads",
                    json={"client_token_id": CLIENT_TOKEN})
  THREAD_ID = th["id"]

def get_response(user_input: str) -> str:
  if CLIENT_TOKEN is None or THREAD_ID is None:
    initialize_session()

  msg = make_request(
    "POST",
    f"/threads/{THREAD_ID}/messages",
    json={"client_token_id": CLIENT_TOKEN, "text": user_input}
  )
  message_id = msg["id"]

  # Poll for reply...
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

  # Fallback scan
  msgs = make_request(
    "GET",
    f"/threads/{THREAD_ID}/messages",
    params={"client_token_id": CLIENT_TOKEN}
  )
  for m in reversed(msgs):
    if m.get("speaker") == "agent" and "text" in m:
      return m["text"]
  raise RuntimeError("No reply from TMPT within timeout")

# ——— Voice cloning ———
def clone_voice_http(text: str, source_wav: str, out_wav: str) -> str:
  with open(source_wav, "rb") as f:
    audio_bytes = f.read()
  payload = {
    "text": text,
    "format": "wav",
    "references": [{"audio": audio_bytes, "text": text}],
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
  cmd = [
    "fish-speech", "--compile", "--half",
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

# ——— SadTalker animation ———
def sadtalker_animate(image_path: str,
                      _driving_video_path: str,
                      audio_path: str) -> str:
  project_root = os.path.abspath(os.path.dirname(__file__))
  run_id = uuid.uuid4().hex
  host_data_dir = os.path.join(project_root, "data", run_id)
  os.makedirs(host_data_dir, exist_ok=True)

  img_fn = os.path.basename(image_path)
  wav_fn = os.path.basename(audio_path)
  host_img = os.path.join(host_data_dir, img_fn)
  host_wav = os.path.join(host_data_dir, wav_fn)
  shutil.copy(image_path, host_img)
  shutil.copy(audio_path,  host_wav)

  cont_img = f"/app/data/{run_id}/{img_fn}"
  cont_wav = f"/app/data/{run_id}/{wav_fn}"

  cmd = [
    "docker", "compose", "exec", "-T", "sadtalker",
    "python", "inference.py",
    "--driven_audio", cont_wav,
    "--source_image", cont_img,
    "--result_dir", "/app/results",
    "--preprocess", "full",
    "--still"
  ]
  res = subprocess.run(
    cmd,
    cwd=project_root,
    check=True,
    capture_output=True,
    text=True
  )

  log = (res.stdout or "") + "\n" + (res.stderr or "")
  pattern = r"The generated video is named[: ]+(\S+\.mp4)"
  matches = re.findall(pattern, log)
  if matches:
    cont_path = matches[-1]
    host_path = cont_path.replace(
      "/app/results",
      os.path.join(project_root, "results")
    )
    host_path = os.path.normpath(host_path)
    if os.path.isfile(host_path):
      return host_path
    if os.path.isdir(host_path):
      for fn in sorted(os.listdir(host_path)):
        if fn.endswith(".mp4"):
          return os.path.join(host_path, fn)

  # fallback
  host_results = os.path.join(project_root, "results")
  subdirs = [
    d for d in os.listdir(host_results)
    if os.path.isdir(os.path.join(host_results, d))
  ]
  if not subdirs:
    raise RuntimeError("SadTalker produced no results")
  newest = max(
    subdirs,
    key=lambda d: os.path.getctime(os.path.join(host_results, d))
  )
  result_dir = os.path.join(host_results, newest)
  for fn in os.listdir(result_dir):
    if fn.endswith(".mp4"):
      return os.path.join(result_dir, fn)
  raise RuntimeError("No .mp4 in SadTalker results")

# ——— Generator pipeline ———
def pipeline(user_text: str, voice_file, image_file, video_file):
    workdir = tempfile.mkdtemp(prefix="dtwn_")
    uid = uuid.uuid4().hex

    src_wav = os.path.join(workdir, f"{uid}_src.wav")
    img_in  = os.path.join(
      workdir, f"{uid}_img{os.path.splitext(image_file.name)[1]}"
    )
    vid_in  = os.path.join(
      workdir, f"{uid}_drv{os.path.splitext(video_file.name)[1]}"
    )

    shutil.copy(voice_file.name, src_wav)
    shutil.copy(image_file.name, img_in)
    shutil.copy(video_file.name, vid_in)

    # 1) Ask the TMPT API
    assistant_text = get_response(user_text)
    # Update only the Textbox, leave audio/video untouched
    yield assistant_text, gr.update(), gr.update()

    # 2) Clone the voice
    cloned_wav = os.path.join(workdir, f"{uid}_clone.wav")
    clone_voice(assistant_text, src_wav, cloned_wav)
    # Update only the Audio, leave text/video untouched
    yield gr.update(), cloned_wav, gr.update()

    # 3) Animate with SadTalker
    final_mp4 = sadtalker_animate(img_in, vid_in, cloned_wav)
    # Update only the Video, leave text/audio untouched
    yield gr.update(), gr.update(), final_mp4

# ——— Gradio UI ———
if __name__ == "__main__":
    load_env()
    initialize_session()

    demo = gr.Blocks()
    with demo:
        gr.Markdown("## AI-Digital-Twin V2  \n"
                    "**Outputs appear as soon as they're ready**")

        with gr.Row():
            txt = gr.Textbox(
                lines=2,
                placeholder="Enter your prompt here",
                label="Input Text"
            )
        with gr.Row():
            voice = gr.File(label="Your voice sample (to clone)",
                            file_types=["audio"])
            img   = gr.File(label="Still image (to clone)",
                            file_types=["image"])
            vid   = gr.File(label="Driving video (motion)",
                            file_types=["video"])

        btn = gr.Button("Generate")

        # New Textbox to show the API response
        api_response = gr.Textbox(label="Assistant Response",
                                  interactive=False)

        with gr.Row():
            audio_out = gr.Audio(label="Cloned Voice",
                                 type="filepath")
            video_out = gr.Video(label="Talking-Head Video")

        btn.click(
            fn=pipeline,
            inputs=[txt, voice, img, vid],
            outputs=[api_response, audio_out, video_out]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860)