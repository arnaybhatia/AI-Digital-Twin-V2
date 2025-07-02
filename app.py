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
    headers.update(
        {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
    )
    resp = requests.request(method, url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp.json()


def initialize_session():
    global CLIENT_TOKEN, THREAD_ID
    tok = make_request(
        "POST",
        "/client_token",
        json={"external_user_id": f"user_{int(time.time())}", "is_reviewable": False},
    )
    CLIENT_TOKEN = tok["client_token_id"]
    th = make_request("POST", "/threads", json={"client_token_id": CLIENT_TOKEN})
    THREAD_ID = th["id"]


def get_response(user_input: str) -> str:
    if CLIENT_TOKEN is None or THREAD_ID is None:
        initialize_session()

    msg = make_request(
        "POST",
        f"/threads/{THREAD_ID}/messages",
        json={"client_token_id": CLIENT_TOKEN, "text": user_input},
    )
    message_id = msg["id"]

    # Poll for reply...
    for _ in range(10):
        try:
            reply = make_request(
                "GET",
                f"/threads/{THREAD_ID}/reply/{message_id}",
                params={"client_token_id": CLIENT_TOKEN, "timeout": 5},
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
        params={"client_token_id": CLIENT_TOKEN},
    )
    for m in reversed(msgs):
        if m.get("speaker") == "agent" and "text" in m:
            return m["text"]
    raise RuntimeError("No reply from TMPT within timeout")


# ——— Voice cloning ———
def clone_voice_docker(text: str, source_wav: str, out_wav: str) -> str:
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
        headers={"Content-Type": "application/msgpack"},
    )
    resp.raise_for_status()
    with open(out_wav, "wb") as out:
        out.write(resp.content)
    return out_wav


def clone_voice(text: str, src_wav: str, out_wav: str) -> str:
    return clone_voice_docker(text, src_wav, out_wav)


# ——— SadTalker animation ———
def sadtalker_animate(
    image_path: str, _driving_video_path: str, audio_path: str
) -> str:
    project_root = os.path.abspath(os.path.dirname(__file__))
    run_id = uuid.uuid4().hex
    host_data_dir = os.path.join(project_root, "data", run_id)
    os.makedirs(host_data_dir, exist_ok=True)

    img_fn = os.path.basename(image_path)
    wav_fn = os.path.basename(audio_path)
    host_img = os.path.join(host_data_dir, img_fn)
    host_wav = os.path.join(host_data_dir, wav_fn)
    shutil.copy(image_path, host_img)
    shutil.copy(audio_path, host_wav)

    cont_img = f"/app/data/{run_id}/{img_fn}"
    cont_wav = f"/app/data/{run_id}/{wav_fn}"

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "sadtalker",
        "python",
        "inference.py",
        "--driven_audio",
        cont_wav,
        "--source_image",
        cont_img,
        "--result_dir",
        "/app/results",
        "--preprocess",
        "full",
        "--still",
    ]
    res = subprocess.run(
        cmd, cwd=project_root, check=True, capture_output=True, text=True
    )

    log = (res.stdout or "") + "\n" + (res.stderr or "")
    pattern = r"The generated video is named[: ]+(\S+\.mp4)"
    matches = re.findall(pattern, log)

    generated_video_path = None
    if matches:
        cont_path = matches[-1]
        host_path = cont_path.replace(
            "/app/results", os.path.join(project_root, "results")
        )
        host_path = os.path.normpath(host_path)
        if os.path.isfile(host_path):
            generated_video_path = host_path
        elif os.path.isdir(host_path):
            for fn in sorted(os.listdir(host_path)):
                if fn.endswith(".mp4"):
                    generated_video_path = os.path.join(host_path, fn)
                    break

    # Fallback search
    if not generated_video_path:
        host_results = os.path.join(project_root, "results")
        subdirs = [
            d
            for d in os.listdir(host_results)
            if os.path.isdir(os.path.join(host_results, d))
        ]
        if subdirs:
            newest = max(
                subdirs, key=lambda d: os.path.getctime(os.path.join(host_results, d))
            )
            result_dir = os.path.join(host_results, newest)
            for fn in os.listdir(result_dir):
                if fn.endswith(".mp4"):
                    generated_video_path = os.path.join(result_dir, fn)
                    break

    if not generated_video_path:
        # Clean up temporary files before raising error
        try:
            shutil.rmtree(host_data_dir)
        except:
            pass
        raise RuntimeError("No .mp4 found in SadTalker results")

    # Convert video to web-compatible format for Gradio
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"output_{run_id}.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        generated_video_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        # Clean up temporary files
        shutil.rmtree(host_data_dir)
        return output_path
    except subprocess.CalledProcessError:
        # If ffmpeg fails, return original file and clean up
        shutil.rmtree(host_data_dir)
        return generated_video_path


# ——— Generator pipeline ———
def pipeline(user_text: str, voice_file, image_file, video_file, use_ai: bool):
    workdir = tempfile.mkdtemp(prefix="dtwn_")
    uid = uuid.uuid4().hex

    try:
        src_wav = os.path.join(workdir, f"{uid}_src.wav")
        img_in = os.path.join(
            workdir, f"{uid}_img{os.path.splitext(image_file.name)[1]}"
        )
        vid_in = os.path.join(
            workdir, f"{uid}_drv{os.path.splitext(video_file.name)[1]}"
        )

        shutil.copy(voice_file.name, src_wav)
        shutil.copy(image_file.name, img_in)
        shutil.copy(video_file.name, vid_in)

        # 1) Get text (either from AI or use raw input)
        if use_ai:
            assistant_text = get_response(user_text)
        else:
            assistant_text = user_text
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

    finally:
        # Clean up temporary working directory
        try:
            shutil.rmtree(workdir)
        except:
            pass


# ——— Gradio UI ———
if __name__ == "__main__":
    load_env()
    initialize_session()

    # Custom CSS for modern black/gray dark theme
    custom_css = """
    .gradio-container {
        background: #0a0a0a !important;
        color: #e0e0e0 !important;
    }
    
    /* Text inputs */
    .gr-textbox, .gr-textbox textarea {
        background: #1a1a1a !important;
        border: 1px solid #404040 !important;
        color: #e0e0e0 !important;
        border-radius: 6px !important;
    }
    
    .gr-textbox:focus, .gr-textbox textarea:focus {
        border-color: #606060 !important;
        box-shadow: 0 0 0 2px rgba(96, 96, 96, 0.2) !important;
    }
    
    /* File inputs */
    .file-upload {
        background: #1a1a1a !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    /* Buttons */
    .gr-button {
        background: #2a2a2a !important;
        border: 1px solid #505050 !important;
        color: #e0e0e0 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    
    .gr-button:hover {
        background: #353535 !important;
        border-color: #606060 !important;
    }
    
    .generate-btn {
        background: #1f1f1f !important;
        border: 1px solid #505050 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    
    .generate-btn:hover {
        background: #2a2a2a !important;
        border-color: #707070 !important;
    }
    
    /* Checkbox */
    .gr-checkbox {
        background: #1a1a1a !important;
    }
    
    /* Audio/Video components */
    .gr-audio, .gr-video {
        background: #1a1a1a !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    /* Labels */
    .gr-label {
        color: #c0c0c0 !important;
        font-weight: 500 !important;
    }
    
    /* Sections */
    .input-section {
        background: #121212 !important;
        border: 1px solid #303030 !important;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .output-section {
        background: #121212 !important;
        border: 1px solid #303030 !important;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .toggle-container {
        background: #1a1a1a !important;
        border: 1px solid #404040 !important;
        border-radius: 6px;
        padding: 16px;
    }
    
    /* Headers */
    .main-header {
        text-align: center;
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        color: #d0d0d0;
        font-size: 18px;
        font-weight: 600;
        margin: 0 0 16px 0;
        border-bottom: 1px solid #303030;
        padding-bottom: 8px;
    }
    
    .description {
        text-align: center;
        color: #909090;
        font-size: 14px;
        margin-bottom: 32px;
        line-height: 1.4;
    }
    
    .progress-info {
        text-align: center;
        color: #808080;
        font-size: 13px;
        margin: 16px 0;
        font-style: italic;
    }
    """

    demo = gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="neutral", secondary_hue="neutral", neutral_hue="zinc"
        ).set(
            body_background_fill="#0a0a0a",
            block_background_fill="#121212",
            border_color_primary="#303030",
            input_background_fill="#1a1a1a",
            button_primary_background_fill="#2a2a2a",
        ),
        css=custom_css,
        title="AI Digital Twin V2",
    )

    with demo:
        gr.HTML('<div class="main-header">AI Digital Twin V2</div>')
        gr.HTML(
            '<div class="description">Create personalized talking avatars with AI-powered voice cloning and facial animation</div>'
        )

        with gr.Column(elem_classes="input-section"):
            gr.HTML('<div class="section-header">Input Configuration</div>')

            with gr.Row():
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        lines=3,
                        placeholder="Enter your message or prompt here...",
                        label="Text Input",
                        container=True,
                    )
                with gr.Column(scale=1, elem_classes="toggle-container"):
                    use_ai_toggle = gr.Checkbox(
                        label="Use AI Response",
                        value=True,
                        info="Toggle between AI generation and raw text",
                    )

            gr.HTML('<div class="section-header">Media Files</div>')
            with gr.Row():
                voice = gr.File(label="Voice Sample", file_types=["audio"], height=100)
                img = gr.File(label="Portrait Image", file_types=["image"], height=100)
                vid = gr.File(label="Driving Video", file_types=["video"], height=100)

        with gr.Row():
            btn = gr.Button(
                "Generate Digital Twin",
                variant="primary",
                size="lg",
                elem_classes="generate-btn",
            )

        gr.HTML(
            '<div class="progress-info">Results will appear progressively as each stage completes</div>'
        )

        with gr.Column(elem_classes="output-section"):
            gr.HTML('<div class="section-header">Generated Response</div>')
            api_response = gr.Textbox(label="AI Response", interactive=False, lines=3)

            gr.HTML('<div class="section-header">Generated Media</div>')
            with gr.Row():
                audio_out = gr.Audio(label="Cloned Voice", type="filepath")
                video_out = gr.Video(label="Talking Avatar")

        btn.click(
            fn=pipeline,
            inputs=[txt, voice, img, vid, use_ai_toggle],
            outputs=[api_response, audio_out, video_out],
        )

    demo.launch(server_name="127.0.0.1")

    with demo:
        gr.HTML('<div class="main-header">AI Digital Twin V2</div>')
        gr.HTML(
            '<div style="text-align: center; color: #888; margin-bottom: 2em;">Create your personalized talking avatar with AI-powered voice cloning and facial animation</div>'
        )

        with gr.Column(elem_classes="input-section"):
            gr.HTML('<div class="section-header">Input Configuration</div>')

            with gr.Row():
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        lines=3,
                        placeholder="Enter your message or prompt here...",
                        label="Text Input",
                        container=True,
                        show_label=True,
                    )
                with gr.Column(scale=1, elem_classes="toggle-container"):
                    use_ai_toggle = gr.Checkbox(
                        label="Use AI Response",
                        value=True,
                        info="Toggle to use AI or raw text",
                        container=True,
                    )

            gr.HTML('<div class="section-header">Media Upload</div>')
            with gr.Row():
                voice = gr.File(
                    label="Voice Sample",
                    file_types=["audio"],
                    container=True,
                    height=120,
                )
                img = gr.File(
                    label="Portrait Image",
                    file_types=["image"],
                    container=True,
                    height=120,
                )
                vid = gr.File(
                    label="Driving Video",
                    file_types=["video"],
                    container=True,
                    height=120,
                )

        with gr.Row():
            btn = gr.Button(
                "Generate Digital Twin",
                variant="primary",
                size="lg",
                elem_classes="generate-btn",
            )

        gr.HTML(
            '<div style="text-align: center; color: #888; margin: 10px 0;">Outputs will appear as soon as they\'re ready...</div>'
        )

        with gr.Column(elem_classes="output-section"):
            gr.HTML('<div class="section-header">AI Response</div>')
            api_response = gr.Textbox(
                label="Assistant Response",
                interactive=False,
                lines=3,
                container=True,
                show_label=True,
            )

            gr.HTML('<div class="section-header">Generated Media</div>')
            with gr.Row():
                audio_out = gr.Audio(
                    label="Cloned Voice",
                    type="filepath",
                    container=True,
                    show_label=True,
                )
                video_out = gr.Video(
                    label="Talking Avatar", container=True, show_label=True
                )

        btn.click(
            fn=pipeline,
            inputs=[txt, voice, img, vid, use_ai_toggle],
            outputs=[api_response, audio_out, video_out],
        )

    demo.launch(
        server_name="127.0.0.1", show_api=False, show_error=True, favicon_path=None
    )

    with demo:
        gr.HTML('<div class="main-header">🎭 AI Digital Twin V2</div>')
        gr.HTML(
            '<div style="text-align: center; color: #888; margin-bottom: 2em;">Create your personalized talking avatar with AI-powered voice cloning and facial animation</div>'
        )

        with gr.Column(elem_classes="input-section"):
            gr.HTML('<div class="section-header">💬 Input Configuration</div>')

            with gr.Row():
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        lines=3,
                        placeholder="💭 Enter your message or prompt here...",
                        label="📝 Text Input",
                        container=True,
                        show_label=True,
                    )
                with gr.Column(scale=1, elem_classes="toggle-container"):
                    use_ai_toggle = gr.Checkbox(
                        label="🧠 AI Response Mode",
                        value=True,
                        info="✨ Toggle to use AI or raw text",
                        container=True,
                    )

            gr.HTML('<div class="section-header">📁 Media Upload</div>')
            with gr.Row():
                voice = gr.File(
                    label="🎤 Voice Sample",
                    file_types=["audio"],
                    container=True,
                    height=120,
                )
                img = gr.File(
                    label="📸 Portrait Image",
                    file_types=["image"],
                    container=True,
                    height=120,
                )
                vid = gr.File(
                    label="🎬 Driving Video",
                    file_types=["video"],
                    container=True,
                    height=120,
                )

        with gr.Row():
            btn = gr.Button(
                "🚀 Generate Digital Twin",
                variant="primary",
                size="lg",
                elem_classes="generate-btn",
            )

        gr.HTML(
            '<div class="progress-text">⏳ Outputs will appear as soon as they\'re ready...</div>'
        )

        with gr.Column(elem_classes="output-section"):
            gr.HTML('<div class="section-header">🎯 AI Response</div>')
            api_response = gr.Textbox(
                label="🤖 Assistant Response",
                interactive=False,
                lines=3,
                container=True,
                show_label=True,
            )

            gr.HTML('<div class="section-header">🎨 Generated Media</div>')
            with gr.Row():
                audio_out = gr.Audio(
                    label="🔊 Cloned Voice",
                    type="filepath",
                    container=True,
                    show_label=True,
                )
                video_out = gr.Video(
                    label="🎭 Talking Avatar", container=True, show_label=True
                )

        btn.click(
            fn=pipeline,
            inputs=[txt, voice, img, vid, use_ai_toggle],
            outputs=[api_response, audio_out, video_out],
        )

    demo.launch(
        server_name="127.0.0.1", show_api=False, show_error=True, favicon_path=None
    )
