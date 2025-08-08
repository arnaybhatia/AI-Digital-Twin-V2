import os
import time
import uuid
import shutil
import subprocess
import tempfile
import re
import requests
from dotenv import load_dotenv
import gradio as gr

# ——— Environment ———
API_KEY = None
CHATTERBOX_API_URL = None


def load_env():
    load_dotenv()
    global API_KEY, CHATTERBOX_API_URL
    API_KEY = os.getenv("TMPT_API_KEY")
    if not API_KEY:
        raise RuntimeError("TMPT_API_KEY not found in .env")
    CHATTERBOX_API_URL = os.getenv("CHATTERBOX_API_URL", "http://localhost:8080")


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
def split_text_into_sentences(text: str, max_tokens: int = 150) -> list:
    """Split text into sentences with natural sentence boundaries and token limit fallback"""
    import re

    # Split by natural sentence endings: . ! ?
    # Keep the punctuation with the sentence
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Clean up sentences and remove empty ones
    sentences = [s.strip() for s in sentences if s.strip()]

    # Further split long sentences by token limit
    final_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_tokens:
            if sentence.strip():  # Only add non-empty sentences
                final_sentences.append(sentence)
        else:
            # Split long sentence into chunks
            for i in range(0, len(words), max_tokens):
                chunk = " ".join(words[i : i + max_tokens])
                if chunk.strip():
                    final_sentences.append(chunk)

    # If no valid sentences found, return original text as single sentence
    if not final_sentences:
        final_sentences = [text.strip()]

    return final_sentences


def clone_voice_sentence(text: str, source_wav: str, out_wav: str) -> str:
    """Clone voice for a single sentence using Chatterbox.

    Ensures the audio_prompt_path is accessible from inside the Chatterbox container
    by copying the provided source_wav into the repo's ./data directory (which is
    volume-mounted to /app/data in the container), then sending that container path
    in the API request.
    """

    # Prepare host/container paths for the audio prompt
    project_root = os.path.abspath(os.path.dirname(__file__))
    host_data_dir = os.path.join(project_root, "data")
    os.makedirs(host_data_dir, exist_ok=True)

    # Generate a unique filename to avoid races across concurrent requests
    prompt_basename = f"voice_prompt_{uuid.uuid4().hex[:8]}.wav"
    host_prompt_path = os.path.join(host_data_dir, prompt_basename)
    container_prompt_path = f"/app/data/{prompt_basename}"

    # Copy the source wav (likely in a temp dir) into the mounted data directory
    try:
        shutil.copy(source_wav, host_prompt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to stage audio prompt for Chatterbox: {e}")

    payload = {
        "text": text,
        "audio_prompt_path": container_prompt_path,
    }

    try:
        resp = requests.post(
            f"{CHATTERBOX_API_URL}/v1/tts",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

        with open(out_wav, "wb") as out:
            out.write(resp.content)
        return out_wav

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 500:
            raise RuntimeError(
                f"Chatterbox server error - likely CUDA/GPU issue. Check container logs: {e}"
            )
        else:
            status = e.response.status_code if e.response is not None else "unknown"
            # Try to surface server-side JSON error if available
            err_detail = None
            try:
                err_detail = e.response.json()
            except Exception:
                pass
            raise RuntimeError(
                f"Chatterbox API error ({status}): {e}. Details: {err_detail}"
            )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Chatterbox API. Ensure container is running on port 8080"
        )
    except Exception as e:
        raise RuntimeError(f"Voice cloning failed: {e}")
    finally:
        # Best-effort cleanup of the staged prompt file
        try:
            if os.path.exists(host_prompt_path):
                os.remove(host_prompt_path)
        except Exception:
            pass


def combine_audio_files(audio_files: list, output_path: str) -> str:
    """Combine multiple audio files using ffmpeg"""
    import tempfile

    # Create a temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for audio_file in audio_files:
            f.write(f"file '{os.path.abspath(audio_file)}'\n")
        file_list_path = f.name

    try:
        # Use ffmpeg to concatenate audio files
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_list_path,
            "-c",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError:
        # Fallback: use ffmpeg with filter_complex for better compatibility
        try:
            inputs = []
            filter_parts = []

            for i, audio_file in enumerate(audio_files):
                inputs.extend(["-i", audio_file])
                filter_parts.append(f"[{i}:0]")

            filter_complex = (
                "".join(filter_parts) + f"concat=n={len(audio_files)}:v=0:a=1[out]"
            )

            cmd = (
                ["ffmpeg", "-y"]
                + inputs
                + ["-filter_complex", filter_complex, "-map", "[out]", output_path]
            )
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            # If both methods fail, return the first audio file
            if audio_files:
                shutil.copy(audio_files[0], output_path)
                return output_path
            raise RuntimeError("Failed to combine audio files")
    finally:
        # Clean up temporary file list
        try:
            os.unlink(file_list_path)
        except:
            pass


def clone_voice_docker(text: str, source_wav: str, out_wav: str) -> str:
    """Clone voice with sentence batching using dedicated temp directory"""
    # Create dedicated temp directory for this operation
    project_root = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Create unique subdirectory for this operation
    operation_id = uuid.uuid4().hex[:8]
    operation_temp_dir = os.path.join(temp_dir, f"voice_clone_{operation_id}")
    os.makedirs(operation_temp_dir, exist_ok=True)

    try:
        # Split text into manageable sentences
        sentences = split_text_into_sentences(text)

        if len(sentences) == 1:
            # Single sentence, process directly to output
            return clone_voice_sentence(text, source_wav, out_wav)

        # Multiple sentences, batch process in temp directory
        temp_audio_files = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            temp_audio_path = os.path.join(operation_temp_dir, f"sentence_{i:03d}.wav")
            clone_voice_sentence(sentence.strip(), source_wav, temp_audio_path)
            temp_audio_files.append(temp_audio_path)

        if not temp_audio_files:
            raise RuntimeError("No valid sentences to process")

        if len(temp_audio_files) == 1:
            # Only one valid sentence, move directly to output
            shutil.move(temp_audio_files[0], out_wav)
        else:
            # Combine multiple audio files
            combine_audio_files(temp_audio_files, out_wav)

        return out_wav

    finally:
        # Clean up the entire operation temp directory
        try:
            if os.path.exists(operation_temp_dir):
                shutil.rmtree(operation_temp_dir)
        except Exception as e:
            print(
                f"Warning: Failed to clean up temp directory {operation_temp_dir}: {e}"
            )


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
def pipeline(
    user_text: str,
    voice_file,
    image_file,
    video_file,
    use_ai: bool,
    history: list,
    progress=gr.Progress(),
):
    workdir = tempfile.mkdtemp(prefix="dtwn_")
    uid = uuid.uuid4().hex
    generation_time = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Clear outputs and show initial progress
        progress(0.1, desc="🔄 Starting generation...")
        yield (
            "🔄 Starting generation...",  # api_response
            None,  # audio_out
            None,  # video_out
            history,  # history unchanged
        )

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
        progress(
            0.2,
            desc="🧠 Generating AI response..." if use_ai else "📝 Processing text...",
        )
        yield (
            "🧠 Generating AI response..." if use_ai else "📝 Processing text...",
            None,
            None,
            history,
        )

        if use_ai:
            assistant_text = get_response(user_text)
        else:
            assistant_text = user_text

        progress(0.4, desc="✅ Text ready")
        yield (assistant_text, None, None, history)

        # 2) Clone the voice
        progress(0.5, desc="🎤 Cloning voice...")
        yield (assistant_text, None, None, history)

        # Clone voice to temp, then persist to results so it remains accessible
        cloned_wav_tmp = os.path.join(workdir, f"{uid}_clone.wav")
        clone_voice(assistant_text, src_wav, cloned_wav_tmp)

        # Persist audio in results directory for playback and history
        project_root = os.path.abspath(os.path.dirname(__file__))
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        cloned_wav = os.path.join(results_dir, f"audio_{uid}.wav")
        try:
            shutil.copy(cloned_wav_tmp, cloned_wav)
        except Exception:
            # If persisting fails, fallback to temp path (may be cleaned later)
            cloned_wav = cloned_wav_tmp

        progress(0.7, desc="✅ Voice cloned")
        yield (assistant_text, cloned_wav, None, history)

        # 3) Animate with SadTalker
        progress(0.8, desc="🎭 Creating talking avatar...")
        yield (assistant_text, cloned_wav, None, history)

        final_mp4 = sadtalker_animate(img_in, vid_in, cloned_wav)

        # Add to history
        history_entry = {
            "timestamp": generation_time,
            "input": user_text,
            "response": assistant_text,
            "audio": cloned_wav,
            "video": final_mp4,
            "mode": "AI Generated" if use_ai else "Raw Text",
        }
        history.append(history_entry)

        # Keep only last 10 generations
        if len(history) > 10:
            history = history[-10:]

        progress(1.0, desc="✅ Complete!")
        yield (assistant_text, cloned_wav, final_mp4, history)

    except Exception as e:
        progress(0, desc="❌ Failed")
        yield (f"❌ Error: {str(e)}", None, None, history)
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
        # State for generation history
        history_state = gr.State([])

        gr.HTML('<div class="main-header">AI Digital Twin V2</div>')
        gr.HTML(
            '<div class="description">Create personalized talking avatars with AI-powered voice cloning and facial animation</div>'
        )

        with gr.Row():
            # Left column - Input and Output
            with gr.Column(scale=2):
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
                        voice = gr.File(
                            label="Voice Sample", file_types=["audio"], height=100
                        )
                        img = gr.File(
                            label="Portrait Image", file_types=["image"], height=100
                        )
                        vid = gr.File(
                            label="Driving Video", file_types=["video"], height=100
                        )

                with gr.Row():
                    btn = gr.Button(
                        "Generate Digital Twin",
                        variant="primary",
                        size="lg",
                        elem_classes="generate-btn",
                    )
                    clear_btn = gr.Button(
                        "Clear Outputs", variant="secondary", size="lg"
                    )

                with gr.Column(elem_classes="output-section"):
                    gr.HTML('<div class="section-header">Generated Response</div>')
                    api_response = gr.Textbox(
                        label="AI Response", interactive=False, lines=3
                    )

                    gr.HTML('<div class="section-header">Generated Media</div>')
                    with gr.Row():
                        audio_out = gr.Audio(label="Cloned Voice", type="filepath", streaming=True, autoplay=True)
                        video_out = gr.Video(label="Talking Avatar", streaming=True, autoplay=True)

            # Right column - Generation History
            with gr.Column(scale=1, elem_classes="input-section"):
                gr.HTML('<div class="section-header">Generation History</div>')

                def format_history(history):
                    if not history:
                        return "No generations yet. Create your first digital twin!"

                    formatted = ""
                    for i, entry in enumerate(reversed(history[-5:])):  # Show last 5
                        formatted += f"""
                        **#{len(history) - i} - {entry["timestamp"]}**
                        - **Input:** {entry["input"][:50]}{"..." if len(entry["input"]) > 50 else ""}
                        - **Mode:** {entry["mode"]}
                        - **Response:** {entry["response"][:100]}{"..." if len(entry["response"]) > 100 else ""}
                        
                        ---
                        """
                    return formatted

                history_display = gr.Markdown(
                    value="No generations yet. Create your first digital twin!",
                    label="Recent Generations",
                )

                # History controls
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh", size="sm")
                    clear_history_btn = gr.Button("🗑️ Clear History", size="sm")

        def clear_outputs():
            return "", None, None

        def clear_history():
            return [], "No generations yet. Create your first digital twin!"

        def refresh_history_display(history):
            return format_history(history)

        # Event handlers
        clear_btn.click(fn=clear_outputs, outputs=[api_response, audio_out, video_out])

        clear_history_btn.click(
            fn=clear_history, outputs=[history_state, history_display]
        )

        refresh_history_btn.click(
            fn=refresh_history_display,
            inputs=[history_state],
            outputs=[history_display],
        )

        btn.click(
            fn=pipeline,
            inputs=[txt, voice, img, vid, use_ai_toggle, history_state],
            outputs=[api_response, audio_out, video_out, history_state],
        ).then(
            fn=refresh_history_display,
            inputs=[history_state],
            outputs=[history_display],
        )

if __name__ == "__main__":
    # Enable queue to ensure reliable streaming and long-running task handling
    demo.queue()
    demo.launch(server_name="127.0.0.1")
