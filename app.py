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
ZONOS_API_URL = None


def load_env():
    load_dotenv()
    global API_KEY, ZONOS_API_URL
    API_KEY = os.getenv("TMPT_API_KEY")
    if not API_KEY:
        raise RuntimeError("TMPT_API_KEY not found in .env")
    ZONOS_API_URL = os.getenv("ZONOS_API_URL", "http://localhost:8090")


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


def get_response(user_input: str, language: str = "en-us") -> str:
    if CLIENT_TOKEN is None or THREAD_ID is None:
        initialize_session()

    # Add language instruction to the user input with better prompting
    language_prompts = {
        "en-us": "",  # No additional prompt for English
        "fr-fr": "Respond naturally in French, maintaining a conversational and expressive tone: ",
        "de": "Respond naturally in German, maintaining a conversational and expressive tone: ",
        "ja": "Respond naturally in Japanese, maintaining a conversational and expressive tone: ",
        "ko": "Respond naturally in Korean, maintaining a conversational and expressive tone: ",
        "cmn": "Respond naturally in Mandarin Chinese, maintaining a conversational and expressive tone: ",
    }
    
    language_prompt = language_prompts.get(language, "")
    enhanced_input = language_prompt + user_input
    
    msg = make_request(
        "POST",
        f"/threads/{THREAD_ID}/messages",
        json={"client_token_id": CLIENT_TOKEN, "text": enhanced_input},
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


def _exec_sentence_tts(text: str, source_wav: str, out_wav: str, language: str) -> str:
    """Execute TTS for a single sentence via Zonos docker container."""
    project_root = os.path.abspath(os.path.dirname(__file__))
    host_data_dir = os.path.join(project_root, "data")
    os.makedirs(host_data_dir, exist_ok=True)

    # Stage speaker reference
    prompt_basename = f"voice_prompt_{uuid.uuid4().hex[:8]}.wav"
    host_prompt_path = os.path.join(host_data_dir, prompt_basename)
    container_prompt_path = f"/app/data/{prompt_basename}"
    shutil.copy(source_wav, host_prompt_path)

    out_basename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    host_intermediate_out = os.path.join(host_data_dir, out_basename)
    container_out_path = f"/app/data/{out_basename}"

    cmd = [
        "docker", "compose", "exec", "zonos", "python", "-u", "zonos_generate.py",
        "--text", text,
        "--output", container_out_path,
        "--speaker_audio", container_prompt_path,
        "--model", "transformer",
        "--language", language,
    ]
    print("[zonos] Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    if proc.stdout:
        for line in proc.stdout:
            print("[zonos]", line, end="", flush=True)
    ret = proc.wait()
    try:
        if ret != 0:
            raise RuntimeError(f"Zonos generation failed (exit {ret})")
        shutil.move(host_intermediate_out, out_wav)
        return out_wav
    finally:
        for p in (host_prompt_path, host_intermediate_out):
            try:
                if os.path.exists(p):
                    os.remove(p)
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
        print(f"🔗 Combining {len(audio_files)} audio files...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
        else:
            raise RuntimeError(f"ffmpeg concat exited with code {result.returncode}")
    except Exception as e:
        print(f"❌ FFmpeg concat failed: {e}")
        print(f"📄 Command: {' '.join(cmd)}")
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
            print(f"🔗 Using filter_complex to combine audio files...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
            else:
                raise RuntimeError(f"ffmpeg filter_complex exited with code {result.returncode}")
        except Exception as e:
            print(f"❌ FFmpeg filter_complex failed: {e}")
            print(f"📄 Command: {' '.join(cmd)}")
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


def clone_voice_docker(text: str, source_wav: str, out_wav: str, language: str = "en-us") -> str:
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
            return _exec_sentence_tts(text, source_wav, out_wav, language)

        # Multiple sentences, batch process in temp directory
        temp_audio_files = []
        print(f"🎤 Processing {len(sentences)} sentences for voice cloning...")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            print(f"🎤 Processing sentence {i+1}/{len(sentences)}")
            temp_audio_path = os.path.join(operation_temp_dir, f"sentence_{i:03d}.wav")
            _exec_sentence_tts(sentence.strip(), source_wav, temp_audio_path, language)
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


def clone_voice(text: str, src_wav: str, out_wav: str, language: str = "en-us") -> str:
    return clone_voice_docker(text, src_wav, out_wav, language)


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
        "sadtalker",
        "python",
        "-u",
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
        "--verbose",
    ]
    print("[sadtalker] Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    st_log_lines = []
    if proc.stdout:
        for line in proc.stdout:
            print("[sadtalker]", line, end="", flush=True)
            st_log_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"SadTalker failed with exit code {ret}. See logs above.")

    log = "".join(st_log_lines)
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
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
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
    language: str,
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
            assistant_text = get_response(user_text, language)
        else:
            assistant_text = user_text

        progress(0.4, desc="✅ Text ready")
        yield (assistant_text, None, None, history)

        # 2) Clone the voice
        progress(0.5, desc="🎤 Cloning voice...")
        yield (assistant_text, None, None, history)

        # Clone voice to temp, then persist to results so it remains accessible
        cloned_wav_tmp = os.path.join(workdir, f"{uid}_clone.wav")
        clone_voice(assistant_text, src_wav, cloned_wav_tmp, language)

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


# ——— Gradio UI (simplified and robust) ———
if __name__ == "__main__":
    load_env()
    initialize_session()

    demo = gr.Blocks(title="AI Digital Twin V2 (Simple)")

    with demo:
        state = gr.State([])

        gr.Markdown("""
        # AI Digital Twin V2
        You can upload a voice sample, portrait image, and driving video — or leave any blank to use defaults from the local data/ folder.
        """)

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text (optional)", placeholder="Type text or leave blank to only clone voice from prompt + audio.", lines=3)
                use_ai = gr.Checkbox(label="Ask AI JimTwin.", value=True)
                language = gr.Dropdown(
                    label="Language",
                    choices=[
                        ("English (US)", "en-us"),
                        ("French", "fr-fr"),
                        ("German", "de"),
                        ("Japanese", "ja"),
                        ("Korean", "ko"),
                        ("Mandarin Chinese", "cmn"),
                    ],
                    value="en-us"
                )
                voice = gr.Audio(label="Voice Sample (optional; defaults to data/trainingaudio.wav)", type="filepath")
                image = gr.Image(label="Portrait Image (optional; defaults to data/screenshot.png)", type="filepath")
                video = gr.Video(label="Driving Video (optional; defaults to data/source_video.mp4)", format="mp4")
                generate = gr.Button("Generate")
                clear = gr.Button("Clear")
            with gr.Column():
                api_out = gr.Textbox(label="Response", interactive=False)
                audio_out = gr.Audio(label="Cloned Voice", type="filepath")
                video_out = gr.Video(label="Talking Avatar")
                history_md = gr.Markdown("No generations yet.")

        def _validate_inputs(txt, v_path, img_path, vid_path):
            # Resolve defaults from data/ if any input is missing
            project_root = os.path.abspath(os.path.dirname(__file__))
            data_dir = os.path.join(project_root, "data")
            default_voice = os.path.join(data_dir, "trainingaudio.wav")
            default_image = os.path.join(data_dir, "screenshot.png")
            default_video = os.path.join(data_dir, "source_video.mp4")

            v_path = v_path or default_voice
            img_path = img_path or default_image
            vid_path = vid_path or default_video

            # Validate existence of resolved paths
            missing = []
            if not (isinstance(v_path, str) and os.path.isfile(v_path)):
                missing.append(f"voice sample at {v_path}")
            if not (isinstance(img_path, str) and os.path.isfile(img_path)):
                missing.append(f"portrait image at {img_path}")
            if not (isinstance(vid_path, str) and os.path.isfile(vid_path)):
                missing.append(f"driving video at {vid_path}")
            if missing:
                raise gr.Error("Missing required file(s): " + ", ".join(missing))

            # Coerce to expected objects mimicking gradio File-like for pipeline
            class _F:
                def __init__(self, name: str):
                    self.name = name
            f1 = _F(v_path)
            f2 = _F(img_path)
            f3 = _F(vid_path)
            return txt or "", f1, f2, f3

        def _run(txt, v_path, img_path, vid_path, use_ai_flag, lang, hist, progress=gr.Progress()):
            # Stream steps using existing pipeline generator
            txt, vf, imf, vif = _validate_inputs(txt, v_path, img_path, vid_path)
            for api_resp, a_out, v_out, hist_out in pipeline(txt, vf, imf, vif, use_ai_flag, lang, hist, progress):
                yield api_resp, a_out, v_out, hist_out

        def _clear():
            return "", None, None

        def _refresh(history):
            if not history:
                return "No generations yet."
            lines = []
            for h in history[-5:]:
                lines.append(f"- {h['timestamp']} | {h['mode']} | {h['input'][:50]}{'...' if len(h['input'])>50 else ''}")
            return "\n".join(lines)

        clear.click(fn=_clear, outputs=[api_out, audio_out, video_out])

        generate.click(
            fn=_run,
            inputs=[text, voice, image, video, use_ai, language, state],
            outputs=[api_out, audio_out, video_out, state],
        ).then(fn=_refresh, inputs=[state], outputs=[history_md])

    demo.queue()
    demo.launch(server_name="127.0.0.1")