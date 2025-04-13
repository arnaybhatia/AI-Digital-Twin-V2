import subprocess
import requests
import os
import sys
import threading
import signal
import time
import uuid
import json
import queue
import wave
import pyaudio
import numpy as np
import argparse
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Global event for interrupting processes
interrupt_event = threading.Event()

# API connection variables
API_KEY = None
BASE_URL = "https://api.tmpt.app/v1"
CLIENT_TOKEN = None
THREAD_ID = None

def load_env():
    """Load environment variables from .env file in testing directory"""
    global API_KEY
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        API_KEY = os.getenv("TMPT_API_KEY")
    
    if not API_KEY:
        print("Error: TMPT_API_KEY not found in testing/.env file")
        print("Please create a .env file in the testing directory with your API key")
        sys.exit(1)

def make_request(method, endpoint, **kwargs):
    """Helper function to make API requests with error handling"""
    url = BASE_URL + endpoint
    headers = kwargs.get('headers', {})
    headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    })
    kwargs['headers'] = headers
    
    try:
        response = requests.request(
            method=method,
            url=url,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making {method} request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

def initialize_session():
    """Initialize a session by creating a client token and a thread"""
    global CLIENT_TOKEN, THREAD_ID
    
    # Get client token
    client_token_data = make_request(
        "POST",
        "/client_token",
        json={"external_user_id": f"user_{int(time.time())}", "is_reviewable": False}
    )
    
    if not client_token_data:
        print("Failed to get client token")
        return False
    
    CLIENT_TOKEN = client_token_data["client_token_id"]
    
    # Create thread
    thread_data = make_request(
        "POST",
        "/threads",
        json={"client_token_id": CLIENT_TOKEN}
    )
    
    if not thread_data:
        print("Failed to create thread")
        return False
    
    THREAD_ID = thread_data["id"]
    return True

def get_response(user_input: str, interrupt_event: Optional[threading.Event] = None) -> str:
    """Send a message to the API and get a response"""
    global CLIENT_TOKEN, THREAD_ID
    
    try:
        # Initialize session if not already done
        if CLIENT_TOKEN is None or THREAD_ID is None:
            if not initialize_session():
                return "I encountered an error while initializing the conversation. Please try again."
        
        # Post the message
        message_data = make_request(
            "POST",
            f"/threads/{THREAD_ID}/messages",
            json={
                "client_token_id": CLIENT_TOKEN,
                "text": user_input
            }
        )
        
        if not message_data:
            # If message posting fails, try to reinitialize the session
            print("Message post failed, trying to reinitialize session...")
            if initialize_session():
                # Try posting again with the new session
                message_data = make_request(
                    "POST",
                    f"/threads/{THREAD_ID}/messages",
                    json={
                        "client_token_id": CLIENT_TOKEN,
                        "text": user_input
                    }
                )
                if not message_data:
                    return "I couldn't process your message after multiple attempts. Please try again."
            else:
                return "I couldn't process your message. Please try again."
        
        message_id = message_data["id"]
        
        # Wait for reply
        max_retries = 5  # Increased from 3 to 5
        for attempt in range(max_retries):
            try:
                if interrupt_event and interrupt_event.is_set():
                    return "Response interrupted."
                
                reply_data = make_request(
                    "GET",
                    f"/threads/{THREAD_ID}/reply/{message_id}",
                    params={"client_token_id": CLIENT_TOKEN, "timeout": 15}
                )
                
                if reply_data and "text" in reply_data:
                    return reply_data["text"]
                else:
                    if attempt < max_retries - 1:
                        print(f"Reply attempt {attempt + 1} didn't return text, retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                        
                    # As a fallback, try to get all messages
                    print("Using fallback: fetching all messages...")
                    messages = make_request(
                        "GET",
                        f"/threads/{THREAD_ID}/messages",
                        params={"client_token_id": CLIENT_TOKEN}
                    )
                    
                    if messages:
                        # Find the most recent message from the agent
                        for msg in reversed(messages):
                            if msg['speaker'] == 'agent':
                                return msg['text']
                    
                    # If we can't get a proper response, provide a generic one
                    return "I'm sorry, I'm having trouble generating a response right now. Could you please try again?"
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[DEBUG] Final API attempt failed: {str(e)}")
                    return "I encountered an error while processing your request. Please try again."
                print(f"[DEBUG] API attempt {attempt + 1} failed, retrying... Error: {e}")
                time.sleep(2)  # Increased from 1s to 2s
                
    except Exception as e:
        print(f"[DEBUG] Error in get_response: {str(e)}")
        return "I encountered an error while processing your request. Please try again."
    
    return "I encountered an unexpected error. Please try again."

def simple_query(query_text):
    """Simple function to query the API with text and get a response"""
    # Initialize session if needed
    if not initialize_session():
        print("Failed to initialize API session")
        return None
    
    print(f"Sending query: '{query_text}'")
    
    # Get response from API
    response_text = get_response(query_text, interrupt_event)
    
    return response_text

class DockerBasedDigitalTwin:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.temp_audio_dir = os.path.join(self.data_dir, "temp_audio")
        self.temp_video_dir = os.path.join(self.data_dir, "temp_video")
        self.source_image = os.path.join(self.data_dir, "source_image.png")
        self.speaker_audio = os.path.join(self.data_dir, "speaker.wav")
        
        # Audio recording settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 16kHz sampling rate
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.recording = False
        self.audio_queue = queue.Queue()
        self.wake_words = ["hey jim", "hello jim", "hi jim"]
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        os.makedirs(self.temp_video_dir, exist_ok=True)
        
        print(f"Data directory: {self.data_dir}")
        print(f"Using source image: {self.source_image}")
        print(f"Using speaker audio: {self.speaker_audio}")
        
        # Check if required files exist
        if not os.path.exists(self.source_image):
            print(f"WARNING: Source image not found at {self.source_image}")
        if not os.path.exists(self.speaker_audio):
            print(f"WARNING: Speaker audio not found at {self.speaker_audio}")
    
    def start_listening(self):
        """Start listening for audio input and detect wake words"""
        self.recording = True
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start recording in a separate thread
        self.listen_thread = threading.Thread(target=self._listen_for_wake_word)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        print("🎤 Listening for wake word... Say 'Hey Jim' to activate")
        
    def _listen_for_wake_word(self):
        """Listen for wake word in the background"""
        # Open stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Buffer for storing audio
        audio_buffer = []
        # How many seconds of audio to keep for wake word detection
        buffer_seconds = 2
        max_buffer_chunks = int(self.rate / self.chunk * buffer_seconds)
        
        print("Listening for 'Hey Jim'...")
        
        try:
            while self.recording:
                # Read chunk of audio
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_buffer.append(data)
                
                # Keep the buffer at a fixed size
                if len(audio_buffer) > max_buffer_chunks:
                    audio_buffer.pop(0)
                
                # Every half second, check for wake word
                if len(audio_buffer) % (max_buffer_chunks // 4) == 0:
                    # Save buffer to a temporary file
                    temp_filename = os.path.join(self.temp_audio_dir, f"wake_word_check_{uuid.uuid4()}.wav")
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.audio.get_sample_size(self.format))
                        wf.setframerate(self.rate)
                        wf.writeframes(b''.join(audio_buffer))
                    
                    # Check for wake word using Whisper
                    transcript = self.transcribe_audio(temp_filename)
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
                    
                    if transcript:
                        transcript = transcript.lower().strip()
                        print(f"Heard: {transcript}")
                        
                        # Check if wake word is in the transcript
                        if any(wake_word in transcript for wake_word in self.wake_words):
                            print("🔔 Wake word detected!")
                            
                            # Start recording the actual command
                            self._record_command(stream)
                            break
        
        except Exception as e:
            print(f"Error in listen thread: {e}")
        finally:
            stream.stop_stream()
            stream.close()
    
    def _record_command(self, stream=None):
        """Record audio command after wake word is detected"""
        print("🎤 Recording your command... Speak now")
        
        # If no stream is provided, start a new one
        if stream is None:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
        # Record for a few seconds
        command_frames = []
        max_command_time = 5  # seconds
        for _ in range(0, int(self.rate / self.chunk * max_command_time)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            command_frames.append(data)
            
        print("✅ Finished recording command")
        
        # Save command to file
        command_filename = os.path.join(self.temp_audio_dir, f"command_{uuid.uuid4()}.wav")
        with wave.open(command_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(command_frames))
            
        print("💾 Command saved to file")
        
        # Transcribe the command
        transcript = self.transcribe_audio(command_filename)
        if transcript:
            print(f"🔊 You said: {transcript}")
            # Process the transcript
            self.audio_queue.put({
                "transcript": transcript,
                "audio_file": command_filename
            })
        else:
            print("❌ Failed to transcribe command")
            try:
                os.remove(command_filename)
            except:
                pass
                
    def transcribe_audio(self, input_audio):
        """Use Whisper container to transcribe audio on demand"""
        try:
            # Generate a unique filename for the output
            output_filename = f"transcript_{uuid.uuid4()}.txt"
            output_path = os.path.join(self.data_dir, output_filename)
            
            # Get absolute paths for docker volume mounting
            abs_data_dir = os.path.abspath(self.data_dir)
            
            # Make sure input audio path is relative to the data directory for the container
            if not input_audio.startswith(self.data_dir):
                rel_path = os.path.relpath(input_audio, os.path.dirname(self.data_dir))
                docker_input = f"/data/{rel_path}"
            else:
                rel_path = os.path.relpath(input_audio, self.data_dir)
                docker_input = f"/data/{rel_path}"
            
            print(f"Transcribing audio with Whisper...")
            cmd = [
                "docker", "run", "--rm", 
                "--gpus", "all",
                "-v", f"{abs_data_dir}:/data",
                "ai-digital-twin-v2-whisper", 
                "whisper_transcribe.py", 
                "--file", docker_input, 
                "--output", f"/data/{output_filename}"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Read the transcript
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    transcript = f.read().strip()
                print(f"Transcript: {transcript}")
                # Clean up
                try:
                    os.remove(output_path)
                except:
                    pass
                return transcript
            else:
                print("Error: Transcript file not found")
                return None
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
            
    def generate_speech(self, text, speaker_audio=None):
        """Use Zonos container to generate speech on demand"""
        try:
            # Generate a unique filename for the output
            output_filename = f"zonos_output_{uuid.uuid4()}.wav"
            output_path = os.path.join(self.temp_audio_dir, output_filename)
            
            speaker = speaker_audio or self.speaker_audio
            
            # Get absolute path for docker volume mounting
            abs_data_dir = os.path.abspath(self.data_dir)
            
            # Get relative paths for docker
            speaker_rel_path = os.path.relpath(speaker, self.data_dir)
            docker_speaker = f"/data/{speaker_rel_path}"
            
            output_rel_dir = os.path.relpath(self.temp_audio_dir, self.data_dir)
            docker_output = f"/data/{output_rel_dir}/{output_filename}"
            
            print(f"Generating speech with Zonos: '{text}'")
            cmd = [
                "docker", "run", "--rm",
                "--gpus", "all",
                "-v", f"{abs_data_dir}:/data",
                "ai-digital-twin-v2-zonos",
                "zonos_generate.py",
                "--text", text,
                "--output", docker_output,
                "--speaker_audio", docker_speaker
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_path):
                return output_path
            else:
                print(f"Error: Generated audio file not found at {output_path}")
                return None
                
        except Exception as e:
            print(f"Speech generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def generate_avatar_video(self, audio_file, image_file=None):
        """Use KDTalker container to generate a talking avatar video on demand"""
        try:
            # Generate a unique filename for the output
            output_filename = f"kdtalker_output_{uuid.uuid4()}.mp4"
            output_path = os.path.join(self.temp_video_dir, output_filename)
            
            image = image_file or self.source_image
            
            # Get absolute path for docker volume mounting
            abs_data_dir = os.path.abspath(self.data_dir)
            
            # Get relative paths for docker
            audio_rel_path = os.path.relpath(audio_file, self.data_dir)
            docker_audio = f"/data/{audio_rel_path}"
            
            image_rel_path = os.path.relpath(image, self.data_dir)
            docker_image = f"/data/{image_rel_path}"
            
            output_rel_dir = os.path.relpath(self.temp_video_dir, self.data_dir)
            docker_output = f"/data/{output_rel_dir}/{output_filename}"
            
            print(f"Generating avatar video with KDTalker")
            cmd = [
                "docker", "run", "--rm",
                "--gpus", "all",
                "-v", f"{abs_data_dir}:/data",
                "ai-digital-twin-v2-kdtalker",
                "inference.py",
                "--source_image", docker_image,
                "--driven_audio", docker_audio,
                "--output", docker_output
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_path):
                return output_path
            else:
                print(f"Error: Generated video file not found at {output_path}")
                return None
                
        except Exception as e:
            print(f"Avatar generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_input_to_video(self, text_input=None, audio_input=None):
        """Process user input (text or audio) and generate a response video"""
        if not text_input and not audio_input:
            print("Error: Either text or audio input is required")
            return None
            
        try:
            # Step 1: Get text from audio if not provided directly
            if not text_input and audio_input:
                text_input = self.transcribe_audio(audio_input)
                if not text_input:
                    return None
                    
            # Step 2: Get API response
            print(f"Getting API response for: '{text_input}'")
            api_response = get_response(text_input, interrupt_event)
            print(f"API response: '{api_response}'")
            
            # Step 3: Generate speech from response
            audio_path = self.generate_speech(api_response)
            if not audio_path:
                return None
                
            # Step 4: Generate avatar video
            video_path = self.generate_avatar_video(audio_path)
            if not video_path:
                return None
                
            return {
                "text": api_response,
                "audio": audio_path,
                "video": video_path
            }
            
        except Exception as e:
            print(f"Error processing input to video: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_audio_dir):
                if file.startswith("zonos_output_"):
                    os.remove(os.path.join(self.temp_audio_dir, file))
            
            for file in os.listdir(self.temp_video_dir):
                if file.startswith("kdtalker_output_"):
                    os.remove(os.path.join(self.temp_video_dir, file))
                    
            for file in os.listdir(self.data_dir):
                if file.startswith("transcript_"):
                    os.remove(os.path.join(self.data_dir, file))
        except Exception as e:
            print(f"Cleanup error: {e}")
            
    def check_containers_running(self):
        """Check if all required containers are running"""
        required_containers = [
            "testing-whisper-1",
            "testing-zonos-1",
            "testing-kdtalker-1"
        ]
        
        for container in required_containers:
            cmd = ["docker", "ps", "--filter", f"name={container}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if container not in result.stdout:
                print(f"Error: Container {container} is not running")
                print("Please start all containers with 'docker compose up -d' before running this script")
                return False
                
        return True

    def check_and_restart_containers(self):
        """Check containers and restart them if needed"""
        required_containers = [
            "testing-whisper-1",
            "testing-zonos-1",
            "testing-kdtalker-1"
        ]
        
        containers_to_restart = []
        
        # Check the status of each container
        for container in required_containers:
            cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0 or "running" not in result.stdout.strip():
                print(f"Container {container} is not running properly. Status: {result.stdout.strip() if result.returncode == 0 else 'not found'}")
                containers_to_restart.append(container)
        
        # If any containers need to be restarted
        if containers_to_restart:
            print(f"Restarting containers: {', '.join(containers_to_restart)}")
            
            # Try to restart each container individually to avoid losing all containers if one fails
            for container in containers_to_restart:
                print(f"Restarting {container}...")
                restart_cmd = ["docker", "restart", container]
                restart_result = subprocess.run(restart_cmd, capture_output=True, text=True)
                
                if restart_result.returncode != 0:
                    print(f"Failed to restart {container}, trying to recreate with docker compose...")
                    # If restart fails, try to recreate the container using docker compose
                    compose_cmd = ["docker", "compose", "up", "-d", container.replace("testing-", "").replace("-1", "")]
                    subprocess.run(compose_cmd, cwd=self.base_dir)
            
            # Wait for containers to start up
            print("Waiting for containers to initialize...")
            time.sleep(10)
            
            # Verify all containers are now running
            all_running = True
            for container in required_containers:
                cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or "running" not in result.stdout.strip():
                    all_running = False
                    print(f"Container {container} failed to start. You may need to troubleshoot it manually.")
            
            return all_running
        
        return True

def handle_exit(signum, frame):
    """Handle clean exit on SIGINT"""
    print("\nShutting down AI Digital Twin...")
    # Signal other threads to stop
    interrupt_event.set()
    
    # Clean up temporary files if available
    if 'digital_twin' in globals():
        digital_twin.cleanup()
        
    print("Temporary files cleaned up")
    sys.exit(0)

def startup_message():
    print("\n" + "="*50)
    print("       AI DIGITAL TWIN - DOCKER TEST VERSION       ")
    print("="*50)
    print("\nSystem capabilities:")
    print("- Interactive conversation with API")
    print("- Text-to-speech using Zonos Docker container")
    print("- Talking avatar generation using KDTalker Docker container")
    print("\nType 'exit' or press Ctrl+C to quit")
    print("="*50 + "\n")

def interactive_chat():
    """Run an interactive chat session"""
    global digital_twin
    
    # Initialize session with API
    initialize_session()
    
    while not interrupt_event.is_set():
        try:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Process input and generate response video
            result = digital_twin.process_input_to_video(text_input=user_input)
            
            if result:
                print(f"\nAI: {result['text']}")
                print(f"Audio generated: {result['audio']}")
                print(f"Video generated: {result['video']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in chat: {e}")

def live_voice_chat():
    """Run a live voice chat session with wake word detection"""
    global digital_twin
    
    # Initialize session with API
    initialize_session()
    
    # Start listening for wake word
    digital_twin.start_listening()
    
    print("📢 AI Digital Twin is now listening. Say 'Hey Jim' to activate.")
    print("Press Ctrl+C to exit.")
    
    try:
        while not interrupt_event.is_set():
            try:
                # Check if there's a command in the queue with a short timeout
                command_data = digital_twin.audio_queue.get(timeout=0.5)
                transcript = command_data.get("transcript", "")
                audio_file = command_data.get("audio_file", "")
                
                if transcript:
                    print(f"📝 Processing command: \"{transcript}\"")
                    
                    # Process the command and get a video response
                    result = digital_twin.process_input_to_video(text_input=transcript)
                    
                    if result:
                        print(f"\nAI: {result['text']}")
                        print(f"Audio generated: {result['audio']}")
                        print(f"Video generated: {result['video']}")
                        
                    # After processing, start listening for wake word again
                    digital_twin.start_listening()
                    
            except queue.Empty:
                # No command in queue, continue listening
                pass
                
    except KeyboardInterrupt:
        print("\nExiting voice chat...")
    finally:
        # Stop recording
        digital_twin.recording = False

def generate_speech_direct(text, speaker_audio_path):
    """Generate speech directly using docker run, without checking container state"""
    try:
        # Generate a unique filename for the output
        output_filename = f"zonos_output_{uuid.uuid4()}.wav"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        temp_audio_dir = os.path.join(data_dir, "temp_audio")
        output_path = os.path.join(temp_audio_dir, output_filename)
        
        # Make sure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        # Get absolute path for docker volume mounting
        abs_data_dir = os.path.abspath(data_dir)
        
        # Get relative paths for docker
        speaker_rel_path = os.path.relpath(speaker_audio_path, data_dir)
        docker_speaker = f"/data/{speaker_rel_path}"
        
        output_rel_dir = os.path.relpath(temp_audio_dir, data_dir)
        docker_output = f"/data/{output_rel_dir}/{output_filename}"
        
        print(f"Generating speech with Zonos: '{text}'")
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{abs_data_dir}:/data",
            "ai-digital-twin-v2-zonos",
            "zonos_generate.py",
            "--text", text,
            "--output", docker_output,
            "--speaker_audio", docker_speaker
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        if os.path.exists(output_path):
            return output_path
        else:
            print(f"Error: Generated audio file not found at {output_path}")
            return None
            
    except Exception as e:
        print(f"Speech generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    global digital_twin
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='AI Digital Twin')
    parser.add_argument('--input', type=str, help='Text input to query the AI directly')
    args = parser.parse_args()
    
    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Load environment variables
    try:
        load_env()
    except:
        print("Note: No .env file found. Continuing without API key.")
    
    # If input is provided, use simple query mode
    if args.input:
        # Get response from API
        response = simple_query(args.input)
        if response:
            print("\n" + "="*50)
            print("RESPONSE:")
            print(response)
            print("="*50)
            
            # Generate speech directly from the response using docker run
            print("\nGenerating speech with Zonos...")
            speaker_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "speaker.wav")
            
            # Check if speaker.wav exists, create a placeholder if it doesn't
            if not os.path.exists(speaker_audio_path):
                print(f"WARNING: Speaker audio not found at {speaker_audio_path}")
                print("Creating empty speaker audio file...")
                os.makedirs(os.path.dirname(speaker_audio_path), exist_ok=True)
                with open(speaker_audio_path, 'wb') as f:
                    # Write an empty WAV file header as placeholder
                    f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            
            try:
                audio_path = generate_speech_direct(response, speaker_audio_path)
                if audio_path:
                    print(f"Speech generated and saved to: {audio_path}")
                    print("You can play this file with any audio player.")
                else:
                    print("Failed to generate speech. Check docker logs for details.")
            except Exception as e:
                print(f"Error generating speech: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Error: Failed to get a response from the API")
        return
    
    # Otherwise, proceed with the full digital twin experience
    startup_message()
    
    try:
        # Initialize digital twin system
        digital_twin = DockerBasedDigitalTwin()
        
        # Check if containers are running
        if not digital_twin.check_containers_running():
            print("\nStarting required containers...")
            subprocess.run(["docker", "compose", "up", "-d"], cwd=os.path.dirname(os.path.abspath(__file__)))
            # Give containers a moment to start up
            print("Waiting for containers to start up...")
            time.sleep(5)
        
        # Check and restart containers if needed
        if not digital_twin.check_and_restart_containers():
            print("Failed to ensure all containers are running. Exiting...")
            sys.exit(1)
        
        # Run voice-activated chat
        live_voice_chat()
        
        # Clean up before exiting
        digital_twin.cleanup()
        
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()