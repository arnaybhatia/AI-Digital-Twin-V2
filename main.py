import subprocess
import requests
import os
import sys
import threading
import signal
import time
import uuid
import json
from typing import Optional
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

class DockerBasedDigitalTwin:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.temp_audio_dir = os.path.join(self.data_dir, "temp_audio")
        self.temp_video_dir = os.path.join(self.data_dir, "temp_video")
        self.source_image = os.path.join(self.data_dir, "source_image.png")
        self.speaker_audio = os.path.join(self.data_dir, "speaker.wav")
        
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
    
    def transcribe_audio(self, input_audio):
        """Use Whisper container to transcribe audio"""
        try:
            # Generate a unique filename for the output
            output_filename = f"transcript_{uuid.uuid4()}.txt"
            output_path = os.path.join(self.data_dir, output_filename)
            
            # Make sure input audio is within the data directory
            if not input_audio.startswith(self.data_dir):
                rel_path = os.path.relpath(input_audio, os.path.dirname(self.data_dir))
                docker_input = f"/data/{rel_path}"
            else:
                rel_path = os.path.relpath(input_audio, self.data_dir)
                docker_input = f"/data/{rel_path}"
            
            print(f"Transcribing audio with Whisper...")
            cmd = [
                "docker", "exec", "testing-whisper-1", 
                "python", "whisper_transcribe.py", 
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
        """Use Zonos container to generate speech"""
        try:
            # Generate a unique filename for the output
            output_filename = f"zonos_output_{uuid.uuid4()}.wav"
            output_path = os.path.join(self.temp_audio_dir, output_filename)
            
            speaker = speaker_audio or self.speaker_audio
            
            # Get relative paths for docker
            speaker_rel_path = os.path.relpath(speaker, self.data_dir)
            docker_speaker = f"/data/{speaker_rel_path}"
            
            output_rel_dir = os.path.relpath(self.temp_audio_dir, self.data_dir)
            docker_output = f"/data/{output_rel_dir}/{output_filename}"
            
            print(f"Generating speech with Zonos: '{text}'")
            cmd = [
                "docker", "exec", "testing-zonos-1",
                "python", "zonos_generate.py",
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
        """Use KDTalker container to generate a talking avatar video"""
        try:
            # Generate a unique filename for the output
            output_filename = f"kdtalker_output_{uuid.uuid4()}.mp4"
            output_path = os.path.join(self.temp_video_dir, output_filename)
            
            image = image_file or self.source_image
            
            # Get relative paths for docker
            audio_rel_path = os.path.relpath(audio_file, self.data_dir)
            docker_audio = f"/data/{audio_rel_path}"
            
            image_rel_path = os.path.relpath(image, self.data_dir)
            docker_image = f"/data/{image_rel_path}"
            
            output_rel_dir = os.path.relpath(self.temp_video_dir, self.data_dir)
            docker_output = f"/data/{output_rel_dir}/{output_filename}"
            
            print(f"Generating avatar video with KDTalker")
            cmd = [
                "docker", "exec", "testing-kdtalker-1",
                "python", "inference.py",
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

def main():
    global digital_twin
    
    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Load environment variables
    load_env()
    
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
        
        # Run interactive chat
        interactive_chat()
        
        # Clean up before exiting
        digital_twin.cleanup()
        
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()