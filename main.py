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
import traceback
import subprocess
import speech_recognition as sr
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from whisper_assistant import VoiceAssistant

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
    def __init__(self, mic_device_index=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.temp_audio_dir = os.path.join(self.data_dir, "temp_audio")
        self.temp_video_dir = os.path.join(self.data_dir, "temp_video")
        self.source_image = os.path.join(self.data_dir, "screenshot.png")
        self.speaker_audio = os.path.join(self.data_dir, "speaker.wav")
        
        # Audio recording settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 16kHz sampling rate
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.recording = False
        self.wake_words = ["hey jim", "hello jim", "hi jim"]
        self.mic_device_index = mic_device_index
        
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
        
        # Create voice assistant for real-time transcription
        self.voice_assistant = None







    def start_listening(self):
        """Start listening for audio input using Whisper"""
        print("Initializing Whisper Voice Assistant...")
        
        # Initialize the voice assistant if not already done
        if not self.voice_assistant:
            self.voice_assistant = VoiceAssistant(
                api_response_func=get_response,
                device_index=self.mic_device_index
            )
        
        # Start listening for wake words
        self.voice_assistant.start_listening()
        
        print("Listening for wake word... Say 'Hey Jim' to activate")
    
    def stop_listening(self):
        """Stop the voice assistant"""
        if self.voice_assistant:
            self.voice_assistant.stop_listening()
            
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            print("Cleaning up temporary audio/video files...")
            for directory in [self.temp_audio_dir, self.temp_video_dir]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                # print(f"Removed temp file: {file_path}") # Optional: uncomment for verbose cleanup
                        except Exception as e:
                            print(f"Warning: Could not remove temporary file {file_path}: {e}")
            print("Cleanup complete.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def handle_exit(signum, frame):
    """Handle clean exit on SIGINT"""
    print("\nShutting down AI Digital Twin...")
    # Signal other threads to stop
    interrupt_event.set()
    
    # Clean up temporary files if available
    if 'digital_twin' in globals():
        digital_twin.stop_listening()  # Stop the voice assistant
        digital_twin.cleanup()
        
    print("Temporary files cleaned up")
    sys.exit(0)

def startup_message():
    print("\n" + "="*50)
    print("       AI DIGITAL TWIN - DOCKER TEST VERSION       ")
    print("="*50)
    print("\nSystem capabilities:")
    print("- Interactive conversation with API")
    print("\nType 'exit' or press Ctrl+C to quit")
    print("="*50 + "\n")



def live_voice_chat():
    """Run a live voice chat session with wake word detection"""
    global digital_twin
    success = False  # Track if the function completes successfully
    
    try:
        # Initialize session with API
        if not initialize_session():
            print("Failed to initialize API session")
            return False
        
        # Start listening for wake word
        digital_twin.start_listening()
        
        print("AI Digital Twin is now listening. Say 'Hey Jim' to activate.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep the main thread alive until interrupted
            while not interrupt_event.is_set():
                time.sleep(0.1)
            success = True  # If we get here without exceptions, mark as success
        except KeyboardInterrupt:
            print("\nExiting voice chat...")
            success = True  # Ctrl+C is a normal exit condition
        finally:
            # Stop the voice assistant
            digital_twin.stop_listening()
            
    except Exception as e:
        print(f"Error in voice chat: {e}")
        traceback.print_exc()
        success = False
        
    return success  # Return whether the function completed successfully

def main():
    global digital_twin
    exit_code = 0

    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='AI Digital Twin')
        parser.add_argument('--input', type=str, help='Text input to query the AI directly')
        args = parser.parse_args()

        # Load environment variables
        try:
            load_env()
        except FileNotFoundError: # More specific exception
            print("Note: No .env file found. Continuing without API key.")
        except Exception as env_e: # Catch other potential env loading errors
            print(f"Error loading .env file: {env_e}")
            # Decide if you want to exit or continue without API key
            # sys.exit(1) # Uncomment to exit if .env loading fails critically

        # If input is provided, use simple query mode
        if args.input:
            # Get response from API
            response = simple_query(args.input)
            if response:
                print("\n" + "="*50)
                print("RESPONSE:")
                print(response)
                print("="*50)
            else:
                print("Error: Failed to get a response from the API")
                exit_code = 1
            # No return here, let it fall through to finally for cleanup if needed

        # Otherwise, proceed with the full digital twin experience
        else:
            startup_message()

            # Use default microphone (skip selection)
            mic_index = None  # None means use default microphone

            # Initialize digital twin system with default microphone
            # Ensure digital_twin is initialized *before* the try block that uses it in finally
            digital_twin = DockerBasedDigitalTwin(mic_device_index=mic_index)

            # Run voice-activated chat with keep-alive loop
            if not live_voice_chat(): # live_voice_chat now handles its own errors/cleanup via finally
                exit_code = 1

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down gracefully...")
        # The handle_exit signal handler should manage the shutdown
        # We might not need explicit cleanup here if handle_exit does it all
    except Exception as e:
        print(f"Error in main program scope: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # This block ensures cleanup happens regardless of how the try block exits
        # Check if digital_twin was successfully initialized before trying cleanup
        if 'digital_twin' in locals() and digital_twin is not None:
            try:
                print("Performing final cleanup in main...")
                # Ensure assistant is stopped, even if live_voice_chat had issues
                digital_twin.stop_listening()
                digital_twin.cleanup() # Perform other cleanup
            except Exception as cleanup_e:
                print(f"Error during final cleanup in main: {cleanup_e}")

    print(f"Exiting with code {exit_code}")
    sys.exit(exit_code) # Exit with the determined code.

if __name__ == "__main__":
    main()