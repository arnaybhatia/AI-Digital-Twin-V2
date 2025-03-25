import os
import sys
import signal
import threading
import time
from RT_STT import start_listening
from TTS import TextToSpeech
from avatar_generator import AvatarGenerator
from threading import Event

# Global event for interrupting processes
interrupt_event = Event()
# Global TTS and Avatar Generator instances
tts = None
avatar_gen = None

def setup_environment():
    """Setup the environment variables and create necessary directories"""
    print("Initializing AI Digital Twin v2...")
    # Create temp directories if they don't exist
    temp_dirs = ["./temp_audio", "./temp_video", "./models", "./data"]
    for directory in temp_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def handle_exit(signum, frame):
    """Handle clean exit on SIGINT"""
    print("\nShutting down AI Digital Twin...")
    # Signal other threads to stop
    interrupt_event.set()
    # Clean up temporary files
    if tts:
        tts.cleanup()
    if avatar_gen:
        avatar_gen.cleanup()
    print("Temporary files cleaned up")
    sys.exit(0)

def startup_message():
    print("\n" + "="*50)
    print("       AI DIGITAL TWIN - VERSION 2.0       ")
    print("="*50)
    print("\nSystem capabilities:")
    print("- Real-time speech recognition with wake word 'Hey Jim'")
    print("- Conversational AI responses")
    print("- Natural text-to-speech output using Zonos")
    print("- Talking avatar generation using KDTalker")
    print("\nTo interact:")
    print("1. Say 'Hey Jim' or 'Hello Jim' to activate")
    print("2. Speak your question or command")
    print("3. Wait for Jim's response and watch the avatar")
    print("\nPress Ctrl+C to exit")
    print("="*50 + "\n")

def process_tts_with_avatar(text):
    """Process TTS and generate avatar video"""
    global tts, avatar_gen
    
    # Generate a unique filename for this audio output
    output_filename = f"zonos_output_{time.time()}.wav"
    output_path = os.path.join("./temp_audio", output_filename)
    
    try:
        # First, generate the audio using Zonos TTS
        print(f"Generating speech for: '{text}'")
        
        # Use the TTS system to generate audio
        tts.speak(text, interrupt_event)
        
        # After TTS is generated, create the avatar video
        if not interrupt_event.is_set() and os.path.exists(output_path):
            video_path = avatar_gen.generate_avatar_video(output_path)
            if video_path:
                print(f"Avatar video generated at: {video_path}")
            else:
                print("Failed to generate avatar video")
    except Exception as e:
        print(f"Error in process_tts_with_avatar: {e}")

def main():
    """Main entry point for the AI Digital Twin system"""
    global tts, avatar_gen, interrupt_event
    
    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Setup environment
    setup_environment()
    
    # Display welcome message
    startup_message()
    
    try:
        # Initialize TTS and Avatar Generator
        print("Initializing speech synthesis and avatar generation systems...")
        tts = TextToSpeech(sample_audio="./data/sample_audio.wav")
        avatar_gen = AvatarGenerator(source_image="./data/source_image.png")
        
        # Optional: Speak a welcome message and generate avatar
        welcome_message = "AI Digital Twin system is now online. I'm listening for your commands."
        tts_thread = threading.Thread(
            target=process_tts_with_avatar,
            args=(welcome_message,),
            daemon=True
        )
        tts_thread.start()
        
        # Start the voice recognition system
        start_listening(interrupt_event=interrupt_event, process_tts_with_avatar=process_tts_with_avatar)
        
        # Keep main thread alive (the start_listening function has its own loop)
        while not interrupt_event.is_set():
            time.sleep(1)
            
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up before exiting
        try:
            if tts:
                tts.cleanup()
            if avatar_gen:
                avatar_gen.cleanup()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
