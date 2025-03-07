import os
import sys
import signal
import threading
import time
from RT_STT import start_listening
from TTS import TextToSpeech

def setup_environment():
    """Setup the environment variables and create necessary directories"""
    print("Initializing AI Digital Twin v2...")

    # Create temp directories if they don't exist
    temp_dirs = ["./temp_audio", "./models"]
    for directory in temp_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def handle_exit(signum, frame):
    """Handle clean exit on SIGINT"""
    print("\nShutting down AI Digital Twin...")
    # Clean up temporary files
    tts = TextToSpeech()
    tts.cleanup()
    print("Temporary files cleaned up")
    sys.exit(0)

def startup_message():
    """Display startup message and instructions"""
    print("\n" + "="*50)
    print("       AI DIGITAL TWIN - VERSION 2.0       ")
    print("="*50)
    print("\nSystem capabilities:")
    print("- Real-time speech recognition with wake word 'Hey Jim'")
    print("- Conversational AI responses")
    print("- Natural text-to-speech output")
    print("\nTo interact:")
    print("1. Say 'Hey Jim' or 'Hello Jim' to activate")
    print("2. Speak your question or command")
    print("3. Wait for Jim's response")
    print("\nPress Ctrl+C to exit")
    print("="*50 + "\n")

def main():
    """Main entry point for the AI Digital Twin system"""
    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Setup environment
    setup_environment()

    # Display welcome message
    startup_message()

    try:
        # Initialize TTS system as a test
        print("Initializing speech synthesis system...")
        tts = TextToSpeech()

        # Optional: Speak a welcome message
        welcome_message = "AI Digital Twin system is now online. I'm listening for your commands."
        tts_thread = threading.Thread(
            target=tts.speak,
            args=(welcome_message, None),
            daemon=True
        )
        tts_thread.start()

        # Start the voice recognition system
        start_listening()

        # Keep main thread alive (the start_listening function has its own loop)
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()

        # Try to clean up before exiting
        try:
            tts = TextToSpeech()
            tts.cleanup()
        except:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
