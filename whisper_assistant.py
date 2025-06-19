import speech_recognition as sr
import whisper
import numpy as np
import torch
import wave
import os
import threading
from queue import Queue, Empty
import warnings
import time
import re

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class VoiceAssistant:
    def __init__(self, api_response_func, process_tts_with_avatar=None, device_index=None):
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device for Whisper")
        self.model = whisper.load_model("tiny", device=device)

        # Function to get API responses
        self.get_response = api_response_func

        # Function to process TTS and avatar together
        self.process_tts_with_avatar = process_tts_with_avatar

        # Queues and events for audio processing
        self.audio_queue = Queue()
        self.processing_queue = Queue()
        self.is_speaking = threading.Event()
        self.interrupt_event = threading.Event()

        # Recording state and resources
        self.recording = False
        self.recognizer = None # Initialize recognizer instance variable
        self.microphone = None # Initialize microphone instance variable
        self.microphone_source = None # Initialize microphone source variable
        self.listen_thread = None # To keep track of threads
        self.process_thread = None # To keep track of threads
        self.wake_words = ["hey jim", "hello jim", "hi jim"]
        self.device_index = device_index # Store the device index

    def start_listening(self):
        """Start listening for wake words using Whisper for transcription"""
        if self.recording:
            # Already listening
            return

        self.recording = True
        self.interrupt_event.clear()

        print("Starting microphone...")
        try:
            # Initialize recognizer and microphone instances
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 2.0
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5

            self.microphone = sr.Microphone(device_index=self.device_index, sample_rate=16000)

            # Manually enter the microphone context
            self.microphone_source = self.microphone.__enter__()
            print("Microphone stream opened.")

            print("Calibrating microphone for ambient noise... Please wait...")
            self.recognizer.adjust_for_ambient_noise(self.microphone_source, duration=5)
            print(f"Initial energy threshold set to: {self.recognizer.energy_threshold}")

            # Start the listener thread
            self.listen_thread = threading.Thread(
                target=self.listen_continuously,
                args=(self.microphone_source, self.recognizer), # Pass instance variables
                daemon=True
            )
            self.listen_thread.start()

            # Start the processing thread
            self.process_thread = threading.Thread(
                target=self.process_audio_continuously,
                daemon=True
            )
            self.process_thread.start()

            print("\nReady to listen! Say 'Hey Jim' or 'Hello Jim' to start...")

        except Exception as e:
             print(f"Error initializing microphone or starting threads: {e}")
             print("Please ensure you have a microphone connected and PyAudio is installed correctly.")
             self.recording = False # Stop if setup fails
             # Clean up microphone if it was partially opened
             if self.microphone:
                 try:
                     self.microphone.__exit__(None, None, None)
                 except Exception as exit_e:
                     print(f"Error closing microphone during setup failure: {exit_e}")
             self.microphone = None
             self.microphone_source = None
             self.recognizer = None


    def listen_continuously(self, source, recognizer):
        """Thread function to continuously listen for audio"""
        while self.recording and not self.interrupt_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=None)
                self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Listening error: {e}")
                if not self.recording:
                    break
    
    def process_audio_continuously(self):
        """Thread function to process audio and check for triggers"""
        while self.recording and not self.interrupt_event.is_set():
            try:
                try:
                    audio = self.audio_queue.get(timeout=1)
                except Empty:
                    continue
                
                # print("Processing audio chunk...") # Reduce verbosity
                self.process_single_audio(audio)
            
            except Exception as e:
                if not isinstance(e, Empty):
                    print(f"Processing error: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    def process_single_audio(self, audio):
        try:
            # print("Processing audio chunk...") # Reduce verbosity
            audio_data = audio.get_wav_data()
            data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data)//2, offset=0)
            float_data = data_s16.astype(np.float32, order='C') / 32768.0
            
            # print("Running Whisper transcription...") # Reduce verbosity
            result = self.model.transcribe(float_data, fp16=torch.cuda.is_available())
            
            if not result or "text" not in result:
                # print("No transcription result") # Reduce verbosity
                return
            
            text = result["text"].strip().lower()
            if text: # Only print if there is text
                print(f"Transcribed: {text}")
            
            trigger_pattern = r"(?:hey|hello|hi)\s*jim"
            if re.search(trigger_pattern, text):
                print("Trigger phrase detected!")
                if self.is_speaking.is_set():
                    print("Interrupting current speech...")
                    self.interrupt_event.set()
                    self.is_speaking.clear()
                    time.sleep(0.5)  # Give time for cleanup
                    self.interrupt_event.clear()
                
                query = re.sub(trigger_pattern, "", text).strip()
                if query:
                    self.handle_query(query)
                else:
                    print("No query after trigger phrase")
        
        except Exception as e:
            print(f"Error in process_single_audio: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_query(self, query):
        """Handle a single query in a separate thread"""
        try:
            print("\nJim is thinking...")
            self.is_speaking.set()
            
            def process_and_speak():
                try:
                    response = self.get_response(query, self.interrupt_event)
                    if not self.interrupt_event.is_set():
                        print("Jim:", response)
                        # Use the combined TTS and avatar generator if available
                        if self.process_tts_with_avatar:
                            self.process_tts_with_avatar(response)
                        else:
                            print("TTS/Avatar function not provided.")
                finally:
                    self.is_speaking.clear()
                    print("\nReady to listen again!")
            
            # Start processing in a new thread
            threading.Thread(target=process_and_speak, daemon=True).start()
        
        except Exception as e:
            print(f"Error handling query: {e}")
            self.is_speaking.clear()
    
    def stop_listening(self):
        """Stop the listening process and clean up resources"""
        print("Stopping voice assistant...")
        self.recording = False # Signal threads to stop looping
        self.interrupt_event.set() # Signal any blocking calls in threads

        # Wait briefly for threads to finish current task (optional but good practice)
        # Note: Joining daemon threads isn't strictly necessary for exit,
        # but helps ensure resources are released cleanly before closing mic.
        if self.listen_thread and self.listen_thread.is_alive():
             self.listen_thread.join(timeout=1.0)
        if self.process_thread and self.process_thread.is_alive():
             self.process_thread.join(timeout=1.0)

        # Clean up microphone resource
        if self.microphone:
            print("Closing microphone stream...")
            try:
                # Manually exit the microphone context
                self.microphone.__exit__(None, None, None)
                print("Microphone stream closed.")
            except Exception as e:
                print(f"Error closing microphone stream: {e}")

        # Reset resources
        self.microphone = None
        self.microphone_source = None
        self.recognizer = None
        self.listen_thread = None
        self.process_thread = None
        print("Voice assistant stopped and resources released.")
