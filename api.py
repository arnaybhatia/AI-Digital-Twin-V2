import requests
import os
import sys
import time
from dotenv import load_dotenv
import threading
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv("TMPT_API_KEY")
BASE_URL = os.getenv("TMPT_API_URL", "https://api.tmpt.app/v1")
CLIENT_TOKEN = None
THREAD_ID = None

if not API_KEY:
    print("Error: TMPT_API_KEY not found in .env file")
    sys.exit(1)

# Common headers for all requests
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def make_request(method, endpoint, **kwargs):
    """Helper function to make API requests with error handling"""
    url = BASE_URL + endpoint
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            **kwargs
        )
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
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
            return "I couldn't process your message. Please try again."
        
        message_id = message_data["id"]
        
        # Wait for reply
        max_retries = 3
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
                    # As a fallback, try to get all messages
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
                    
                    return "I didn't receive a proper response. Please try again."
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[DEBUG] Final API attempt failed: {str(e)}")
                    return "I encountered an error while processing your request. Please try again."
                print(f"[DEBUG] API attempt {attempt + 1} failed, retrying... Error: {e}")
                time.sleep(1)
                
    except Exception as e:
        print(f"[DEBUG] Error in get_response: {str(e)}")
        return "I encountered an error while processing your request. Please try again."
    
    return "I encountered an unexpected error. Please try again."
