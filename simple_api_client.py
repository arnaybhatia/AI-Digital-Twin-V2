import requests
import os
import sys
import time
from dotenv import load_dotenv


"""
This script demonstrates how to use the Tmpt.me REST API to create a thread and post a message.
To use it, you must create a .env file in the current directory with the following content:

    TMPT_API_KEY=your_api_key

The full API specification is available at:

    https://dev.tmpt.app/api.yaml

The Website https://editor-next.swagger.io/ offers a convenient way to load the API and explore it interactively.
"""


# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("TMPT_API_KEY")
BASE_URL = os.getenv("TMPT_API_URL", "https://api.tmpt.app/v1")

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
    print(url)
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
        sys.exit(1)


def main():
    print("Getting client token...")
    client_token_data = make_request(
        "POST",
        "/client_token",
        json={"external_user_id": f"user_{int(time.time())}", "is_reviewable": False}
    )
    client_token = client_token_data["client_token_id"]
    print(f"Client token received: {client_token}")

    print("Creating thread...")
    thread_data = make_request(
        "POST",
        "/threads",
        json={"client_token_id": client_token}
    )
    thread_id = thread_data["id"]
    print(f"Thread created with ID: {thread_id}")

    message_text = "What is a carburetor?"
    print(f"Posting message: '{message_text}'")
    message_data = make_request(
        "POST",
        f"/threads/{thread_id}/messages",
        json={
            "client_token_id": client_token,
            "text": message_text
        }
    )
    message_id = message_data["id"]
    print(f"Message posted with ID: {message_id}")

    print("Waiting for reply...")
    try:
        reply_data = make_request(
            "GET",
            f"/threads/{thread_id}/reply/{message_id}",
            params={"client_token_id": client_token, "timeout": 15}
        )

        print("\nAgent's reply:")
        print("-" * 40)
        print(reply_data["text"])
        print("-" * 40)
        return

    except Exception as e:
        print(f"Error getting reply: {e}")

    # As a fallback, simply list all messages
    print("Fetching all messages in the thread...")
    messages = make_request(
        "GET",
        f"/threads/{thread_id}/messages",
        params={"client_token_id": client_token}
    )

    for msg in messages:
        print(f"\n{msg['speaker']} said: {msg['text']}")


if __name__ == "__main__":
    main()