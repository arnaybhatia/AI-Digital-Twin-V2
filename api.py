from openai import OpenAI
import os
from typing import Optional
import time
from dotenv import load_dotenv
import threading
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_response(user_input: str, interrupt_event: Optional[threading.Event] = None) -> str:
    try:
        system_prompt = """You are Jim, a helpful AI assistant. Your responses will be read aloud, so format them to be clear and suitable for text-to-speech output. Focus on being helpful and direct while maintaining a conversational tone. Always organize your responses into clear, structured paragraphs without any bullet points, lists, or line breaks."""

        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_input})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                 if interrupt_event and interrupt_event.is_set():
                     return "Response interrupted."

                 response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                 if response.choices and response.choices[0].message and response.choices[0].message.content:
                     return response.choices[0].message.content
                 else:
                    return "No response generated."


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
