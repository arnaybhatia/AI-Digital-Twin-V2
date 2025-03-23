import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('TMPT_API_KEY')
BASE_URL = "https://api.tmpt.app/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


