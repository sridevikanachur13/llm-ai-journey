import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

payload = {
    "contents": [
        {"parts": [{"text": "Explain what an API is, in 2 sentences, to a frontend developer."}]}
    ]
}

response = requests.post(url, json=payload)
data = response.json()

print(data["candidates"][0]["content"]["parts"][0]["text"])