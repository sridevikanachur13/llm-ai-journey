import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

# Read the file
with open("Python/notes.txt", "r") as f:
    content = f.read()

    prompt = f"""Summarize the following text into exactly 3 bullet points.
Respond ONLY with valid JSON in this exact format, no markdown, no extra text:
{{"bullets": ["point1", "point2", "point3"]}}

Text: {content}"""

payload = {
    "contents": [{"parts": [{"text": prompt}]}]
}

response = requests.post(url, json=payload)
data = response.json()
raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

# Clean up in case the model wraps it in markdown code fences
clean_text = raw_text.replace("```json", "").replace("```", "").strip()

parsed = json.loads(clean_text)

for bullet in parsed["bullets"]:
    print("-", bullet)
