import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

# Read the file
with open("Python/notes.txt", "r") as f:
    content = f.read()

# Build the prompt dynamically using the file content
prompt = f"Summarize the following text in exactly 3 bullet points:\n\n{content}"

payload = {
    "contents": [{"parts": [{"text": prompt}]}]
}

response = requests.post(url, json=payload)
data = response.json()
summary = data["candidates"][0]["content"]["parts"][0]["text"]

print(summary)

# Save the result to a new file
with open("Python/summary.txt", "w") as f:
    f.write(summary)

print("\n✅ Summary saved to summary.txt")