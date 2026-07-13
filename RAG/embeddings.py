import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"

def get_embedding(text):
    payload = {
        "content": {"parts": [{"text": text}]}
    }
    response = requests.post(url, json=payload)
    data = response.json()
    return data["embedding"]["values"]

# Try it on a simple sentence
# vector = get_embedding("The dog ran fast across the park")
# print("Vector length:", len(vector))
# print("First 10 numbers:", vector[:10])

sentence1 = "The dog ran fast across the park"
sentence2 = "The canine sprinted quickly through the field"
sentence3 = "I love eating pizza on weekends"

vec1 = get_embedding(sentence1)
vec2 = get_embedding(sentence2)
vec3 = get_embedding(sentence3)

print("Vector 1 (first 5):", vec1[:5])
print("Vector 2 (first 5):", vec2[:5])
print("Vector 3 (first 5):", vec3[:5])