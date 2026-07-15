import os
import requests
from dotenv import load_dotenv
import math

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

def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    return dot_product / (magnitude_a * magnitude_b)

# Test sentences
sentence1 = "The dog ran fast across the park"
sentence2 = "The canine sprinted quickly through the field"
sentence3 = "I love eating pizza on weekends"

vec1 = get_embedding(sentence1)
vec2 = get_embedding(sentence2)
vec3 = get_embedding(sentence3)

print("Similarity (dog/canine - should be HIGH):", cosine_similarity(vec1, vec2))
print("Similarity (dog/pizza - should be LOW):", cosine_similarity(vec1, vec3))

documents = [
    "The dog ran fast across the park",
    "I love eating pizza on weekends",
    "Stock markets fell sharply today",
    "My cat sleeps all day on the couch",
    "The economy is showing signs of recession"
]

query = "How is the financial market doing?"

query_vec = get_embedding(query)
doc_vecs = [get_embedding(doc) for doc in documents]

scores = [(documents[i], cosine_similarity(query_vec, doc_vecs[i])) for i in range(len(documents))]
scores.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by relevance to query:")
for doc, score in scores:
    print(f"{score:.4f} - {doc}")