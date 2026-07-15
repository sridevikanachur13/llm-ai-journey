import chromadb
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"

def get_embedding(text):
    payload = {"content": {"parts": [{"text": text}]}}
    response = requests.post(embed_url, json=payload)
    data = response.json()
    return data["embedding"]["values"]

client = chromadb.PersistentClient(path="./chroma_data_gemini")
collection = client.get_or_create_collection(name="my_documents_gemini")

documents = [
    "The dog ran fast across the park",
    "I love eating pizza on weekends",
    "Stock markets fell sharply today",
    "My cat sleeps all day on the couch",
    "The economy is showing signs of recession"
]

# Generate embeddings ourselves, pass them in directly
doc_embeddings = [get_embedding(doc) for doc in documents]

collection.add(
    documents=documents,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# For querying, we must ALSO embed the query ourselves now (Chroma won't auto-embed if we're supplying our own)
query = "How is the financial market doing?"
query_embedding = get_embedding(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("Top matches (using Gemini embeddings):")
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"{distance:.4f} - {doc}")