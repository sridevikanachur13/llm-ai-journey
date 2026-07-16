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

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

with open("RAG/sample_doc.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = chunk_text(full_text)
print(f"Created {len(chunks)} chunks, embedding and storing them...")

client = chromadb.PersistentClient(path="./chroma_doc_data")
collection = client.get_or_create_collection(name="my_document")

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

collection.add(
    documents=chunks,
    embeddings=chunk_embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Now query it with a real question about YOUR document's content
query = "Explain react hooks"
query_embedding = get_embedding(query)

results = collection.query(query_embeddings=[query_embedding], n_results=2)

print("\nMost relevant chunks:")
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"\nDistance: {distance:.4f}")
    print(doc[:200], "...")