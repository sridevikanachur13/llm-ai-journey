import chromadb
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

def get_embedding(text):
    payload = {"content": {"parts": [{"text": text}]}}
    response = requests.post(embed_url, json=payload)
    return response.json()["embedding"]["values"]

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(generate_url, json=payload)
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ===== STEP 1: INDEXING (run once) =====
print("Step 1: Indexing document...")

with open("RAG/sample_doc.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = chunk_text(full_text)
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

client = chromadb.PersistentClient(path="./chroma_rag_pipeline")
collection = client.get_or_create_collection(name="rag_docs")

# Avoid re-adding on repeated runs
existing = collection.count()
if existing == 0:
    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks.")
else:
    print(f"Already indexed ({existing} chunks found), skipping re-add.")

# ===== STEP 2: RETRIEVAL =====
question = "Name few react hooks"
print(f"\nStep 2: Retrieving relevant chunks for: '{question}'")

question_embedding = get_embedding(question)
results = collection.query(query_embeddings=[question_embedding], n_results=3)
retrieved_chunks = results["documents"][0]

print(f"Retrieved {len(retrieved_chunks)} chunks.")

# ===== STEP 3: GENERATION =====
print("\nStep 3: Generating grounded answer...")
answer = generate_answer(question, retrieved_chunks)

print("\n=== FINAL ANSWER ===")
print(answer)