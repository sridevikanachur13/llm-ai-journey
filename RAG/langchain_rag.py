import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")

# ===== STEP 1: LOAD + CHUNK (LangChain's splitter, vs your manual chunk_text()) =====
with open("RAG/sample_doc.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([full_text])  # returns Document objects, not plain strings

print(f"Created {len(chunks)} chunks using LangChain's splitter.")

# ===== STEP 2: EMBEDDINGS (LangChain's wrapper around Gemini's embedding API) =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# ===== STEP 3: VECTOR STORE (LangChain's Chroma wrapper - handles embedding AND storing in one call) =====
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_data"
)

print("Indexed into ChromaDB via LangChain.")

# ===== STEP 4: RETRIEVAL =====
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

question = "Name few react hooks"
retrieved_docs = retriever.invoke(question)

print(f"\nRetrieved {len(retrieved_docs)} chunks for: '{question}'")

# ===== STEP 5: GENERATION =====
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])
prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

response = llm.invoke(prompt)
print("\n=== FINAL ANSWER ===")
print(response.content)