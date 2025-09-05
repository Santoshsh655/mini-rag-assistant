# app.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize FastAPI
app = FastAPI(title="Mini RAG Assistant")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index
index_path = "docs.index"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print("✅ FAISS index loaded")
else:
    index = None
    print("⚠️ FAISS index not found, run embed_store.py first")

# Load document chunks
chunks_path = "chunks.pkl"
if os.path.exists(chunks_path):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
else:
    chunks = []
    print("⚠️ Chunks not found, run ingest.py first")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

# Load LLM (small, CPU-friendly)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # better short answers
    device=-1                      # CPU only
)
print("✅ LLM pipeline ready")

# Pydantic model for incoming queries
class Query(BaseModel):
    question: str

# Root route
@app.get("/")
def root():
    return {"message": "RAG backend is running"}

# Ask route
@app.post("/ask")
def ask(query: Query):
    if index is None or len(chunks) == 0:
        return {"answer": "Index or chunks not loaded. Run ingest.py and embed_store.py first."}

    # Convert question to embedding
    q_embedding = embed_model.encode([query.question])

    # Retrieve top 3 most relevant chunks
    distances, indices = index.search(np.array(q_embedding).astype("float32"), k=3)
    context = " ".join([chunks[i].page_content for i in indices[0]])

    # Create prompt for LLM
    prompt = f"Answer the question briefly based only on the context.\n\nContext:\n{context}\n\nQuestion: {query.question}\nAnswer:"

    # Generate response
    response = llm(prompt, max_new_tokens=100, do_sample=False, temperature=0.3)

    # Extract only the part after "Answer:"
    generated = response[0]["generated_text"]
    if "Answer:" in generated:
        final_answer = generated.split("Answer:")[-1].strip()
    else:
        final_answer = generated.strip()

    return {"answer": final_answer}
