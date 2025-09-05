# embed_store.py
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

chunks_file = "chunks.pkl"
if not os.path.exists(chunks_file):
    print(f"⚠️ {chunks_file} not found! Run ingest.py first.")
    exit()

# Load chunks
with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)

if len(chunks) == 0:
    print("⚠️ No chunks found in chunks.pkl! Please check ingest.py output.")
    exit()

print(f"✅ Loaded {len(chunks)} chunks")

# Debug: check chunk lengths
for i, c in enumerate(chunks):
    print(f"Chunk {i} length:", len(c.page_content))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

# Create embeddings
embeddings = []
for c in chunks:
    vec = model.encode(c.page_content)
    embeddings.append(vec)

embeddings = np.array(embeddings).astype("float32")
print("✅ Embeddings shape:", embeddings.shape)

if embeddings.shape[0] == 0:
    print("⚠️ No embeddings created. Exiting.")
    exit()

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"✅ FAISS index created with {index.ntotal} vectors")

# Save index
faiss.write_index(index, "docs.index")
print("✅ FAISS index saved to docs.index")
