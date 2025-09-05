# ingest.py
import os
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

docs = []

# Process TXT and PDF files in ../docs
docs_dir = "../docs"

for filename in os.listdir(docs_dir):
    file_path = os.path.join(docs_dir, filename)
    
    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text))
        print(f"✅ TXT file added: {filename}")
    
    elif filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append(Document(page_content=text))
        print(f"✅ PDF file added: {filename}")

# Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Chunks saved: {len(chunks)}")
