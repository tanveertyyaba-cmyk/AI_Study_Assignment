from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path("data")
DB_DIR = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("python_notes")

def chunk_text(text, size=800):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

documents = []
ids = []

for pdf in DATA_DIR.glob("*.pdf"):
    reader = PdfReader(str(pdf))

    for page_no, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chunks = chunk_text(text)

        for chunk_no, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(chunk)
                ids.append(f"{pdf.stem}_{page_no}_{chunk_no}")

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print(f"Created {len(documents)} chunks and saved embeddings to ChromaDB.")