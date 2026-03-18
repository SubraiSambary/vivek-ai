# backend/vector_store.py
# ChromaDB wrapper — everything from Phase 2, now production-ready

import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
import os

CHROMA_PATH = "data/vector_store"
MODEL_NAME  = "all-MiniLM-L6-v2"

_client     = None
_collection = None

def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        emb_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        _collection = _client.get_or_create_collection(
            name="vivek_docs",
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def ingest_text(text: str, source: str, user_id: str = "default") -> int:
    """Chunk, embed and store plain text. Returns number of chunks added."""
    collection = _get_collection()
    chunks     = chunk_text(text)
    if not chunks:
        return 0

    existing = set(collection.get()["ids"])
    new_ids, new_docs, new_metas = [], [], []

    for i, chunk in enumerate(chunks):
        doc_id = f"{user_id}_{source}_{i}"
        if doc_id not in existing:
            new_ids.append(doc_id)
            new_docs.append(chunk)
            new_metas.append({"source": source, "user_id": user_id})

    if new_ids:
        collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas)

    return len(new_ids)

def ingest_pdf(file_path: str, user_id: str = "default") -> int:
    """Extract text from PDF, chunk and store. Returns chunks added."""
    doc  = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    source = os.path.basename(file_path)
    return ingest_text(text, source=source, user_id=user_id)

def retrieve_context(question: str, user_id: str = "default",
                     top_k: int = 3) -> str:
    """Return formatted context string to inject into VIVEK's prompt."""
    collection = _get_collection()
    if collection.count() == 0:
        return ""

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        where={"user_id": user_id} if user_id != "default" else None,
        include=["documents", "distances", "metadatas"]
    )

    if not results["documents"][0]:
        return ""

    parts = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        relevance = round(1 - dist, 3)
        if relevance > 0.25:  # only include actually relevant chunks
            parts.append(f"[From: {meta['source']} | Relevance: {relevance}]\n{doc}")

    return "\n\n".join(parts)

def get_doc_count(user_id: str = "default") -> int:
    collection = _get_collection()
    try:
        results = collection.get(where={"user_id": user_id})
        return len(results["ids"])
    except Exception:
        return collection.count()