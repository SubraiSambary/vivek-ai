# backend/vector_store.py
# Lightweight RAG store — TF-IDF instead of sentence-transformers
# RAM usage: ~10MB vs ~400MB. No model download. No PyTorch. 🎉

import os
import json
import pickle
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

STORE_PATH = "data/vector_store.pkl"   # persists across requests

# ── In-process store ─────────────────────────────────────
# Structure: { "doc_id": {"text": str, "source": str, "user_id": str} }
_store: dict = {}
_vectorizer: TfidfVectorizer = None
_matrix = None          # sparse TF-IDF matrix (all docs)
_doc_ids: list = []     # ordered list matching matrix rows

def _load():
    """Load persisted store from disk on cold start."""
    global _store
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "rb") as f:
                _store = pickle.load(f)
        except Exception:
            _store = {}

def _save():
    """Persist store to disk."""
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(_store, f)

def _rebuild_index():
    """Rebuild TF-IDF matrix from current _store. Call after any add."""
    global _vectorizer, _matrix, _doc_ids
    if not _store:
        _vectorizer, _matrix, _doc_ids = None, None, []
        return
    _doc_ids   = list(_store.keys())
    texts      = [_store[d]["text"] for d in _doc_ids]
    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams → better recall
        min_df=1,
        max_features=20_000,  # cap RAM usage
        sublinear_tf=True,    # log-scale TF
    )
    _matrix = _vectorizer.fit_transform(texts)

# Load on import (cold start)
_load()
_rebuild_index()


# ── Text chunking ────────────────────────────────────────
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


# ── Ingestion ────────────────────────────────────────────
def ingest_text(text: str, source: str, user_id: str = "default") -> int:
    """Chunk and store plain text. Returns number of new chunks added."""
    chunks  = chunk_text(text)
    added   = 0
    for i, chunk in enumerate(chunks):
        doc_id = f"{user_id}_{source}_{i}"
        if doc_id not in _store:
            _store[doc_id] = {"text": chunk, "source": source, "user_id": user_id}
            added += 1
    if added:
        _save()
        _rebuild_index()
    return added

def ingest_pdf(file_path: str, user_id: str = "default") -> int:
    """Extract text from PDF, chunk and store. Returns chunks added."""
    doc    = fitz.open(file_path)
    text   = "\n".join(page.get_text() for page in doc)
    doc.close()
    source = os.path.basename(file_path)
    return ingest_text(text, source=source, user_id=user_id)


# ── Retrieval ────────────────────────────────────────────
def retrieve_context(question: str, user_id: str = "default",
                     top_k: int = 3) -> str:
    """Return formatted context string to inject into VIVEK's prompt."""
    if _vectorizer is None or _matrix is None or not _doc_ids:
        return ""

    # Filter to this user's docs
    user_indices = [
        i for i, d in enumerate(_doc_ids)
        if _store[d]["user_id"] == user_id
    ]
    if not user_indices:
        # Fall back to all docs if user has none specifically tagged
        user_indices = list(range(len(_doc_ids)))
    if not user_indices:
        return ""

    q_vec        = _vectorizer.transform([question])
    user_matrix  = _matrix[user_indices]
    scores       = cosine_similarity(q_vec, user_matrix)[0]

    top_local    = np.argsort(scores)[::-1][:top_k]
    parts        = []
    for local_i in top_local:
        score  = scores[local_i]
        if score < 0.05:        # skip irrelevant chunks
            continue
        global_i = user_indices[local_i]
        doc_id   = _doc_ids[global_i]
        entry    = _store[doc_id]
        relevance = round(float(score), 3)
        parts.append(
            f"[From: {entry['source']} | Relevance: {relevance}]\n{entry['text']}"
        )

    return "\n\n".join(parts)


# ── Utility ──────────────────────────────────────────────
def get_doc_count(user_id: str = "default") -> int:
    return sum(1 for v in _store.values() if v["user_id"] == user_id)
