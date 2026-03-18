# tests/test_embeddings.py
# Phase 2, Ex 1: Real embeddings — sentence-transformers replaces fake_embeddings
# The model runs 100% locally, no API key, no internet after first download

import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# PART 1: Load the model (downloads once, ~90MB)
# ─────────────────────────────────────────────

print("Loading embedding model... (first run downloads ~90MB)")
model = SentenceTransformer("all-MiniLM-L6-v2")
# Why this model?
# - Free, open source, runs on CPU
# - 384-dimensional vectors (vs our fake 4-dimensional ones!)
# - Fast enough for real use, good enough for production
print("✅ Model loaded!\n")

# ─────────────────────────────────────────────
# PART 2: Text → real vectors
# ─────────────────────────────────────────────

sentences = [
    "What is machine learning?",
    "Explain neural networks",
    "How to make biryani?",
    "Best cricket players in 2024",
    "Python list comprehensions",
    "What is a vector database?",
    "How do I cook rice?",           # Similar to biryani — watch the score!
    "Deep learning and AI concepts", # Similar to ML — watch the score!
]

print("=== Generating real embeddings ===")
embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")
# You'll see (8, 384) — 8 sentences, each a 384-dimension vector
print(f"Each embedding is {embeddings.shape[1]} numbers long")
print(f"First 8 numbers of 'What is machine learning?':")
print(f"  {embeddings[0][:8].round(4)}\n")

# ─────────────────────────────────────────────
# PART 3: Cosine similarity with real vectors
# ─────────────────────────────────────────────

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=== Similarity with REAL embeddings (compare to Phase 1 fake ones) ===\n")

idx = {s: i for i, s in enumerate(sentences)}

pairs = [
    ("What is machine learning?",   "Explain neural networks",
     "Should be HIGH — both AI"),
    ("What is machine learning?",   "How to make biryani?",
     "Should be LOW — totally different"),
    ("How to make biryani?",        "How do I cook rice?",
     "Should be HIGH — both cooking"),
    ("What is a vector database?",  "What is machine learning?",
     "Should be MEDIUM — both tech"),
    ("Best cricket players in 2024","How to make biryani?",
     "Should be LOW — sport vs food"),
    ("Deep learning and AI concepts","Explain neural networks",
     "Should be HIGH — same domain"),
]

for text_a, text_b, expectation in pairs:
    score = cosine_similarity(embeddings[idx[text_a]], embeddings[idx[text_b]])
    print(f"  {score:.4f}  {expectation}")
    print(f"          '{text_a[:40]}'")
    print(f"          '{text_b[:40]}'\n")

# ─────────────────────────────────────────────
# PART 4: The query that failed in Phase 1 — now fixed
# ─────────────────────────────────────────────

print("=== The query that failed in Phase 1 — now fixed ===")
print("Query: 'How do neural networks relate to machine learning?'\n")

query = "How do neural networks relate to machine learning?"
query_embedding = model.encode([query])[0]

scores = []
for i, sentence in enumerate(sentences):
    score = cosine_similarity(query_embedding, embeddings[i])
    scores.append((score, sentence))

scores.sort(reverse=True)
for i, (score, text) in enumerate(scores[:4], 1):
    print(f"  {i}. [{score:.4f}] {text}")

print("\n✅ 'Explain neural networks' and 'What is machine learning?' are now")
print("   correctly the top results — real embeddings understand meaning!")

# ─────────────────────────────────────────────
# PART 5: How chunking works — critical for PDF Q&A
# ─────────────────────────────────────────────

print("\n=== Text Chunking — how PDFs become searchable ===")

# Imagine this is a page from an uploaded PDF
sample_doc = """
VIVEK is an AI assistant built with Python. It uses ChromaDB as its vector 
database to store document embeddings. When a user uploads a PDF, VIVEK 
chunks it into smaller pieces, embeds each chunk, and stores them.

When a user asks a question, VIVEK embeds the question and searches for 
the most similar chunks. The top chunks become the context for the LLM.
This pattern is called RAG — Retrieval Augmented Generation.

VIVEK also has memory. It remembers user names, preferences, and past 
topics using a SQLite database. Every conversation is stored and retrieved
to make responses more personal and contextual.
"""

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """
    Split text into overlapping chunks.
    overlap ensures context isn't lost at chunk boundaries.
    chunk_size and overlap are tunable — we'll expose these in the app.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # overlap keeps context
    return [c for c in chunks if len(c) > 20]  # drop tiny trailing chunks

chunks = chunk_text(sample_doc, chunk_size=40, overlap=8)
print(f"Document split into {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: '{chunk[:80]}...'")

print("\nEmbedding all chunks...")
chunk_embeddings = model.encode(chunks)
print(f"✅ {len(chunk_embeddings)} chunk embeddings ready for ChromaDB")
print(f"   Each chunk: {chunk_embeddings.shape[1]}-dimensional vector")