# tests/test_chromadb.py
# Phase 2, Ex 2: ChromaDB — our toy vector store, but production-grade
# ChromaDB is just our ToyVectorStore from Phase 1 with:
# - HNSW index (fast even with millions of docs)
# - Persistent storage to disk
# - Metadata filtering
# - Built-in embedding support

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

CHROMA_PATH = "data/vector_store"

# ─────────────────────────────────────────────
# PART 1: Create persistent ChromaDB client
# ─────────────────────────────────────────────

print("=== Setting up ChromaDB ===")

# PersistentClient saves to disk — data survives restarts
# Just like our JSON file in Phase 1, but optimized for vectors
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use sentence-transformers as the embedding function
# ChromaDB will auto-embed text — no manual model.encode() needed!
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# A "collection" = a table in a regular database
# get_or_create = safe to run multiple times
collection = client.get_or_create_collection(
    name="vivek_knowledge",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

print(f"✅ Collection ready: '{collection.name}'")
print(f"   Docs already stored: {collection.count()}\n")

# ─────────────────────────────────────────────
# PART 2: Add documents (with auto-embedding)
# ─────────────────────────────────────────────

# Simulate chunks from different sources
docs = [
    {"id": "ml_001",      "text": "Machine learning is a subset of AI where systems learn from data to improve automatically.", "source": "textbook"},
    {"id": "ml_002",      "text": "Neural networks are inspired by the human brain with layers of interconnected nodes.", "source": "textbook"},
    {"id": "vivek_001",   "text": "VIVEK uses ChromaDB to store document embeddings for semantic search.", "source": "vivek_docs"},
    {"id": "vivek_002",   "text": "RAG stands for Retrieval Augmented Generation — retrieve context, then generate answer.", "source": "vivek_docs"},
    {"id": "cooking_001", "text": "Biryani is a fragrant rice dish made with spices, basmati rice, and either meat or vegetables.", "source": "recipes"},
    {"id": "coding_001",  "text": "Python list comprehensions create lists in a single readable line: [x*2 for x in range(10)]", "source": "python_docs"},
    {"id": "coding_002",  "text": "FastAPI is a modern Python web framework for building APIs with automatic docs generation.", "source": "python_docs"},
    {"id": "vivek_003",   "text": "VIVEK remembers users with SQLite — storing name, topics, and conversation history.", "source": "vivek_docs"},
]

# Only add docs not already in the collection
existing_ids = set(collection.get()["ids"])
new_docs = [d for d in docs if d["id"] not in existing_ids]

if new_docs:
    collection.add(
        ids=[d["id"] for d in new_docs],
        documents=[d["text"] for d in new_docs],
        metadatas=[{"source": d["source"]} for d in new_docs],
    )
    print(f"✅ Added {len(new_docs)} new documents")
else:
    print("✅ All documents already in collection")

print(f"   Total in collection: {collection.count()}\n")

# ─────────────────────────────────────────────
# PART 3: Query — semantic search
# ─────────────────────────────────────────────

print("=== Semantic Search Queries ===\n")

queries = [
    "How does machine learning work?",
    "Tell me about VIVEK's memory system",
    "I want to cook Indian food",
    "How do I build an API in Python?",
]

for query in queries:
    results = collection.query(
        query_texts=[query],
        n_results=2,
        include=["documents", "distances", "metadatas"]
    )
    print(f"Query: '{query}'")
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    )):
        similarity = 1 - dist  # ChromaDB returns distance, not similarity
        print(f"  {i+1}. [{similarity:.4f}] ({meta['source']}) {doc[:70]}...")
    print()

# ─────────────────────────────────────────────
# PART 4: Metadata filtering — only search specific sources
# ─────────────────────────────────────────────

print("=== Metadata Filtering (search only vivek_docs) ===\n")

results = collection.query(
    query_texts=["How does VIVEK work?"],
    n_results=3,
    where={"source": "vivek_docs"},  # filter by metadata!
    include=["documents", "distances"]
)

print("Query: 'How does VIVEK work?' (filtered to vivek_docs only)")
for i, (doc, dist) in enumerate(zip(
    results["documents"][0],
    results["distances"][0]
)):
    print(f"  {i+1}. [{1-dist:.4f}] {doc[:80]}...")

# ─────────────────────────────────────────────
# PART 5: This is VIVEK's real RAG function
# ─────────────────────────────────────────────

print("\n=== VIVEK's real retrieve_context() function ===\n")

def retrieve_context(question: str, collection, top_k: int = 3,
                     source_filter: str = None) -> str:
    """
    This exact function goes into backend/vector_store.py in Phase 3.
    Returns a formatted string of relevant chunks to inject into VIVEK's prompt.
    """
    query_params = {
        "query_texts": [question],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"]
    }
    if source_filter:
        query_params["where"] = {"source": source_filter}

    results = collection.query(**query_params)

    if not results["documents"][0]:
        return "No relevant context found in documents."

    context_parts = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = round(1 - dist, 3)
        context_parts.append(
            f"[Source: {meta['source']} | Relevance: {similarity}]\n{doc}"
        )

    return "\n\n".join(context_parts)

context = retrieve_context("How does VIVEK use memory and embeddings?", collection)
print("Context retrieved for VIVEK's prompt:")
print(context)
print("\n✅ Phase 2 complete! ChromaDB is ready. This goes into backend/vector_store.py next.")