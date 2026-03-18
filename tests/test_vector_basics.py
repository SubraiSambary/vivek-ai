# tests/test_vector_basics.py
# Phase 1, Ex 3: Build a mini vector store from scratch
# Goal: understand WHY vector databases exist and HOW similarity search works
# No ChromaDB yet — just numpy + dicts

import numpy as np
import json

# ─────────────────────────────────────────────
# PART 1: What IS a vector / embedding?
# ─────────────────────────────────────────────

# Imagine every piece of text gets converted to a list of numbers.
# Similar texts → similar numbers → close together in "space"
# This is what sentence-transformers does in Phase 2.
# For now, we fake it with hand-crafted vectors.

# Think of these as 4-dimensional "meaning coordinates":
# [topic_ai, topic_food, topic_sport, topic_code]
fake_embeddings = {
    "What is machine learning?":    np.array([0.9, 0.0, 0.0, 0.6]),
    "Explain neural networks":      np.array([0.8, 0.0, 0.0, 0.7]),
    "How to make biryani?":         np.array([0.0, 0.9, 0.0, 0.0]),
    "Best cricket players in 2024": np.array([0.0, 0.0, 0.9, 0.0]),
    "Python list comprehensions":   np.array([0.1, 0.0, 0.0, 0.9]),
    "What is a vector database?":   np.array([0.7, 0.0, 0.0, 0.5]),
}

# ─────────────────────────────────────────────
# PART 2: Cosine similarity — the heart of vector search
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Measures how similar two vectors are.
    Returns 1.0 = identical direction, 0.0 = unrelated, -1.0 = opposite.
    This exact formula runs inside ChromaDB, Pinecone, Weaviate — all of them.
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

# Quick test
vec1 = np.array([1.0, 0.0, 0.0, 0.0])  # pure AI topic
vec2 = np.array([0.9, 0.0, 0.0, 0.1])  # mostly AI
vec3 = np.array([0.0, 1.0, 0.0, 0.0])  # pure food topic

print("=== Cosine Similarity Tests ===")
print(f"AI vs mostly-AI:  {cosine_similarity(vec1, vec2):.4f}  (should be HIGH ~0.99)")
print(f"AI vs food:       {cosine_similarity(vec1, vec3):.4f}  (should be LOW  ~0.00)")

# ─────────────────────────────────────────────
# PART 3: Our toy vector store — a dict of dicts
# ─────────────────────────────────────────────

class ToyVectorStore:
    """
    A miniature ChromaDB. Stores text + its embedding + metadata.
    ChromaDB does exactly this, just with millions of vectors and
    fast index structures (HNSW) instead of a plain loop.
    """

    def __init__(self):
        # The "database" is just a dict!
        # { doc_id: { "text": ..., "embedding": ..., "metadata": ... } }
        self.store: dict = {}
        self.next_id: int = 0

    def add(self, text: str, embedding: np.ndarray, metadata: dict = None) -> str:
        """Add a document + its embedding to the store."""
        doc_id = f"doc_{self.next_id}"
        self.store[doc_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.next_id += 1
        return doc_id

    def query(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        """
        Find the top_k most similar documents to the query.
        This is a brute-force scan — compares query against every stored vector.
        ChromaDB uses HNSW index to do this in milliseconds even with millions of docs.
        """
        scores = []
        for doc_id, doc in self.store.items():
            score = cosine_similarity(query_embedding, doc["embedding"])
            scores.append((score, doc_id, doc))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        return [
            {
                "score": round(score, 4),
                "text": doc["text"],
                "metadata": doc["metadata"],
            }
            for score, doc_id, doc in scores[:top_k]
        ]

    def count(self) -> int:
        return len(self.store)


# ─────────────────────────────────────────────
# PART 4: Simulate VIVEK's doc Q&A pipeline
# ─────────────────────────────────────────────

print("\n=== Building VIVEK's Toy Vector Store ===")

vs = ToyVectorStore()

# Simulate "chunking + embedding" a document
# In Phase 2, sentence-transformers does the embedding automatically
# Here we use our fake_embeddings dict
for text, embedding in fake_embeddings.items():
    doc_id = vs.add(
        text=text,
        embedding=embedding,
        metadata={"source": "vivek_knowledge_base", "phase": 1}
    )
    print(f"  Added [{doc_id}]: {text[:45]}...")

print(f"\nTotal docs stored: {vs.count()}")

# ─────────────────────────────────────────────
# PART 5: Query the store — this is RAG in action
# ─────────────────────────────────────────────

print("\n=== Semantic Search Queries ===")

# Query 1: AI question
query1 = np.array([0.85, 0.0, 0.0, 0.55])  # AI + code flavor
print("\nQuery: 'Tell me about AI and coding'")
results = vs.query(query1, top_k=3)
for i, r in enumerate(results, 1):
    print(f"  {i}. [{r['score']}] {r['text']}")

# Query 2: Food question  
query2 = np.array([0.0, 0.85, 0.05, 0.0])
print("\nQuery: 'I want to cook something'")
results = vs.query(query2, top_k=3)
for i, r in enumerate(results, 1):
    print(f"  {i}. [{r['score']}] {r['text']}")

# ─────────────────────────────────────────────
# PART 6: The RAG loop — how VIVEK answers from docs
# ─────────────────────────────────────────────

print("\n=== RAG Pipeline Simulation ===")

def vivek_rag_answer(user_question: str, query_embedding: np.ndarray,
                     vector_store: ToyVectorStore) -> str:
    """
    RAG = Retrieval Augmented Generation
    Step 1: Retrieve relevant chunks from vector store
    Step 2: Stuff them into VIVEK's prompt as context
    Step 3: LLM generates answer using that context
    In Phase 3 this calls Groq. Here we just show the prompt.
    """
    # Step 1: Retrieve
    results = vector_store.query(query_embedding, top_k=2)
    context_chunks = "\n".join([f"- {r['text']}" for r in results])

    # Step 2: Build prompt (this goes to Groq in Phase 3)
    prompt = f"""You are VIVEK. Answer the question using ONLY the context below.
If the answer isn't in the context, say "Yaar, I don't have that info! 🤷"

Context from documents:
{context_chunks}

User question: {user_question}

VIVEK's answer:"""

    # Step 3: (Phase 3 sends this to Groq — for now we print it)
    print(f"  Question: {user_question}")
    print(f"  Top retrieved chunks:")
    for r in results:
        print(f"    [{r['score']}] {r['text']}")
    print(f"\n  Prompt sent to LLM:\n{prompt}")
    return prompt

vivek_rag_answer(
    user_question="How do neural networks relate to machine learning?",
    query_embedding=np.array([0.85, 0.0, 0.0, 0.6]),
    vector_store=vs,
)

print("\n✅ You now understand what ChromaDB does internally!")
print("   Next: sentence-transformers will replace fake_embeddings automatically.")