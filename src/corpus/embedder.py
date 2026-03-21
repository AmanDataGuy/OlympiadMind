"""
embedder.py
-----------
Step 3.4 — Corpus Embedding + FAISS Index

Embeds all corpus chunks using sentence-transformers (all-MiniLM-L6-v2)
and builds a FAISS IndexFlatIP for cosine similarity search.

Sources combined:
  - data/corpus/wikipedia_chunks.jsonl
  - data/corpus/stackexchange_chunks.jsonl
  - data/corpus/arxiv_chunks.jsonl

Output:
  - data/corpus/physics.faiss     — FAISS index
  - data/corpus/chunks.jsonl      — all chunks with metadata (same order as index)
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

CORPUS_FILES = [
    Path("data/corpus/wikipedia_chunks.jsonl"),
    Path("data/corpus/stackexchange_chunks.jsonl"),
    Path("data/corpus/arxiv_chunks.jsonl"),
]

FAISS_INDEX_PATH = Path("data/corpus/physics.faiss")
CHUNKS_PATH      = Path("data/corpus/chunks.jsonl")

MODEL_NAME   = "all-MiniLM-L6-v2"
BATCH_SIZE   = 256
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


# ── loaders ───────────────────────────────────────────────────────────────────

def load_all_chunks() -> list:
    """Load and combine all corpus chunks."""
    all_chunks = []

    for path in CORPUS_FILES:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        chunks = [json.loads(l) for l in lines]
        all_chunks.extend(chunks)
        print(f"  {path.name}: {len(chunks)} chunks")

    return all_chunks


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list, model) -> np.ndarray:
    """
    Embed all chunks in batches.
    Returns float32 numpy array of shape (n_chunks, embedding_dim).
    """
    texts       = [c.get("text", "") for c in chunks]
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
        batch      = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype(np.float32)


# ── faiss index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors).
    Normalize vectors first so inner product equals cosine similarity.
    """
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading corpus chunks...\n")
    chunks = load_all_chunks()
    print(f"\nTotal chunks: {len(chunks)}\n")

    print(f"Loading embedding model: {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    print("Embedding all chunks...")
    embeddings = embed_chunks(chunks, model)
    print(f"Embeddings shape: {embeddings.shape}\n")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"Index size: {index.ntotal} vectors\n")

    # save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    # save chunks in same order as index
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Chunks saved to: {CHUNKS_PATH}")

    print(f"\nDone.")
    print(f"  Total vectors indexed : {index.ntotal}")
    print(f"  Embedding dimension   : {EMBEDDING_DIM}")
    print(f"  Model                 : {MODEL_NAME}")


if __name__ == "__main__":
    main()