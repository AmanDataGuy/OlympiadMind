"""
wikipedia_ingest.py
-------------------
Step 3.1 — Wikipedia Physics Corpus Ingestion

Fetches 30 core physics topic articles from Wikipedia and chunks them
into 512-token segments with 64-token overlap.

Output: data/corpus/wikipedia_chunks.jsonl
"""

import json
import re
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("data/corpus/wikipedia_chunks.jsonl")

CHUNK_SIZE  = 512   # tokens (approximate — using word count as proxy)
OVERLAP     = 64

PHYSICS_TOPICS = [
    # Mechanics
    "Newton's laws of motion",
    "Conservation of energy",
    "Conservation of momentum",
    "Rotational dynamics",
    "Simple harmonic motion",
    "Fluid mechanics",
    "Gravitational potential energy",

    # Electromagnetism
    "Gauss's law",
    "Faraday's law of induction",
    "Maxwell's equations",
    "Electromagnetic waves",
    "Electric potential",
    "Magnetic field",

    # Thermodynamics
    "Laws of thermodynamics",
    "Carnot cycle",
    "Entropy",
    "Ideal gas law",
    "Heat transfer",

    # Optics
    "Snell's law",
    "Diffraction",
    "Interference of light",
    "Geometric optics",

    # Modern Physics
    "Special relativity",
    "Photoelectric effect",
    "Bohr model",
    "De Broglie wavelength",
    "Nuclear fission",

    # Waves
    "Wave equation",
    "Doppler effect",
    "Standing wave",
]


# ── chunking ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove Wikipedia markup artifacts and normalize whitespace."""
    # remove section headers left by wikipedia library
    text = re.sub(r"\n==+.+?==+\n", "\n", text)
    # remove citation markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, topic: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    Each chunk has overlap words from the previous chunk for context continuity.
    """
    words  = text.split()
    chunks = []
    i      = 0
    idx    = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text  = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"wiki_{topic.replace(' ', '_')}_{idx}",
            "source":   "wikipedia",
            "topic":    topic,
            "text":     chunk_text,
            "word_count": len(chunk_words),
        })

        i   += chunk_size - overlap
        idx += 1

    return chunks


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest_wikipedia(topics: list) -> list:
    import wikipedia

    wikipedia.set_lang("en")

    all_chunks = []
    failed     = []

    for topic in tqdm(topics, desc="Fetching Wikipedia articles"):
        try:
            page  = wikipedia.page(topic, auto_suggest=False)
            text  = clean_text(page.content)
            chunks = chunk_text(text, topic)
            all_chunks.extend(chunks)
            print(f"  {topic}: {len(page.content.split())} words → {len(chunks)} chunks")

        except wikipedia.exceptions.DisambiguationError as e:
            # try the first option
            try:
                page   = wikipedia.page(e.options[0], auto_suggest=False)
                text   = clean_text(page.content)
                chunks = chunk_text(text, topic)
                all_chunks.extend(chunks)
                print(f"  {topic} (via {e.options[0]}): {len(chunks)} chunks")
            except Exception:
                failed.append(topic)
                print(f"  FAILED: {topic}")

        except Exception as ex:
            failed.append(topic)
            print(f"  FAILED: {topic} — {ex}")

    return all_chunks, failed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(PHYSICS_TOPICS)} Wikipedia articles...\n")
    chunks, failed = ingest_wikipedia(PHYSICS_TOPICS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Failed topics: {len(failed)}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()