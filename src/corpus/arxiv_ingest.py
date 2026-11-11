"""
arxiv_ingest.py
---------------
arXiv Physics Abstracts Ingestion

Fetches physics education and classical physics abstracts from arXiv API.
Focuses on categories relevant to olympiad-level physics.

Output: data/corpus/arxiv_chunks.jsonl
"""

import json
import time
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE = Path("data/corpus/arxiv_chunks.jsonl")

SEARCH_QUERIES = [
    "classical mechanics olympiad physics",
    "electromagnetism undergraduate physics problems",
    "thermodynamics statistical mechanics teaching",
    "optics wave physics undergraduate",
    "special relativity problems solutions",
    "quantum mechanics introductory physics",
    "fluid mechanics physics problems",
    "oscillations waves physics education",
    "electromagnetic induction Faraday physics",
    "gravitational physics classical problems",
]

MAX_RESULTS_PER_QUERY = 50


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest_arxiv(queries: list, max_per_query: int) -> list:
    import arxiv

    all_chunks = []
    seen_ids   = set()

    for query in tqdm(queries, desc="Fetching arXiv abstracts"):
        try:
            search = arxiv.Search(
                query       = query,
                max_results = max_per_query,
                sort_by     = arxiv.SortCriterion.Relevance,
            )

            for result in search.results():
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)

                text = f"{result.title}. {result.summary}"
                text = text.replace("\n", " ").strip()

                if len(text.split()) < 30:
                    continue

                all_chunks.append({
                    "chunk_id":   f"arxiv_{result.entry_id.split('/')[-1]}",
                    "source":     "arxiv",
                    "topic":      query,
                    "title":      result.title,
                    "text":       text,
                    "word_count": len(text.split()),
                    "categories": result.categories if isinstance(result.categories, list) else [],
                    "published":  str(result.published.date()),
                })

            time.sleep(1)

        except Exception as e:
            print(f"\n  FAILED: {query} — {e}")

    return all_chunks


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching arXiv abstracts for {len(SEARCH_QUERIES)} queries...\n")
    chunks = ingest_arxiv(SEARCH_QUERIES, MAX_RESULTS_PER_QUERY)

    seen          = set()
    unique_chunks = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique_chunks.append(c)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in unique_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Total abstracts : {len(unique_chunks)}")
    print(f"  Output          : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()