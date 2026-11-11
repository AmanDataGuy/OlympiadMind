"""
graph_builder.py
----------------
Neo4j Knowledge Graph Builder

Reads data/extracted/documents.jsonl and writes every document
as a Problem node in Neo4j with metadata for filtering.

Node properties stored:
  - id, source, doc_type, competition, year
  - problem_text (first page text, truncated)
  - equation_count, figure_count
  - total_pages

Does NOT write concept relationships yet — those come from
src/graph/concept_extractor_modal.py.

Output: Problem nodes in Neo4j AuraDB
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

DOCUMENTS_PATH = Path("data/extracted/documents.jsonl")
BATCH_SIZE     = 100   # nodes per transaction


# ── connection ────────────────────────────────────────────────────────────────

def get_driver():
    load_dotenv()
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )


# ── node writer ───────────────────────────────────────────────────────────────

def write_problem_batch(session, batch: list) -> None:
    """
    Write a batch of Problem nodes using MERGE — safe to rerun.
    MERGE creates the node if it doesn't exist, updates if it does.
    """
    session.run("""
        UNWIND $batch AS doc
        MERGE (p:Problem {id: doc.id})
        SET
            p.source         = doc.source,
            p.doc_type       = doc.doc_type,
            p.competition    = doc.competition,
            p.year           = doc.year,
            p.problem_text   = doc.problem_text,
            p.equation_count = doc.equation_count,
            p.figure_count   = doc.figure_count,
            p.total_pages    = doc.total_pages
    """, batch=batch)


def extract_problem_text(doc: dict) -> str:
    """
    Get the first meaningful text from the document.
    Truncated to 500 chars for Neo4j storage.
    """
    pages = doc.get("pages", [])
    for page in pages:
        text = page.get("text", "").strip()
        if len(text) > 50:
            return text[:500]
    return ""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    driver = get_driver()

    print("Loading documents...")
    docs = []
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"  {len(docs)} documents loaded\n")

    # prepare node records
    records = []
    for doc in docs:
        records.append({
            "id":            doc["doc_id"],
            "source":        doc.get("source", ""),
            "doc_type":      doc.get("doc_type", ""),
            "competition":   doc.get("competition") or "",
            "year":          str(doc.get("year") or ""),
            "problem_text":  extract_problem_text(doc),
            "equation_count": doc.get("equation_count", 0),
            "figure_count":   doc.get("figure_count", 0),
            "total_pages":    doc.get("total_pages", 0),
        })

    # write in batches
    print(f"Writing {len(records)} Problem nodes to Neo4j...\n")
    written = 0

    with driver.session() as session:
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Writing nodes"):
            batch = records[i:i + BATCH_SIZE]
            write_problem_batch(session, batch)
            written += len(batch)

    # verify
    with driver.session() as session:
        result = session.run("MATCH (p:Problem) RETURN count(p) AS total")
        total  = result.single()["total"]

    print(f"\nDone.")
    print(f"  Nodes written : {written}")
    print(f"  Total in Neo4j: {total}")

    driver.close()


if __name__ == "__main__":
    main()