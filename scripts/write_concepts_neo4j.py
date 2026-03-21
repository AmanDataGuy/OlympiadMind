import json
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

labels = [json.loads(l) for l in Path("data/corpus/concept_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

written = 0
with driver.session() as session:
    for record in tqdm(labels, desc="Writing concepts"):
        doc_id   = record["doc_id"]
        concepts = record["concepts"]
        if not concepts:
            continue
        for concept in concepts:
            session.run("""
                MERGE (c:Concept {canonical_name: $concept})
                WITH c
                MATCH (p:Problem {id: $doc_id})
                MERGE (p)-[:REQUIRES]->(c)
            """, concept=concept.lower().strip(), doc_id=doc_id)
            written += 1

print(f"Done. Relationships written: {written}")
driver.close()