"""
retriever.py
------------
Phase 5 — Hybrid Retriever

Combines three retrieval sources for maximum context quality:

  1. RAPTOR  — hierarchical tree search over physics problems
               searches all levels simultaneously, returns most relevant
               nodes at the right abstraction level

  2. Neo4j   — concept-based graph retrieval
               finds problems sharing the most physics concepts
               with the query

  3. FAISS   — external knowledge corpus search
               Wikipedia, Stack Exchange, arXiv chunks

Each source is queried independently then results are merged and
deduplicated before being passed to the generation step.
"""

import json
import os
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


# ── config ────────────────────────────────────────────────────────────────────

RAPTOR_FAISS_PATH  = Path("data/raptor/raptor.faiss")
RAPTOR_NODES_PATH  = Path("data/raptor/nodes.jsonl")
CORPUS_FAISS_PATH  = Path("data/corpus/physics.faiss")
CORPUS_CHUNKS_PATH = Path("data/corpus/chunks.jsonl")

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384

# number of results from each source
RAPTOR_TOP_K  = 5
NEO4J_TOP_K   = 3
CORPUS_TOP_K  = 5


# ── singleton model loader ────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ── raptor retriever ──────────────────────────────────────────────────────────

class RaptorRetriever:
    """
    Searches the RAPTOR tree FAISS index.
    Returns nodes from all levels — the tree structure means higher-level
    nodes provide broad context while leaf nodes provide specific problems.
    """

    def __init__(self):
        self.index = faiss.read_index(str(RAPTOR_FAISS_PATH))
        self.nodes = [
            json.loads(l)
            for l in RAPTOR_NODES_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        print(f"RAPTOR index loaded: {self.index.ntotal} nodes")

    def search(self, query: str, top_k: int = RAPTOR_TOP_K) -> list:
        model      = get_embedding_model()
        embedding  = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embedding)

        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            node = self.nodes[idx].copy()
            node["score"]  = float(dist)
            node["source_type"] = "raptor"
            results.append(node)

        return results


# ── neo4j retriever ───────────────────────────────────────────────────────────

class Neo4jRetriever:
    """
    Searches Neo4j for problems sharing concepts with the query.
    Extracts concept keywords from the query text and matches against
    Concept nodes in the graph.
    """

    def __init__(self):
        load_dotenv()
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        print("Neo4j retriever connected")

    def search(self, query: str, concepts: list = None, top_k: int = NEO4J_TOP_K) -> list:
        """
        Find problems that share the most concepts with the query.
        If concepts list is provided, use those directly.
        Otherwise falls back to keyword matching against concept names.
        """
        with self.driver.session() as session:

            if concepts:
                # search by extracted concepts
                result = session.run("""
                    MATCH (p:Problem)-[:REQUIRES]->(c:Concept)
                    WHERE c.canonical_name IN $concepts
                    WITH p, COUNT(c) AS relevance
                    ORDER BY relevance DESC
                    LIMIT $top_k
                    RETURN p.id AS doc_id,
                           p.problem_text AS text,
                           p.source AS source,
                           p.competition AS competition,
                           p.year AS year,
                           relevance
                """, concepts=[c.lower() for c in concepts], top_k=top_k)
            else:
                # fallback: keyword search in problem text
                result = session.run("""
                    MATCH (p:Problem)
                    WHERE toLower(p.problem_text) CONTAINS toLower($query)
                    RETURN p.id AS doc_id,
                           p.problem_text AS text,
                           p.source AS source,
                           p.competition AS competition,
                           p.year AS year,
                           1 AS relevance
                    LIMIT $top_k
                """, query=query[:50], top_k=top_k)

            results = []
            for record in result:
                results.append({
                    "doc_id":      record["doc_id"],
                    "text":        record["text"] or "",
                    "source":      record["source"],
                    "competition": record["competition"],
                    "year":        record["year"],
                    "relevance":   record["relevance"],
                    "source_type": "neo4j",
                })

            return results

    def get_similar_problems(self, doc_id: str, top_k: int = 3) -> list:
        """Get problems connected via SIMILAR_TO edges."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Problem {id: $doc_id})-[r:SIMILAR_TO]-(similar:Problem)
                RETURN similar.id AS doc_id,
                       similar.problem_text AS text,
                       similar.source AS source,
                       r.shared_concepts AS shared
                ORDER BY r.shared_concepts DESC
                LIMIT $top_k
            """, doc_id=doc_id, top_k=top_k)

            return [dict(r) for r in result]

    def close(self):
        self.driver.close()


# ── faiss corpus retriever ────────────────────────────────────────────────────

class CorpusRetriever:
    """
    Searches the external knowledge corpus FAISS index.
    Wikipedia + Stack Exchange + arXiv chunks.
    """

    def __init__(self):
        self.index  = faiss.read_index(str(CORPUS_FAISS_PATH))
        self.chunks = [
            json.loads(l)
            for l in CORPUS_CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        print(f"Corpus index loaded: {self.index.ntotal} chunks")

    def search(self, query: str, top_k: int = CORPUS_TOP_K) -> list:
        model     = get_embedding_model()
        embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embedding)

        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"]       = float(dist)
            chunk["source_type"] = "corpus"
            results.append(chunk)

        return results


# ── hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines RAPTOR + Neo4j + FAISS retrieval.
    Each source is queried independently and results are merged.
    """

    def __init__(self):
        print("Initializing hybrid retriever...")
        self.raptor = RaptorRetriever()
        self.neo4j  = Neo4jRetriever()
        self.corpus = CorpusRetriever()
        print("Hybrid retriever ready.\n")

    def retrieve(
        self,
        query: str,
        concepts: list = None,
        raptor_k: int = RAPTOR_TOP_K,
        neo4j_k:  int = NEO4J_TOP_K,
        corpus_k: int = CORPUS_TOP_K,
    ) -> dict:
        """
        Run all three retrievers and return combined results.

        Returns:
            {
                "raptor_results":  [...],  # hierarchical problem search
                "neo4j_results":   [...],  # concept-based graph search
                "corpus_results":  [...],  # external knowledge chunks
                "combined_context": str,   # formatted string for LLM
            }
        """
        raptor_results = self.raptor.search(query, top_k=raptor_k)
        neo4j_results  = self.neo4j.search(query, concepts=concepts, top_k=neo4j_k)
        corpus_results = self.corpus.search(query, top_k=corpus_k)

        combined_context = self._format_context(
            raptor_results, neo4j_results, corpus_results
        )

        return {
            "raptor_results":   raptor_results,
            "neo4j_results":    neo4j_results,
            "corpus_results":   corpus_results,
            "combined_context": combined_context,
        }

    def _format_context(
        self,
        raptor_results: list,
        neo4j_results:  list,
        corpus_results: list,
    ) -> str:
        """Format all retrieved content into a single context string for the LLM."""
        sections = []

        if raptor_results:
            sections.append("=== SIMILAR PAST PROBLEMS ===")
            for i, r in enumerate(raptor_results, 1):
                level = r.get("level", 0)
                tag   = f"[Level {level} — {'Specific problem' if level == 0 else 'Topic summary'}]"
                sections.append(f"{i}. {tag}\n{r['text'][:300]}")

        if neo4j_results:
            sections.append("\n=== PROBLEMS WITH MATCHING CONCEPTS ===")
            for i, r in enumerate(neo4j_results, 1):
                comp = r.get("competition", "")
                year = r.get("year", "")
                sections.append(f"{i}. [{comp} {year}]\n{r['text'][:300]}")

        if corpus_results:
            sections.append("\n=== RELEVANT PHYSICS KNOWLEDGE ===")
            for i, r in enumerate(corpus_results, 1):
                source = r.get("source", "")
                topic  = r.get("topic", r.get("title", ""))
                sections.append(f"{i}. [{source}] {topic}\n{r['text'][:300]}")

        return "\n\n".join(sections)

    def close(self):
        self.neo4j.close()


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = HybridRetriever()

    query    = "A charged particle moves in a uniform magnetic field in a circular path"
    concepts = ["lorentz force", "circular motion", "centripetal acceleration"]

    print(f"Query: {query}\n")
    results = retriever.retrieve(query, concepts=concepts)

    print("RAPTOR results:")
    for r in results["raptor_results"]:
        print(f"  [{r.get('level','?')}] score={r['score']:.3f} — {r['text'][:80]}")

    print("\nNeo4j results:")
    for r in results["neo4j_results"]:
        print(f"  relevance={r['relevance']} — {r['text'][:80]}")

    print("\nCorpus results:")
    for r in results["corpus_results"]:
        print(f"  score={r['score']:.3f} [{r['source']}] — {r['text'][:80]}")

    retriever.close()