"""
main.py
-------
Phase 7.1 — FastAPI Backend

REST API wrapping the OlympiadMind agent and Neo4j graph.

Routes:
  POST /v1/solve       — solve a physics problem
  GET  /v1/problems    — search problems by filters
  GET  /v1/similar     — find similar problems via Neo4j
  GET  /v1/health      — pipeline status
  POST /mcp/solve      — MCP-compatible tool endpoint
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title       = "OlympiadMind API",
    description = "Multimodal physics problem solver with RAPTOR + Neo4j + FAISS hybrid retrieval",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── lazy loaders ──────────────────────────────────────────────────────────────

_agent    = None
_driver   = None
_retriever = None


def get_agent():
    global _agent
    if _agent is None:
        import sys
        sys.path.insert(0, ".")
        from src.agent.agent import build_agent
        _agent = build_agent()
    return _agent


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
    return _driver


# ── request / response models ─────────────────────────────────────────────────

class SolveRequest(BaseModel):
    problem_text: str
    source:       Optional[str] = None


class SolveResponse(BaseModel):
    topic:      str
    concepts:   list
    solution:   str
    confidence: float
    citations:  list
    latency_s:  float


class ProblemSearchRequest(BaseModel):
    source:      Optional[str] = None
    competition: Optional[str] = None
    year:        Optional[str] = None
    topic:       Optional[str] = None
    limit:       int           = 10


class MCPToolCall(BaseModel):
    name:      str
    arguments: dict


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
def health():
    """Check pipeline status."""
    return {
        "status":    "ok",
        "version":   "1.0.0",
        "components": {
            "neo4j":   os.getenv("NEO4J_URI") is not None,
            "raptor":  Path("data/raptor/raptor.faiss").exists(),
            "corpus":  Path("data/corpus/physics.faiss").exists(),
        }
    }


@app.post("/v1/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    """
    Solve a physics problem using the full OlympiadMind pipeline.
    RAPTOR + Neo4j + FAISS retrieval → Mistral 7B generation.
    """
    if not request.problem_text.strip():
        raise HTTPException(status_code=400, detail="problem_text cannot be empty")

    import sys
    sys.path.insert(0, ".")
    from src.agent.agent import solve as agent_solve

    start  = time.time()
    result = agent_solve(request.problem_text)
    elapsed = time.time() - start

    return SolveResponse(
        topic      = result.get("topic", "Unknown"),
        concepts   = result.get("concepts", []),
        solution   = result.get("solution", ""),
        confidence = result.get("confidence", 0.0),
        citations  = result.get("citations", []),
        latency_s  = round(elapsed, 2),
    )


@app.post("/v1/problems")
def search_problems(request: ProblemSearchRequest):
    """
    Search problems in Neo4j by source, competition, year, or topic.
    """
    driver = get_driver()

    filters = []
    params  = {"limit": request.limit}

    if request.source:
        filters.append("p.source = $source")
        params["source"] = request.source

    if request.competition:
        filters.append("p.competition = $competition")
        params["competition"] = request.competition

    if request.year:
        filters.append("p.year = $year")
        params["year"] = request.year

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    query = f"""
        MATCH (p:Problem)
        {where}
        RETURN p.id AS doc_id,
               p.source AS source,
               p.competition AS competition,
               p.year AS year,
               p.doc_type AS doc_type,
               p.problem_text AS problem_text,
               p.equation_count AS equation_count,
               p.figure_count AS figure_count
        LIMIT $limit
    """

    with driver.session() as session:
        result  = session.run(query, **params)
        records = [dict(r) for r in result]

    return {"count": len(records), "problems": records}


@app.get("/v1/similar/{doc_id}")
def get_similar(doc_id: str, limit: int = 5):
    """
    Find problems similar to the given doc_id via SIMILAR_TO edges in Neo4j.
    """
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Problem {id: $doc_id})-[r:SIMILAR_TO]-(similar:Problem)
            RETURN similar.id AS doc_id,
                   similar.problem_text AS problem_text,
                   similar.competition AS competition,
                   similar.year AS year,
                   r.shared_concepts AS shared_concepts
            ORDER BY r.shared_concepts DESC
            LIMIT $limit
        """, doc_id=doc_id, limit=limit)

        similar = [dict(r) for r in result]

    return {"doc_id": doc_id, "similar": similar}


@app.get("/v1/concepts/{doc_id}")
def get_concepts(doc_id: str):
    """Get all concepts required by a specific problem."""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Problem {id: $doc_id})-[:REQUIRES]->(c:Concept)
            RETURN c.canonical_name AS concept
        """, doc_id=doc_id)

        concepts = [r["concept"] for r in result]

    return {"doc_id": doc_id, "concepts": concepts}


@app.get("/v1/stats")
def get_stats():
    """Graph statistics from Neo4j."""
    driver = get_driver()

    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count")
        rels  = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count")

        node_stats = {r["label"]: r["count"] for r in nodes}
        rel_stats  = {r["type"]:  r["count"] for r in rels}

    corpus_chunks = sum(
        1 for f in [
            Path("data/corpus/wikipedia_chunks.jsonl"),
            Path("data/corpus/stackexchange_chunks.jsonl"),
            Path("data/corpus/arxiv_chunks.jsonl"),
        ]
        if f.exists()
        for _ in f.read_text(encoding="utf-8").splitlines()
        if _.strip()
    )

    return {
        "graph":  {"nodes": node_stats, "relationships": rel_stats},
        "corpus": {"total_chunks": corpus_chunks},
        "raptor": {"nodes": 1073, "levels": 3},
    }


# ── mcp endpoint ──────────────────────────────────────────────────────────────

@app.post("/mcp/solve")
def mcp_solve(tool_call: MCPToolCall):
    """
    MCP-compatible tool endpoint.
    Allows Claude and other MCP clients to call OlympiadMind directly.

    Expected format:
    {
        "name": "solve_physics_problem",
        "arguments": {"problem_text": "..."}
    }
    """
    if tool_call.name != "solve_physics_problem":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool: {tool_call.name}. Available: solve_physics_problem"
        )

    problem_text = tool_call.arguments.get("problem_text", "")
    if not problem_text:
        raise HTTPException(status_code=400, detail="arguments.problem_text is required")

    import sys
    sys.path.insert(0, ".")
    from src.agent.agent import solve as agent_solve

    result = agent_solve(problem_text)

    # MCP response format
    return {
        "tool_call_id": tool_call.name,
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "topic":      result.get("topic"),
                    "concepts":   result.get("concepts"),
                    "solution":   result.get("solution"),
                    "confidence": result.get("confidence"),
                    "citations":  result.get("citations"),
                }, indent=2)
            }
        ]
    }


# ── mcp tool manifest ─────────────────────────────────────────────────────────

@app.get("/mcp/tools")
def mcp_tools():
    """Return available MCP tools in standard format."""
    return {
        "tools": [
            {
                "name":        "solve_physics_problem",
                "description": "Solve a physics olympiad problem using RAG with RAPTOR hierarchical retrieval, Neo4j concept graph, and FAISS external knowledge corpus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "problem_text": {
                            "type":        "string",
                            "description": "The physics problem to solve"
                        }
                    },
                    "required": ["problem_text"]
                }
            }
        ]
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)