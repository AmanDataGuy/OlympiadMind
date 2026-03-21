"""
agent.py
--------
Phase 5.2 — LangGraph Physics Problem Solving Agent

State machine that takes a physics problem and returns a step-by-step solution.

Flow:
  classify_topic → extract_concepts → retrieve → generate → validate → END

Uses:
  - Fine-tuned Gemma 2B for concept extraction (Phase 8 — falls back to keyword)
  - Hybrid retriever (RAPTOR + Neo4j + FAISS)
  - Mistral 7B via transformers for solution generation (local)
"""

import json
import os
import re
import sys
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()


# ── state definition ──────────────────────────────────────────────────────────

class QueryState(TypedDict):
    # input
    problem_text:   str
    problem_source: Optional[str]   # olympiad / science_bowl / cbse etc

    # intermediate
    topic:          Optional[str]
    concepts:       Optional[list]
    retrieval:      Optional[dict]

    # output
    solution:       Optional[str]
    confidence:     Optional[float]
    citations:      Optional[list]

    # control
    retry_count:    int
    error:          Optional[str]


# ── topic classifier ──────────────────────────────────────────────────────────

PHYSICS_TOPICS = [
    "Mechanics",
    "Electromagnetism",
    "Thermodynamics",
    "Optics",
    "Modern Physics",
    "Waves",
]

TOPIC_KEYWORDS = {
    "Mechanics":        ["force", "mass", "velocity", "acceleration", "momentum", "energy", "gravity", "friction", "torque", "rotation"],
    "Electromagnetism": ["electric", "magnetic", "charge", "current", "voltage", "field", "capacitor", "inductor", "circuit", "flux"],
    "Thermodynamics":   ["heat", "temperature", "entropy", "pressure", "gas", "carnot", "thermodynamic", "calorimeter", "latent"],
    "Optics":           ["light", "lens", "mirror", "refraction", "diffraction", "interference", "wavelength", "prism", "optical"],
    "Modern Physics":   ["quantum", "photon", "relativity", "nuclear", "radioactive", "electron", "atom", "bohr", "de broglie"],
    "Waves":            ["wave", "frequency", "amplitude", "oscillation", "harmonic", "doppler", "standing", "resonance"],
}


def classify_topic(state: QueryState) -> QueryState:
    """
    Classify problem into one of 6 physics topics using keyword matching.
    Falls back to 'Mechanics' if no clear match.
    """
    text   = state["problem_text"].lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in text)

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        best_topic = "Mechanics"

    return {**state, "topic": best_topic}


# ── concept extractor ─────────────────────────────────────────────────────────

def extract_concepts(state: QueryState) -> QueryState:
    """
    Extract physics concepts from the problem text.
    Uses fine-tuned Gemma if available, otherwise keyword extraction.
    """
    text = state["problem_text"]

    # try fine-tuned Gemma first (Phase 8 — not yet available)
    # falls back to keyword-based extraction
    concepts = keyword_extract_concepts(text, state.get("topic", "Mechanics"))

    return {**state, "concepts": concepts}


def keyword_extract_concepts(text: str, topic: str) -> list:
    """
    Extract physics concepts using keyword matching as fallback.
    Returns list of concept strings.
    """
    text_lower = text.lower()
    concepts   = []

    concept_patterns = {
        "conservation of energy":        ["energy", "conserv"],
        "conservation of momentum":       ["momentum", "conserv"],
        "conservation of angular momentum": ["angular momentum"],
        "lorentz force":                  ["magnetic field", "charge", "velocity"],
        "circular motion":                ["circular", "radius", "centripetal"],
        "simple harmonic motion":         ["oscillat", "harmonic", "period"],
        "gauss's law":                    ["electric flux", "gauss"],
        "faraday's law":                  ["induction", "faraday", "emf"],
        "snell's law":                    ["refraction", "snell", "refractive"],
        "special relativity":             ["relativit", "speed of light", "lorentz"],
        "ideal gas law":                  ["ideal gas", "pv=nrt", "pressure", "volume", "temperature"],
        "newton's laws":                  ["force", "acceleration", "mass"],
        "work-energy theorem":            ["work", "kinetic energy"],
        "ohm's law":                      ["resistance", "current", "voltage"],
        "doppler effect":                 ["doppler", "frequency shift"],
    }

    for concept, patterns in concept_patterns.items():
        if all(p in text_lower for p in patterns[:1]):
            concepts.append(concept)

    if not concepts:
        concepts = [TOPIC_KEYWORDS[topic][0] if topic in TOPIC_KEYWORDS else "physics"]

    return concepts[:5]


# ── retrieval node ────────────────────────────────────────────────────────────

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        import sys
        sys.path.insert(0, ".")
        from src.agent.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


def retrieve(state: QueryState) -> QueryState:
    """Retrieve relevant context from RAPTOR + Neo4j + FAISS."""
    retriever = get_retriever()
    results   = retriever.retrieve(
        query    = state["problem_text"],
        concepts = state.get("concepts", []),
    )
    return {**state, "retrieval": results}


# ── solution generator ────────────────────────────────────────────────────────

SOLUTION_PROMPT = """You are an expert physics tutor helping solve olympiad-level physics problems.

RETRIEVED CONTEXT:
{context}

PROBLEM TO SOLVE:
{problem}

Provide a clear step-by-step solution:
1. Identify the key physics concepts
2. Set up the relevant equations
3. Solve step by step showing all algebra
4. State the final answer clearly

Be precise and show all working. If the problem involves equations, write them clearly."""

def generate_solution(state: QueryState) -> QueryState:
    """
    Generate solution using Mistral 7B on Modal.
    Falls back to structured response if Modal fails.
    """
    context = state.get("retrieval", {}).get("combined_context", "")
    problem = state["problem_text"]
    prompt  = SOLUTION_PROMPT.format(context=context[:2000], problem=problem)

    try:
        solution, confidence = generate_with_modal(prompt)
    except Exception as e:
        print(f"Modal generation failed: {e}")
        solution   = generate_structured_response(state)
        confidence = 0.5

    citations = extract_citations(state.get("retrieval", {}))

    return {
        **state,
        "solution":   solution,
        "confidence": confidence,
        "citations":  citations,
    }


def generate_with_modal(prompt: str) -> tuple:
    import subprocess
    import tempfile
    import os

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(prompt)
        prompt_path = f.name

    result = subprocess.run(
        ["modal", "run", "src/agent/modal_solver.py", "--prompt-path", prompt_path],
        capture_output=True, timeout=180,
        encoding="utf-8", errors="replace",
        env=env
    )

    if result.returncode != 0:
        raise Exception(result.stderr[:200])

    solution   = result.stdout.strip()
    confidence = 0.75 if len(solution) > 100 else 0.4
    return solution, confidence


    def run_mistral(prompt: str) -> str:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        formatted = f"[INST] {prompt} [/INST]"
        inputs    = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    app = modal.App("olympiadmind-solver")
    with app.run():
        solution = run_mistral.remote(prompt)

    confidence = 0.75 if len(solution) > 100 else 0.4
    return solution, confidence

def generate_with_local_model(prompt: str) -> tuple:
    """
    Generate solution using local Mistral or Gemma model.
    Returns (solution_text, confidence_score).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    formatted = f"[INST] {prompt} [/INST]"
    inputs    = tokenizer(formatted, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = 500,
            do_sample      = False,
            temperature    = 0.1,
        )

    generated  = output[0][inputs["input_ids"].shape[1]:]
    solution   = tokenizer.decode(generated, skip_special_tokens=True).strip()
    confidence = 0.75 if len(solution) > 100 else 0.4

    return solution, confidence


def generate_structured_response(state: QueryState) -> str:
    """Fallback structured response using retrieved context."""
    topic    = state.get("topic", "Physics")
    concepts = state.get("concepts", [])
    retrieval = state.get("retrieval", {})

    lines = [
        f"**Topic:** {topic}",
        f"**Key Concepts:** {', '.join(concepts)}",
        "",
        "**Approach:**",
        "Based on the retrieved similar problems and physics knowledge:",
    ]

    neo4j = retrieval.get("neo4j_results", [])
    if neo4j:
        lines.append(f"\nSimilar problem found: {neo4j[0].get('text', '')[:200]}")

    corpus = retrieval.get("corpus_results", [])
    if corpus:
        lines.append(f"\nRelevant knowledge: {corpus[0].get('text', '')[:200]}")

    lines.append("\n**Note:** Full solution requires local model. Context retrieved successfully.")
    return "\n".join(lines)


def extract_citations(retrieval: dict) -> list:
    """Extract citation strings from retrieval results."""
    citations = []

    for r in retrieval.get("raptor_results", [])[:2]:
        doc_id = r.get("doc_id", "")
        if doc_id:
            citations.append(f"Problem: {doc_id}")

    for r in retrieval.get("neo4j_results", [])[:2]:
        doc_id = r.get("doc_id", "")
        comp   = r.get("competition", "")
        year   = r.get("year", "")
        if doc_id:
            citations.append(f"Similar: {comp} {year} ({doc_id})")

    for r in retrieval.get("corpus_results", [])[:2]:
        source = r.get("source", "")
        topic  = r.get("topic", r.get("title", ""))[:50]
        citations.append(f"Reference: [{source}] {topic}")

    return citations


# ── validator ─────────────────────────────────────────────────────────────────

def validate_solution(state: QueryState) -> QueryState:
    """
    Basic validation — check solution has content and confidence is reasonable.
    Routes to fallback if confidence < 0.5.
    """
    solution   = state.get("solution", "")
    confidence = state.get("confidence", 0.0)

    if not solution or len(solution) < 50:
        confidence = 0.3

    return {**state, "confidence": confidence}


def should_retry(state: QueryState) -> str:
    """Conditional edge — retry if confidence too low."""
    if state.get("confidence", 1.0) < 0.4 and state.get("retry_count", 0) < 1:
        return "retry"
    return "end"


# ── graph assembly ────────────────────────────────────────────────────────────

def build_agent():
    """Build and compile the LangGraph state machine."""
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(QueryState)

    workflow.add_node("classify_topic",    classify_topic)
    workflow.add_node("extract_concepts",  extract_concepts)
    workflow.add_node("retrieve",          retrieve)
    workflow.add_node("generate_solution", generate_solution)
    workflow.add_node("validate",          validate_solution)

    workflow.set_entry_point("classify_topic")
    workflow.add_edge("classify_topic",   "extract_concepts")
    workflow.add_edge("extract_concepts", "retrieve")
    workflow.add_edge("retrieve",         "generate_solution")
    workflow.add_edge("generate_solution","validate")

    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "generate_solution",
            "end":   END,
        }
    )

    return workflow.compile()


# ── main ──────────────────────────────────────────────────────────────────────

def solve(problem_text: str) -> dict:
    """
    Main entry point — solve a physics problem.
    Returns dict with solution, confidence, citations, topic, concepts.
    """
    agent = build_agent()

    initial_state = QueryState(
        problem_text   = problem_text,
        problem_source = None,
        topic          = None,
        concepts       = None,
        retrieval      = None,
        solution       = None,
        confidence     = None,
        citations      = None,
        retry_count    = 0,
        error          = None,
    )

    final_state = agent.invoke(initial_state)

    return {
        "topic":      final_state.get("topic"),
        "concepts":   final_state.get("concepts"),
        "solution":   final_state.get("solution"),
        "confidence": final_state.get("confidence"),
        "citations":  final_state.get("citations"),
    }


if __name__ == "__main__":
    problem = """A charged particle of mass m and charge q enters a uniform magnetic field B 
    perpendicular to the field direction with velocity v. Find the radius of the circular 
    path and the time period of revolution."""

    print("Solving problem...\n")
    result = solve(problem)

    print(f"Topic    : {result['topic']}")
    print(f"Concepts : {result['concepts']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nCitations:")
    for c in result.get("citations", []):
        print(f"  {c}")
    print(f"\nSolution:\n{result['solution']}")