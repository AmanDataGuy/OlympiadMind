"""
benchmark.py
------------
Phase 6 — Benchmark: OlympiadMind vs Zero-Shot Mistral

Runs 38 held-out problems through two pipelines:
  1. OlympiadMind — Mistral 7B + RAPTOR + Neo4j + FAISS retrieval
  2. Baseline     — Mistral 7B zero-shot (no retrieval)

Metrics per problem:
  - has_equations  : solution contains LaTeX equations
  - has_steps      : solution has numbered steps
  - has_answer     : solution contains a final answer statement
  - solution_length: character count of solution
  - latency_s      : end-to-end time in seconds

Output: data/evaluation/benchmark_results.json
"""

import json
import time
import subprocess
import tempfile
import os
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = Path("data/evaluation/ground_truth.jsonl")
OUTPUT_PATH       = Path("data/evaluation/benchmark_results.json")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_benchmark_problems() -> list:
    """Load olympiad problems from ground truth as benchmark set."""
    problems = []
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source") == "olympiad":
                text = record.get("typeset_text_sample", "")
                if text and text != "TODO" and len(text) > 50:
                    problems.append({
                        "doc_id":       record["doc_id"],
                        "problem_text": text,
                        "equations_gt": record.get("equations", []),
                    })
    return problems


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate_solution(solution: str, problem: dict) -> dict:
    """
    Evaluate solution quality on structural metrics.
    We don't have gold standard answers so we measure solution structure
    as a proxy for quality.
    """
    if not solution:
        return {
            "has_equations":   False,
            "has_steps":       False,
            "has_answer":      False,
            "solution_length": 0,
        }

    sol_lower = solution.lower()

    return {
        # does solution contain math equations
        "has_equations": any(c in solution for c in ["=", "\\frac", "\\sqrt", "^2", "$$"]),

        # does solution have step-by-step structure
        "has_steps": any(s in sol_lower for s in ["step", "1.", "first", "second", "therefore"]),

        # does solution state a final answer
        "has_answer": any(s in sol_lower for s in ["answer", "result", "therefore", "final", "thus"]),

        # length as proxy for completeness
        "solution_length": len(solution),
    }


# ── olympiadmind pipeline ─────────────────────────────────────────────────────

def run_olympiadmind(problem_text: str) -> tuple:
    """
    Run OlympiadMind full pipeline on a problem.
    Returns (solution, latency_seconds).
    """
    import sys
    sys.path.insert(0, ".")
    from src.agent.agent import solve

    start    = time.time()
    result   = solve(problem_text)
    latency  = time.time() - start
    solution = result.get("solution", "")

    return solution, latency


# ── zero-shot baseline ────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """Solve this physics problem step by step.
Show all working. State the final answer clearly.

Problem: {problem}"""


def run_zero_shot_modal(problem_text: str) -> tuple:
    """
    Run Mistral 7B zero-shot on Modal (no retrieval).
    Returns (solution, latency_seconds).
    """
    prompt = ZERO_SHOT_PROMPT.format(problem=problem_text)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(prompt)
        prompt_path = f.name

    start  = time.time()
    result = subprocess.run(
        ["modal", "run", "src/agent/modal_solver.py", "--prompt-path", prompt_path],
        capture_output=True, timeout=180,
        encoding="utf-8", errors="replace",
        env=env,
    )
    latency = time.time() - start

    if result.returncode != 0:
        return "", latency

    # extract just the solution from Modal output (skip progress lines)
    lines    = result.stdout.strip().split("\n")
    solution = "\n".join(
        l for l in lines
        if not l.startswith(("✓", "├", "└", "Warning", "Fetching", "Loading", "Setting"))
    ).strip()

    return solution, latency


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(problems: list, skip_olympiadmind: bool = False) -> dict:
    """
    Run both pipelines on all problems and collect results.

    skip_olympiadmind: if True only run baseline (useful for testing)
    """
    results = []

    for problem in tqdm(problems, desc="Running benchmark"):
        doc_id       = problem["doc_id"]
        problem_text = problem["problem_text"]

        print(f"\n  [{doc_id}]")

        # run OlympiadMind
        if not skip_olympiadmind:
            try:
                om_solution, om_latency = run_olympiadmind(problem_text)
                om_metrics = evaluate_solution(om_solution, problem)
            except Exception as e:
                print(f"  OlympiadMind failed: {e}")
                om_solution = ""
                om_latency  = 0
                om_metrics  = evaluate_solution("", problem)
        else:
            om_solution = ""
            om_latency  = 0
            om_metrics  = {}

        # run zero-shot baseline
        try:
            zs_solution, zs_latency = run_zero_shot_modal(problem_text)
            zs_metrics = evaluate_solution(zs_solution, problem)
        except Exception as e:
            print(f"  Zero-shot failed: {e}")
            zs_solution = ""
            zs_latency  = 0
            zs_metrics  = evaluate_solution("", problem)

        print(f"  OM: {om_metrics.get('solution_length',0)} chars, {om_latency:.1f}s")
        print(f"  ZS: {zs_metrics.get('solution_length',0)} chars, {zs_latency:.1f}s")

        results.append({
            "doc_id":       doc_id,
            "problem_text": problem_text[:200],
            "olympiadmind": {
                "solution": om_solution[:500],
                "latency":  round(om_latency, 2),
                **om_metrics,
            },
            "zero_shot": {
                "solution": zs_solution[:500],
                "latency":  round(zs_latency, 2),
                **zs_metrics,
            },
        })

    return {"n_problems": len(results), "results": results}


# ── report printer ────────────────────────────────────────────────────────────

def print_report(benchmark: dict) -> None:
    results = benchmark["results"]
    n       = len(results)

    if n == 0:
        print("No results to report.")
        return

    def avg(key, pipeline):
        vals = [r[pipeline].get(key, 0) for r in results]
        return sum(vals) / len(vals) if vals else 0

    def pct(key, pipeline):
        vals = [r[pipeline].get(key, False) for r in results]
        return sum(vals) / len(vals) * 100 if vals else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print(f"OlympiadMind (RAG) vs Zero-Shot Mistral 7B")
    print(f"Problems: {n}")
    print("=" * 60)
    print(f"{'Metric':<25} {'OlympiadMind':>15} {'Zero-Shot':>12}")
    print("-" * 60)
    print(f"{'Has equations (%)' :<25} {pct('has_equations','olympiadmind'):>14.1f}% {pct('has_equations','zero_shot'):>11.1f}%")
    print(f"{'Has steps (%)' :<25} {pct('has_steps','olympiadmind'):>14.1f}% {pct('has_steps','zero_shot'):>11.1f}%")
    print(f"{'Has answer (%)' :<25} {pct('has_answer','olympiadmind'):>14.1f}% {pct('has_answer','zero_shot'):>11.1f}%")
    print(f"{'Avg solution length':<25} {avg('solution_length','olympiadmind'):>14.0f}  {avg('solution_length','zero_shot'):>11.0f}")
    print(f"{'Avg latency (s)':<25} {avg('latency','olympiadmind'):>14.1f}  {avg('latency','zero_shot'):>11.1f}")
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    problems = load_benchmark_problems()
    print(f"Benchmark problems: {len(problems)}")
    print("Running both pipelines on Modal...\n")

    benchmark = run_benchmark(problems)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print_report(benchmark)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()