"""
concept_extractor_modal.py
--------------------------
Concept Extraction using Llama 3.1 8B on Modal

Runs Llama 3.1 8B on Modal A10G GPU to extract physics concepts
from all olympiad problem texts in documents.jsonl.

Output: data/corpus/concept_labels.jsonl
  {"doc_id": "IPhO_2022_Q1", "concepts": ["Lorentz force", "circular motion"]}

These labels:
  1. Get written as Concept nodes in Neo4j with REQUIRES edges
  2. Become training data for Gemma 2B fine-tuning

Run with: modal run src/graph/concept_extractor_modal.py
"""

import json
import modal
from pathlib import Path


# ── modal app ─────────────────────────────────────────────────────────────────

app = modal.App("olympiadmind-concept-extractor")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "huggingface_hub",
    )
)


# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a physics expert. Given a physics problem, extract the specific physics concepts required to solve it.

Rules:
- Be specific: not "mechanics" but "conservation of angular momentum"
- List 2-5 concepts maximum
- Return ONLY a JSON array of strings
- Example: ["conservation of energy", "magnetic flux", "harmonic oscillator"]
"""


def build_prompt(problem_text: str) -> str:
    return f"[INST] {SYSTEM_PROMPT}\n\nExtract the physics concepts from this problem:\n\n{problem_text[:500]} [/INST]"

# ── modal function ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    memory=16384,
)
def extract_concepts_batch(problems: list[dict]) -> list[dict]:
    """
    Extract concepts for a batch of problems.
    Runs on Modal A10G GPU.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import json
    import re

    print(f"Loading Llama 3.1 8B...")
    model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    results = []

    for problem in problems:
        doc_id       = problem["doc_id"]
        problem_text = problem["problem_text"]

        if not problem_text.strip():
            results.append({"doc_id": doc_id, "concepts": []})
            continue

        prompt  = build_prompt(problem_text)
        inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        response  = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # parse JSON array from response
        try:
            clean    = re.sub(r"```(?:json)?|```", "", response).strip()
            concepts = json.loads(clean)
            if not isinstance(concepts, list):
                concepts = []
        except Exception:
            # try to extract array from response if JSON parsing fails
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                try:
                    concepts = json.loads(match.group())
                except Exception:
                    concepts = []
            else:
                concepts = []

        results.append({"doc_id": doc_id, "concepts": concepts})
        print(f"  {doc_id}: {concepts}")

    return results


# ── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    DOCUMENTS_PATH = Path("data/extracted/documents.jsonl")
    OUTPUT_PATH    = Path("data/corpus/concept_labels.jsonl")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # load olympiad problem docs only
    problems = []
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("source") == "olympiad" and doc.get("doc_type") == "problem":
                pages        = doc.get("pages", [])
                problem_text = pages[0]["text"] if pages else ""
                if problem_text.strip():
                    problems.append({
                        "doc_id":       doc["doc_id"],
                        "problem_text": problem_text,
                    })

    print(f"Loaded {len(problems)} olympiad problems\n")

    # split into batches of 50 for Modal
    batch_size = 50
    batches    = [problems[i:i+batch_size] for i in range(0, len(problems), batch_size)]
    print(f"Running {len(batches)} batches on Modal A10G...\n")

    all_results = []
    for i, batch_results in enumerate(extract_concepts_batch.map(batches)):
        all_results.extend(batch_results)
        print(f"Batch {i+1}/{len(batches)} done — {len(batch_results)} concepts extracted")

    # write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Total labels : {len(all_results)}")
    print(f"  Output       : {OUTPUT_PATH}")