"""
generate_ground_truth.py
------------------------
Randomly picks 50 PDFs from the dataset, stratified by source type,
and writes a blank ground_truth.jsonl template to data/evaluation/.

Run this once, then manually fill in the template for each document.
The filled file is used to compute CER, BLEU, and WER.
"""

import json
import random
import pandas as pd
from pathlib import Path


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH = Path("data/manifests/inventory.csv")
OUTPUT_DIR     = Path("data/evaluation")
OUTPUT_FILE    = OUTPUT_DIR / "ground_truth.jsonl"
RANDOM_SEED    = 42

# How many docs to sample from each source.
# Olympiad gets the most because it is the core dataset.
SAMPLE_DISTRIBUTION = {
    "olympiad":     38,
    "science_bowl":  2,
    "cbse":          2,
    "hindi":         2,
    "multilingual":  1,
    "scanned":       1,
    "tabular":       1,
    "other":         3,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def infer_source(filename: str) -> str:
    """Infer source from filename when inventory has no source column."""
    f = filename.lower()
    if any(x in f for x in ["ipho", "apho", "eupho", "nbpho", "rmph", "wopho"]):
        return "olympiad"
    if "sciencebowl" in f:
        return "science_bowl"
    if "cbse" in f:
        return "cbse"
    if "rbse" in f or "hi_" in f:
        return "hindi"
    if "maha" in f or "_mr_" in f:
        return "multilingual"
    if "kumwenda" in f or "scanned" in f:
        return "scanned"
    if "nist" in f or "table" in f:
        return "tabular"
    return "other"


def blank_record(row: pd.Series) -> dict:
    """
    Returns one blank template record for a document.
    Every TODO field gets filled in manually after running this script.
    """
    return {
        "doc_id": row["filename"].replace(".pdf", ""),
        "source": row.get("source", infer_source(row["filename"])),
        "pages":  int(row["page_count"]),

        # Open the PDF and type out exactly what you see on page 1.
        # Keep punctuation, spacing, and line breaks as accurate as possible.
        "typeset_text_sample": "TODO",

        # Write 5 equations from this document in LaTeX.
        # Example: "\\frac{mv^2}{r} = qvB"
        "equations": [
            "TODO_eq1",
            "TODO_eq2",
            "TODO_eq3",
            "TODO_eq4",
            "TODO_eq5",
        ],

        # One sentence per figure describing what physics concept it illustrates.
        "figure_descriptions": [
            "TODO_fig1",
            "TODO_fig2",
            "TODO_fig3",
        ],

        # Only fill this if the doc is scanned (avg_chars_per_page < 50).
        # Leave as null for born-digital documents.
        "scanned_text_sample": None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    df = pd.read_csv(INVENTORY_PATH)
    print(f"Loaded inventory: {len(df)} documents\n")

    # attach source column if inventory does not already have one
    if "source" not in df.columns:
        df["source"] = df["filename"].apply(infer_source)

    # stratified sampling by source
    selected = []

    for source, n in SAMPLE_DISTRIBUTION.items():
        pool = df[df["source"] == source]

        if len(pool) == 0:
            print(f"  WARNING: no documents found for source '{source}', skipping")
            continue

        n_actual = min(n, len(pool))
        sampled  = pool.sample(n=n_actual, random_state=RANDOM_SEED)
        selected.append(sampled)
        print(f"  {source:<15} — sampled {n_actual} of {len(pool)}")

    selected_df = pd.concat(selected).reset_index(drop=True)
    print(f"\nTotal selected: {len(selected_df)} documents")

    # write blank template
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, row in selected_df.iterrows():
            record = blank_record(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nBlank template written to: {OUTPUT_FILE}")
    print("\nNext steps:")
    print("  1. Open each PDF listed below")
    print("  2. Fill in the TODO fields in ground_truth.jsonl")
    print("  3. ~5-6 minutes per document, ~4-5 hours total")
    print("  4. This file is your measuring stick for all extraction quality metrics\n")

    # print the full list so you know exactly what to open
    print("─" * 65)
    print("Documents to annotate:\n")
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        source = row.get("source", infer_source(row["filename"]))
        pages  = int(row["page_count"])
        print(f"  {i:>2}. [{source:<15}]  {row['filename']}  ({pages}pp)")


if __name__ == "__main__":
    main()