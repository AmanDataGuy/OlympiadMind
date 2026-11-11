"""
merger.py
---------
Document Merger

Combines all extraction stream outputs into a single PhysicsDocument
Pydantic schema per PDF. Reads from:

  data/extracted/text/        — typeset text
  data/extracted/equations/   — LaTeX equations
  data/extracted/figures/     — figure descriptions
  data/extracted/scanned/     — scanned page OCR
  data/extracted/hindi/       — Hindi extraction
  data/extracted/structured/  — structured format parsing

Output: data/extracted/documents.jsonl — one PhysicsDocument per line
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH = Path("data/manifests/inventory.csv")
OUTPUT_FILE    = Path("data/extracted/documents.jsonl")

EXTRACTED = {
    "text":       Path("data/extracted/text"),
    "equations":  Path("data/extracted/equations"),
    "figures":    Path("data/extracted/figures"),
    "scanned":    Path("data/extracted/scanned"),
    "hindi":      Path("data/extracted/hindi"),
    "structured": Path("data/extracted/structured"),
}


# ── pydantic schemas ──────────────────────────────────────────────────────────

class Equation(BaseModel):
    latex:   str
    source:  str   # "nougat" | "pix2tex"
    page:    str


class Figure(BaseModel):
    figure_id:     str
    page:          str
    image_path:    str
    diagram_type:  str
    description:   Optional[str]  = None
    physics_concept: Optional[str] = None


class PageText(BaseModel):
    page:       str
    text:       str
    language:   Optional[str] = None
    layout:     Optional[str] = None
    strategy:   Optional[str] = None  # pymupdf_direct | easyocr_hindi


class PhysicsDocument(BaseModel):
    doc_id:      str
    source:      str
    doc_type:    str
    competition: Optional[str] = None
    year:        Optional[str] = None

    # text content
    pages:       list[PageText]     = []

    # equations
    equations:   list[Equation]     = []

    # figures
    figures:     list[Figure]       = []

    # structured content (MCQ, assertion-reason, constants)
    structured:  Optional[dict]     = None

    # extraction metadata
    extraction_sources: dict        = {}
    total_pages:        int         = 0
    equation_count:     int         = 0
    figure_count:       int         = 0


# ── loaders ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Optional[dict]:
    """Load a JSON file safely — returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_text(doc_id: str) -> list[PageText]:
    """Load typeset text extraction output."""
    data = load_json(EXTRACTED["text"] / f"{doc_id}_text.json")
    if not data:
        # try hindi extraction
        data = load_json(EXTRACTED["hindi"] / f"{doc_id}_hindi.json")

    if not data:
        # try scanned extraction
        data = load_json(EXTRACTED["scanned"] / f"{doc_id}_scanned.json")

    if not data:
        return []

    pages = []
    for page_num, page_data in data.get("pages", {}).items():
        text = page_data.get("text") or page_data.get("raw_text", "")
        if text.strip():
            pages.append(PageText(
                page     = page_num,
                text     = text,
                language = page_data.get("language"),
                layout   = page_data.get("layout"),
                strategy = page_data.get("strategy"),
            ))
    return pages


def load_equations(doc_id: str) -> list[Equation]:
    """Load equation extraction output."""
    data = load_json(EXTRACTED["equations"] / f"{doc_id}_equations.json")
    if not data:
        return []

    equations = []
    for page_num, page_data in data.get("pages", {}).items():
        source = page_data.get("source", "unknown")

        for latex in page_data.get("equations_inline", []):
            if latex.strip():
                equations.append(Equation(
                    latex  = latex,
                    source = source,
                    page   = page_num,
                ))

        for latex in page_data.get("equations_display", []):
            if latex.strip():
                equations.append(Equation(
                    latex  = latex,
                    source = source,
                    page   = page_num,
                ))

    return equations


def load_figures(doc_id: str) -> list[Figure]:
    """Load figure extraction output."""
    data = load_json(EXTRACTED["figures"] / f"{doc_id}_figures.json")
    if not data:
        return []

    figures = []
    for fig in data.get("figures", []):
        figures.append(Figure(
            figure_id      = fig.get("figure_id", ""),
            page           = fig.get("page", ""),
            image_path     = fig.get("image_path", ""),
            diagram_type   = fig.get("diagram_type", ""),
            description    = fig.get("description"),
            physics_concept = fig.get("physics_concept"),
        ))
    return figures


def load_structured(doc_id: str) -> Optional[dict]:
    """Load structured format parsing output."""
    data = load_json(EXTRACTED["structured"] / f"{doc_id}_structured.json")
    return data


# ── merger ────────────────────────────────────────────────────────────────────

def merge_document(row: pd.Series) -> Optional[PhysicsDocument]:
    """
    Merge all extraction streams for one document into a PhysicsDocument.
    Returns None if no content was extracted from any stream.
    """
    doc_id   = row["filename"].replace(".pdf", "")
    source   = row["source"]
    doc_type = row["doc_type"]

    pages      = load_text(doc_id)
    equations  = load_equations(doc_id)
    figures    = load_figures(doc_id)
    structured = load_structured(doc_id)

    # skip completely empty documents
    if not pages and not equations and not figures and not structured:
        return None

    # track which extraction streams contributed
    extraction_sources = {
        "text":       len(pages) > 0,
        "equations":  len(equations) > 0,
        "figures":    len(figures) > 0,
        "structured": structured is not None,
    }

    return PhysicsDocument(
        doc_id      = doc_id,
        source      = source,
        doc_type    = doc_type,
        competition = str(row.get("competition", "")) if pd.notna(row.get("competition")) else None,
        year        = str(row.get("year", ""))        if pd.notna(row.get("year"))        else None,
        pages       = pages,
        equations   = equations,
        figures     = figures,
        structured  = structured,
        extraction_sources = extraction_sources,
        total_pages    = int(row.get("page_count", 0)),
        equation_count = len(equations),
        figure_count   = len(figures),
    )


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INVENTORY_PATH)
    print(f"Merging {len(df)} documents...\n")

    merged   = 0
    skipped  = 0
    failed   = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging documents"):
            try:
                doc = merge_document(row)
            except Exception as e:
                print(f"\n  ERROR — {row['filename']}: {e}")
                failed += 1
                continue

            if doc is None:
                skipped += 1
                continue

            out.write(doc.model_dump_json() + "\n")
            merged += 1

    print(f"\nDone.")
    print(f"  Merged  : {merged}")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {OUTPUT_FILE}")

    # quick stats
    print(f"\nRunning stats on output...")
    total_eq  = 0
    total_fig = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            total_eq  += doc.get("equation_count", 0)
            total_fig += doc.get("figure_count", 0)

    print(f"  Total equations : {total_eq}")
    print(f"  Total figures   : {total_fig}")


if __name__ == "__main__":
    main()