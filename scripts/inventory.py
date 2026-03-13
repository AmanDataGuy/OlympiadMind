# scripts/inventory.py
# =====================
# PURPOSE : Walk all 1120 PDFs and record basic metadata into a CSV
# USES    : PyMuPDF (fitz) to open PDFs, csv to write output
# OUTPUT  : data/manifests/inventory.csv — one row per PDF
# RUN     : python scripts/inventory.py

import fitz  # PyMuPDF
import csv
import re
from pathlib import Path

RAW_DIR  = Path("data/raw")
OUTPUT   = Path("data/manifests/inventory.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_olympiad_filename(stem):
    """Extract competition, year, number from e.g. IPhO_2023_Q2 or APhO_2015_S1"""
    match = re.match(r"^([A-Za-z]+)_(\d{4})_[QS](\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None, None, None

def get_pdf_info(pdf_path):
    """Open a PDF and return page count, image presence, avg chars per page"""
    try:
        doc = fitz.open(pdf_path)
        page_count    = len(doc)
        has_images    = any(len(page.get_images()) > 0 for page in doc)
        total_chars   = sum(len(page.get_text()) for page in doc)
        avg_chars     = round(total_chars / max(page_count, 1), 1)
        doc.close()
        return page_count, has_images, avg_chars
    except Exception as e:
        print(f"  ERROR reading {pdf_path.name}: {e}")
        return 0, False, 0.0

# ── Main Loop ─────────────────────────────────────────────────────────────────

rows = []

for pdf in sorted(RAW_DIR.rglob("*.pdf")):

    parts = pdf.parts  # e.g. ('data', 'raw', 'olympiad', 'problems', 'IPhO_2023_Q2.pdf')

    # Determine source and doc_type from folder name
    if "olympiad" in parts:
        source    = "olympiad"
        doc_type  = "problem" if "problems" in parts else "solution"
        competition, year, qnum = parse_olympiad_filename(pdf.stem)

    elif "science_bowl" in parts:
        source, doc_type = "science_bowl", "mcq"
        competition, year, qnum = None, None, None

    elif "cbse" in parts:
        source, doc_type = "cbse", "assertion_reason"
        competition, year, qnum = None, None, None

    elif "hindi" in parts:
        source, doc_type = "hindi", "mixed"
        competition, year, qnum = None, None, None

    elif "multilingual" in parts:
        source, doc_type = "multilingual", "mcq"
        competition, year, qnum = None, None, None

    elif "tabular" in parts:
        source, doc_type = "tabular", "constants_table"
        competition, year, qnum = None, None, None

    elif "scanned" in parts:
        source, doc_type = "scanned", "qa_scanned"
        competition, year, qnum = None, None, None

    else:
        source, doc_type = "unknown", "unknown"
        competition, year, qnum = None, None, None

    # Get PDF metadata
    page_count, has_images, avg_chars = get_pdf_info(pdf)

    rows.append({
        "filename"         : pdf.name,
        "source"           : source,
        "doc_type"         : doc_type,
        "competition"      : competition,
        "year"             : year,
        "question_number"  : qnum,
        "page_count"       : page_count,
        "file_size_kb"     : round(pdf.stat().st_size / 1024, 1),
        "has_images"       : has_images,
        "avg_chars_per_page": avg_chars,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ── Print Summary ─────────────────────────────────────────────────────────────

print(f"\nDone — {len(rows)} PDFs written to {OUTPUT}\n")

sources = {}
for r in rows:
    sources[r["source"]] = sources.get(r["source"], 0) + 1

print(f"{'Source':<20} {'Count'}")
print("-" * 28)
for s, c in sorted(sources.items()):
    print(f"{s:<20} {c}")