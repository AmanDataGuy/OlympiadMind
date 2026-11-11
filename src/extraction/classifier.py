# src/extraction/classifier.py
# ==========================
# PURPOSE : For every page in every PDF, detect what type of content it has
# USES    : PyMuPDF (fitz) to read pages
# OUTPUT  : data/manifests/page_classification.json
# RUN     : python src/extraction/classifier.py

import fitz
import json
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUTPUT  = Path("data/manifests/page_classification.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Greek characters used in physics equations ────────────────────────────────

GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")

# ── Page Classifier ───────────────────────────────────────────────────────────

def classify_page(page) -> list:
    """Look at one page and return a list of content types found on it"""
    types = []

    blocks     = page.get_text("dict")["blocks"]
    text_chars = sum(
    len(span.get("text", ""))
    for b in blocks if b["type"] == 0
    for line in b.get("lines", [])
    for span in line.get("spans", [])
    )
    has_images = len(page.get_images()) > 0

    # Typeset text — PyMuPDF found real selectable text
    if text_chars > 100:
        types.append("typeset_text")

    # Figure — page has at least one image block
    if has_images:
        types.append("figure")

    # Scanned — page renders visually but PyMuPDF got almost no text
    if text_chars < 30:
        pix         = page.get_pixmap()
        not_blank   = any(b != 255 for b in pix.samples[:500])
        if not_blank:
            types.append("scanned")

    # Equations — count Greek letters across all text spans
    greek_count = 0
    for b in blocks:
        if b["type"] == 0:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    greek_count += sum(1 for c in span.get("text", "") if c in GREEK_CHARS)

    if greek_count > 5:
        types.append("equations")

    # Unknown — nothing detected
    if not types:
        types.append("unknown")

    return types

# ── Main Loop ─────────────────────────────────────────────────────────────────

all_pdfs = sorted(RAW_DIR.rglob("*.pdf"))
results  = {}

for pdf_path in tqdm(all_pdfs, desc="Classifying pages"):
    doc_id = pdf_path.stem
    try:
        doc   = fitz.open(pdf_path)
        pages = {}

        for page_num in range(len(doc)):
            page         = doc[page_num]
            content_types = classify_page(page)
            pages[page_num] = {"content_types": content_types}

        doc.close()
        results[doc_id] = {
            "filename" : pdf_path.name,
            "source"   : pdf_path.parts[pdf_path.parts.index("raw") + 1],
            "pages"    : pages
        }

    except Exception as e:
        print(f"\n  ERROR: {pdf_path.name} — {e}")
        results[doc_id] = {"filename": pdf_path.name, "error": str(e)}

# ── Write Output ──────────────────────────────────────────────────────────────

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────

type_counts = {}
unknown_pages = 0
total_pages   = 0

for doc in results.values():
    for page in doc.get("pages", {}).values():
        total_pages += 1
        for t in page["content_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1
        if page["content_types"] == ["unknown"]:
            unknown_pages += 1

print(f"\nTotal pages classified : {total_pages}")
print(f"Unknown pages          : {unknown_pages} ({round(100*unknown_pages/max(total_pages,1), 1)}%)")
print(f"\nContent type breakdown:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:<20} {c}")