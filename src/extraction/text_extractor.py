"""
text_extractor.py
-----------------
Typeset Text Extraction

Extracts selectable text from born-digital PDFs using PyMuPDF.
Handles two-column layouts, sub-question segmentation, and language detection.
Only processes pages classified as typeset_text in page_classification.json.

Output: data/extracted/text/{doc_id}_text.json
"""

import json
import re
import fitz
import pandas as pd
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH      = Path("data/manifests/inventory.csv")
CLASSIFICATION_PATH = Path("data/manifests/page_classification.json")
OUTPUT_DIR          = Path("data/extracted/text")

SUPPORTED_SOURCES = {"olympiad", "cbse", "science_bowl", "multilingual", "tabular"}

SOURCE_ROOTS = {
    "olympiad":     Path("data/raw/olympiad"),
    "cbse":         Path("data/raw/cbse"),
    "science_bowl": Path("data/raw/science_bowl"),
    "multilingual": Path("data/raw/multilingual"),
    "tabular":      Path("data/raw/tabular"),
    "scanned":      Path("data/raw/scanned"),
    "hindi":        Path("data/raw/hindi"),
}

OLYMPIAD_SUBFOLDERS = {
    "problem":  "problems",
    "solution": "solutions",
}


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_path(row: pd.Series) -> Path:
    """
    Reconstruct the full PDF path from inventory row.
    Olympiad files live in problems/ or solutions/ subfolders.
    All other sources live directly under their source root.
    """
    source   = row["source"]
    filename = row["filename"]
    root     = SOURCE_ROOTS[source]

    if source == "olympiad":
        subfolder = OLYMPIAD_SUBFOLDERS.get(row["doc_type"], "problems")
        return root / subfolder / filename

    return root / filename


# ── language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect language of extracted text.
    Returns 'unknown' if text is too short or detection fails.
    """
    if len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ── column layout detection ───────────────────────────────────────────────────

def is_two_column(blocks: list, page_width: float) -> bool:
    """
    Returns True if the page has a two-column layout.
    Heuristic: text blocks exist clearly on both left and right halves.
    A margin of 10% page width avoids false positives from centered headings.
    """
    mid    = page_width / 2
    margin = page_width * 0.1

    has_left  = any(b[2] < mid - margin for b in blocks if b[6] == 0)
    has_right = any(b[0] > mid + margin for b in blocks if b[6] == 0)

    return has_left and has_right


def extract_two_column(blocks: list, page_width: float) -> str:
    """
    For two-column pages: sort each column independently top-to-bottom,
    then concatenate left column + right column to preserve reading order.
    """
    mid = page_width / 2

    left  = sorted([b for b in blocks if b[6] == 0 and b[2] <= mid],  key=lambda b: b[1])
    right = sorted([b for b in blocks if b[6] == 0 and b[0] >= mid],  key=lambda b: b[1])

    left_text  = "\n".join(b[4].strip() for b in left  if b[4].strip())
    right_text = "\n".join(b[4].strip() for b in right if b[4].strip())

    return left_text + "\n\n" + right_text


# ── sub-question segmentation ─────────────────────────────────────────────────

# matches: a) b) c)  a. b. c.  (a) (b)  i) ii)  1. 2. 3.
SUBQUESTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[a-eA-E][).\]]\s"
    r"|[ivxIVX]+[).\]]\s"
    r"|\([a-eA-E]\)\s"
    r"|\d+[).]\s"
    r")"
)


def segment_subquestions(text: str) -> list:
    """
    Split problem text into labeled sub-questions.
    Returns a list of dicts: [{"label": "a", "text": "..."}, ...]
    If no sub-question markers found, returns the whole text as one block.
    """
    splits = SUBQUESTION_PATTERN.split(text)
    labels = SUBQUESTION_PATTERN.findall(text)

    if not labels:
        return [{"label": "main", "text": text.strip()}]

    segments = []

    # text before first label is the problem stem
    if splits[0].strip():
        segments.append({"label": "stem", "text": splits[0].strip()})

    for label, content in zip(labels, splits[1:]):
        segments.append({
            "label": label.strip().rstrip(").] "),
            "text":  content.strip()
        })

    return segments


# ── core extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path, page_types: dict) -> dict:
    """
    Extract typeset text from all pages classified as typeset_text.

    Args:
        pdf_path:   path to the PDF file
        page_types: dict from page_classification.json —
                    {"0": {"content_types": [...]}, "1": {...}, ...}

    Returns:
        dict keyed by page number string, each value contains:
        raw_text, layout, language, segments, char_count
    """
    doc     = fitz.open(pdf_path)
    results = {}

    for page_num_str, page_info in page_types.items():

        # page_info is {"content_types": [...]}
        content_types = page_info.get("content_types", [])

        if "typeset_text" not in content_types:
            continue

        page_num   = int(page_num_str)
        page       = doc[page_num]
        page_width = page.rect.width

        # extract blocks sorted top-to-bottom, left-to-right
        blocks = page.get_text("blocks", sort=True)

        # route to two-column or single-column extraction
        if is_two_column(blocks, page_width):
            raw_text = extract_two_column(blocks, page_width)
            layout   = "two_column"
        else:
            text_blocks = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
            raw_text    = "\n".join(text_blocks)
            layout      = "single_column"

        if not raw_text.strip():
            continue

        language = detect_language(raw_text)
        segments = segment_subquestions(raw_text)

        results[page_num_str] = {
            "raw_text":   raw_text,
            "layout":     layout,
            "language":   language,
            "segments":   segments,
            "char_count": len(raw_text),
        }

    doc.close()
    return results


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df             = pd.read_csv(INVENTORY_PATH)
    classification = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))

    df = df[df["source"].isin(SUPPORTED_SOURCES)].reset_index(drop=True)
    print(f"Processing {len(df)} documents\n")

    skipped = 0
    failed  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text"):
        doc_id   = row["filename"].replace(".pdf", "")
        pdf_path = resolve_path(row)

        if not pdf_path.exists():
            skipped += 1
            continue

        page_types = classification.get(doc_id, {}).get("pages", {})
        if not page_types:
            skipped += 1
            continue

        try:
            result = extract_text_from_pdf(pdf_path, page_types)
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        out_path = OUTPUT_DIR / f"{doc_id}_text.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":      doc_id,
                "source":      row["source"],
                "doc_type":    row["doc_type"],
                "competition": row.get("competition", ""),
                "year":        row.get("year", ""),
                "pages":       result,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()