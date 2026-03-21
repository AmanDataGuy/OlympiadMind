"""
hindi_extractor.py
------------------
Step 2.5 — Hindi and Multilingual Extraction

Two strategies depending on document type:

  Strategy A — PyMuPDF direct extraction:
    Used when PyMuPDF extracts clean Devanagari Unicode text.
    Fast, accurate, no OCR needed.
    Example: cbse_hi_labmanual_classxi.pdf

  Strategy B — EasyOCR Hindi:
    Used when font encoding is broken (garbled text) or page is scanned.
    Renders page to image, runs EasyOCR with lang=['hi', 'en'].
    Example: rbse_hi_pyq_classxii.pdf

Language detection per page decides which strategy to use:
  - If extracted text contains Devanagari Unicode range (0x0900-0x097F) → Strategy A
  - If extracted text is garbled or empty → Strategy B

Output: data/extracted/hindi/{doc_id}_hindi.json
"""

import json
import re
import cv2
import numpy as np
import fitz
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH      = Path("data/manifests/inventory.csv")
CLASSIFICATION_PATH = Path("data/manifests/page_classification.json")
OUTPUT_DIR          = Path("data/extracted/hindi")

HINDI_SOURCES    = {"hindi", "multilingual"}

SOURCE_ROOTS = {
    "hindi":        Path("data/raw/hindi"),
    "multilingual": Path("data/raw/multilingual"),
}

RENDER_DPI = 300

# Devanagari Unicode range — presence means clean Hindi text
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")

# minimum Devanagari character ratio to consider text clean
DEVANAGARI_THRESHOLD = 0.05


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_path(row: pd.Series) -> Path:
    source   = row["source"]
    filename = row["filename"]
    return SOURCE_ROOTS[source] / filename


# ── language detection ────────────────────────────────────────────────────────

def is_clean_devanagari(text: str) -> bool:
    """
    Check if extracted text contains enough Devanagari characters
    to be considered clean Hindi — not broken font encoding.
    """
    if not text or len(text) < 10:
        return False

    devanagari_chars = len(DEVANAGARI_PATTERN.findall(text))
    ratio = devanagari_chars / len(text)
    return ratio >= DEVANAGARI_THRESHOLD


def is_broken_encoding(text: str) -> bool:
    """
    Detect broken font encoding — text extracted but not valid Unicode Hindi.
    Broken encoding typically shows as ASCII chars that look like transliterated text
    e.g. 'ek/;fed' instead of 'माध्यमिक'
    """
    if not text or len(text) < 10:
        return False

    # if text has no Devanagari but has unusual ASCII patterns → broken encoding
    has_devanagari = bool(DEVANAGARI_PATTERN.search(text))
    has_slash_pattern = "/" in text and len(text) > 20

    return not has_devanagari and has_slash_pattern


# ── strategy a — pymupdf direct ───────────────────────────────────────────────

def extract_text_direct(page: fitz.Page) -> str:
    """
    Strategy A: Extract text directly using PyMuPDF.
    Used when font encoding is correct and Devanagari renders properly.
    """
    blocks = page.get_text("blocks", sort=True)
    text_blocks = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
    return "\n".join(text_blocks)


# ── strategy b — easyocr ─────────────────────────────────────────────────────

def load_hindi_ocr():
    """Load EasyOCR with Hindi + English support."""
    import easyocr
    reader = easyocr.Reader(["hi", "en"], gpu=True)
    return reader


def render_page_to_image(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render PDF page to grayscale numpy array."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


def extract_text_ocr(page: fitz.Page, ocr) -> dict:
    """
    Strategy B: Render page and run EasyOCR with Hindi + English.
    Used for broken font encoding or scanned Hindi pages.
    """
    image = render_page_to_image(page)
    rgb   = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    result = ocr.readtext(rgb)

    if not result:
        return {"text": "", "avg_confidence": 0.0}

    blocks      = []
    confidences = []

    for bbox, text, confidence in result:
        blocks.append(text)
        confidences.append(confidence)

    full_text      = "\n".join(blocks)
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return {"text": full_text, "avg_confidence": avg_confidence}


# ── core extraction ───────────────────────────────────────────────────────────

def extract_hindi_doc(pdf_path: Path, page_types: dict, ocr) -> dict:
    """
    Extract text from a Hindi or multilingual document.

    Per page decision:
    - Try PyMuPDF first
    - If text is clean Devanagari → use it (Strategy A)
    - If text is empty or broken encoding → run EasyOCR (Strategy B)
    """
    doc     = fitz.open(pdf_path)
    results = {}

    for page_num_str, page_info in page_types.items():
        page_num = int(page_num_str)
        page     = doc[page_num]

        # try direct extraction first
        direct_text = extract_text_direct(page)

        if is_clean_devanagari(direct_text):
            # strategy A — clean Devanagari text
            results[page_num_str] = {
                "text":      direct_text,
                "strategy":  "pymupdf_direct",
                "language":  "hi",
                "char_count": len(direct_text),
            }

        elif is_broken_encoding(direct_text) or not direct_text.strip():
            # strategy B — broken encoding or empty, run OCR
            ocr_result = extract_text_ocr(page, ocr)

            if ocr_result["text"].strip():
                results[page_num_str] = {
                    "text":           ocr_result["text"],
                    "strategy":       "easyocr_hindi",
                    "avg_confidence": ocr_result["avg_confidence"],
                    "language":       "hi",
                    "char_count":     len(ocr_result["text"]),
                }
        else:
            # english or mixed content — store as-is
            if direct_text.strip():
                results[page_num_str] = {
                    "text":      direct_text,
                    "strategy":  "pymupdf_direct",
                    "language":  "en",
                    "char_count": len(direct_text),
                }

    doc.close()
    return results


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df             = pd.read_csv(INVENTORY_PATH)
    classification = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))

    df = df[df["source"].isin(HINDI_SOURCES)].reset_index(drop=True)
    print(f"Hindi/multilingual documents: {len(df)}")
    print(f"Loading EasyOCR Hindi model...\n")

    ocr = load_hindi_ocr()
    print("Model loaded. Starting extraction...\n")

    skipped = 0
    failed  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Hindi docs"):
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
            result = extract_hindi_doc(pdf_path, page_types, ocr)
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        if not result:
            continue

        # strategy breakdown for reporting
        strategies = {}
        for page_data in result.values():
            s = page_data.get("strategy", "unknown")
            strategies[s] = strategies.get(s, 0) + 1

        out_path = OUTPUT_DIR / f"{doc_id}_hindi.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":     doc_id,
                "source":     row["source"],
                "pages":      result,
                "strategies": strategies,
            }, f, ensure_ascii=False, indent=2)

        print(f"  {doc_id}: {len(result)} pages — {strategies}")

    print(f"\nDone.")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()