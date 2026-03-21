"""
scanned_ocr.py
--------------
Step 2.4 — Scanned Page OCR

Extracts text from pages classified as 'scanned' or 'unknown' in page_classification.json.
These are pages where PyMuPDF found no selectable text — content is a raster image.

Pipeline per page:
  1. Render page to 300 DPI image using PyMuPDF
  2. OpenCV preprocessing — deskew, denoise, binarize
  3. EasyOCR reads the cleaned image
  4. Output text per page

Output: data/extracted/scanned/{doc_id}_scanned.json
"""

import json
import cv2
import numpy as np
import fitz
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH      = Path("data/manifests/inventory.csv")
CLASSIFICATION_PATH = Path("data/manifests/page_classification.json")
OUTPUT_DIR          = Path("data/extracted/scanned")

ALL_SOURCES = {"olympiad", "cbse", "science_bowl", "multilingual", "scanned", "hindi", "tabular"}

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

RENDER_DPI = 300


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_path(row: pd.Series) -> Path:
    source   = row["source"]
    filename = row["filename"]
    root     = SOURCE_ROOTS[source]

    if source == "olympiad":
        subfolder = OLYMPIAD_SUBFOLDERS.get(row["doc_type"], "problems")
        return root / subfolder / filename

    return root / filename


# ── image preprocessing ───────────────────────────────────────────────────────

def render_page_to_image(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render a PDF page to a high-resolution numpy array."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in scanned pages.
    Detects rotation angle and rotates to fix it.
    Scans are often 1-3 degrees off — enough to hurt OCR accuracy.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return image

    h, w  = image.shape
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from scanned images.
    Non-local means denoising preserves text edges better than Gaussian blur.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Convert to clean black-and-white using adaptive thresholding.
    Handles uneven lighting across the page — common in scanned documents.
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: deskew → denoise → binarize."""
    image = deskew(image)
    image = denoise(image)
    image = binarize(image)
    return image


# ── ocr ───────────────────────────────────────────────────────────────────────

def load_ocr():
    """Load EasyOCR — stable on Windows, supports English and Hindi."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True)
    return reader


def run_ocr(ocr, image: np.ndarray) -> dict:
    """
    Run EasyOCR on a preprocessed page image.
    Returns extracted text blocks with confidence scores.
    """
    rgb    = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    result = ocr.readtext(rgb)

    if not result:
        return {"text": "", "blocks": [], "avg_confidence": 0.0}

    blocks      = []
    confidences = []

    for bbox, text, confidence in result:
        blocks.append({
            "text":       text,
            "confidence": round(confidence, 3),
            "bbox":       bbox,
        })
        confidences.append(confidence)

    full_text      = "\n".join(b["text"] for b in blocks)
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return {
        "text":           full_text,
        "blocks":         blocks,
        "avg_confidence": avg_confidence,
    }


# ── core extraction ───────────────────────────────────────────────────────────

def extract_scanned_pages(pdf_path: Path, page_types: dict, ocr) -> dict:
    """
    Extract text from all pages classified as 'scanned' or 'unknown'.

    For each page:
    1. Render to 300 DPI grayscale image
    2. Preprocess: deskew + denoise + binarize
    3. Run EasyOCR
    4. Store text + confidence
    """
    doc     = fitz.open(pdf_path)
    results = {}

    for page_num_str, page_info in page_types.items():
        content_types = page_info.get("content_types", [])

        if "scanned" not in content_types and "unknown" not in content_types:
            continue

        page_num = int(page_num_str)
        page     = doc[page_num]

        raw_image    = render_page_to_image(page, dpi=RENDER_DPI)
        clean_image  = preprocess(raw_image)

        raw_result   = run_ocr(ocr, raw_image)
        clean_result = run_ocr(ocr, clean_image)

        ocr_result = (
            raw_result if raw_result["avg_confidence"] > clean_result["avg_confidence"]
            else clean_result
)

        if not ocr_result["text"].strip():
            continue

        results[page_num_str] = {
            "text":           ocr_result["text"],
            "avg_confidence": ocr_result["avg_confidence"],
            "block_count":    len(ocr_result["blocks"]),
            "char_count":     len(ocr_result["text"]),
        }

    doc.close()
    return results


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df             = pd.read_csv(INVENTORY_PATH)
    classification = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))

    df = df[df["source"].isin(ALL_SOURCES)].reset_index(drop=True)

    # filter to docs with at least one scanned or unknown page
    def has_scanned_pages(filename: str) -> bool:
        doc_id = filename.replace(".pdf", "")
        pages  = classification.get(doc_id, {}).get("pages", {})
        return any(
            "scanned" in p.get("content_types", []) or "unknown" in p.get("content_types", [])
            for p in pages.values()
        )

    df = df[df["filename"].apply(has_scanned_pages)].reset_index(drop=True)
    print(f"Documents with scanned/unknown pages: {len(df)}")
    print(f"Loading EasyOCR...\n")

    ocr = load_ocr()
    print("EasyOCR loaded. Starting extraction...\n")

    skipped = 0
    failed  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="OCR scanned pages"):
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
            result = extract_scanned_pages(pdf_path, page_types, ocr)
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        if not result:
            continue

        out_path = OUTPUT_DIR / f"{doc_id}_scanned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":   doc_id,
                "source":   row["source"],
                "doc_type": row["doc_type"],
                "pages":    result,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()