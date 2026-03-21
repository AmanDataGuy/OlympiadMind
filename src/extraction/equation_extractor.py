"""
equation_extractor.py
---------------------
Step 2.2 — LaTeX Equation Extraction

Primary:  Nougat — processes full PDF pages end-to-end, outputs markdown with
          inline LaTeX. Best for dense equation pages in born-digital PDFs.

Fallback: pix2tex — takes a cropped image of a single equation and returns
          LaTeX. Used when Nougat output looks garbled or confidence is low.

Ensemble: both models run on the same page when Nougat confidence is low.
          The output with more valid LaTeX tokens wins.

Output: data/extracted/equations/{doc_id}_equations.json
"""

import json
import re
import io
import torch
import fitz
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH      = Path("data/manifests/inventory.csv")
CLASSIFICATION_PATH = Path("data/manifests/page_classification.json")
OUTPUT_DIR          = Path("data/extracted/equations")

SUPPORTED_SOURCES   = {"olympiad", "cbse", "science_bowl", "multilingual"}

SOURCE_ROOTS = {
    "olympiad":     Path("data/raw/olympiad"),
    "cbse":         Path("data/raw/cbse"),
    "science_bowl": Path("data/raw/science_bowl"),
    "multilingual": Path("data/raw/multilingual"),
}

OLYMPIAD_SUBFOLDERS = {
    "problem":  "problems",
    "solution": "solutions",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# nougat confidence threshold — below this we run pix2tex as well
# and pick the better output
NOUGAT_CONFIDENCE_THRESHOLD = 0.4

# minimum equation length — shorter strings are likely noise
MIN_EQUATION_LENGTH = 3


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_path(row: pd.Series) -> Path:
    source   = row["source"]
    filename = row["filename"]
    root     = SOURCE_ROOTS[source]

    if source == "olympiad":
        subfolder = OLYMPIAD_SUBFOLDERS.get(row["doc_type"], "problems")
        return root / subfolder / filename

    return root / filename


# ── latex parsing ─────────────────────────────────────────────────────────────

# inline equations: $...$
INLINE_PATTERN  = re.compile(r'\$([^$]{3,})\$')

# display equations: $$...$$
DISPLAY_PATTERN = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)

# Nougat uses \(...\) for inline and \[...\] or \begin{equation} for display
NOUGAT_INLINE_PATTERN  = re.compile(r'\\\((.+?)\\\)', re.DOTALL)
NOUGAT_DISPLAY_PATTERN = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
NOUGAT_ENV_PATTERN     = re.compile(r'\\begin\{(?:equation|align|gather)\*?\}(.+?)\\end\{(?:equation|align|gather)\*?\}', re.DOTALL)

# count valid latex tokens as a confidence proxy
LATEX_TOKEN_PATTERN = re.compile(
    r'\\(?:frac|sqrt|sum|int|prod|lim|infty|alpha|beta|gamma|delta|theta|'
    r'omega|sigma|pi|mu|lambda|epsilon|phi|psi|nabla|partial|cdot|times|'
    r'leq|geq|neq|approx|equiv|rightarrow|leftarrow|vec|hat|bar|dot)'
)


def count_latex_tokens(text: str) -> int:
    """Count recognized LaTeX command tokens — used as a confidence proxy."""
    return len(LATEX_TOKEN_PATTERN.findall(text))


def extract_equations_from_markdown(markdown: str) -> dict:
    display = (
        DISPLAY_PATTERN.findall(markdown) +
        NOUGAT_DISPLAY_PATTERN.findall(markdown) +
        NOUGAT_ENV_PATTERN.findall(markdown)
    )
    cleaned = DISPLAY_PATTERN.sub("", markdown)
    cleaned = NOUGAT_DISPLAY_PATTERN.sub("", cleaned)
    cleaned = NOUGAT_ENV_PATTERN.sub("", cleaned)

    inline = (
        INLINE_PATTERN.findall(cleaned) +
        NOUGAT_INLINE_PATTERN.findall(cleaned)
    )

    inline  = [eq.strip() for eq in inline  if len(eq.strip()) >= MIN_EQUATION_LENGTH]
    display = [eq.strip() for eq in display if len(eq.strip()) >= MIN_EQUATION_LENGTH]

    return {
        "inline":        inline,
        "display":       display,
        "full_markdown": markdown,
        "latex_tokens":  count_latex_tokens(markdown),
    }


# ── nougat extraction ─────────────────────────────────────────────────────────

def load_nougat():
    """Load Nougat model — called once, reused across all documents."""
    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint

    checkpoint = get_checkpoint(model_tag="0.1.0-small")
    model      = NougatModel.from_pretrained(checkpoint)
    model      = model.to(DEVICE)
    model.eval()
    return model


def run_nougat_on_page(model, page_image: Image.Image) -> tuple[str, float]:
    """
    Run Nougat on a single page image.
    Returns (markdown_text, confidence_score).
    Confidence is approximated by the ratio of valid LaTeX tokens to page area.
    """
    tensor = model.encoder.prepare_input(page_image, random_padding=False)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model.inference(image_tensors=tensor, early_stopping=True)

    markdown = output["predictions"][0] if output["predictions"] else ""

    confidence = min(count_latex_tokens(markdown) / max(len(markdown), 1) * 100, 1.0)

    return markdown, confidence


# ── pix2tex fallback ──────────────────────────────────────────────────────────

def load_pix2tex():
    """Load pix2tex model — called once as fallback."""
    from pix2tex.cli import LatexOCR
    return LatexOCR()


def run_pix2tex_on_page(model, page_image: Image.Image) -> list[str]:
    """
    Run pix2tex on a full page image.
    pix2tex works best on individual equation crops — here we pass the full
    page and let it find equations. Returns a list of LaTeX strings.
    """
    try:
        result = model(page_image)
        if result and len(result.strip()) >= MIN_EQUATION_LENGTH:
            return [result.strip()]
    except Exception:
        pass
    return []


# ── ensemble logic ────────────────────────────────────────────────────────────

def ensemble_pick(nougat_result: dict, pix2tex_equations: list) -> dict:
    """
    Compare Nougat and pix2tex outputs and return the better one.
    Decision is based on total LaTeX token count — more recognized
    commands means higher confidence output.
    """
    nougat_score  = nougat_result["latex_tokens"]
    pix2tex_score = sum(count_latex_tokens(eq) for eq in pix2tex_equations)

    if pix2tex_score > nougat_score and pix2tex_equations:
        return {
            "inline":        pix2tex_equations,
            "display":       [],
            "full_markdown": "\n".join(pix2tex_equations),
            "latex_tokens":  pix2tex_score,
            "source":        "pix2tex",
        }

    nougat_result["source"] = "nougat"
    return nougat_result


# ── page renderer ─────────────────────────────────────────────────────────────

def render_page(page: fitz.Page, dpi: int = 150) -> Image.Image:
    """Render a PDF page to a PIL Image at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ── core extraction ───────────────────────────────────────────────────────────

def extract_equations_from_pdf(
    pdf_path: Path,
    page_types: dict,
    nougat_model,
    pix2tex_model,
) -> dict:
    """
    Extract LaTeX equations from all equation-classified pages in a PDF.

    Strategy per page:
    1. Render page to image
    2. Run Nougat → get markdown + confidence
    3. If confidence < threshold → also run pix2tex → ensemble pick winner
    4. Parse and store all equations found
    """
    doc     = fitz.open(pdf_path)
    results = {}

    for page_num_str, page_info in page_types.items():
        content_types = page_info.get("content_types", [])

        if "equations" not in content_types:
            continue

        page_num = int(page_num_str)
        page     = doc[page_num]

        # render at 150 DPI — good balance of quality vs speed for Nougat
        page_image = render_page(page, dpi=150)

        # run Nougat
        markdown, confidence = run_nougat_on_page(nougat_model, page_image)
        nougat_result        = extract_equations_from_markdown(markdown)

        # run pix2tex ensemble if Nougat confidence is low
        if confidence < NOUGAT_CONFIDENCE_THRESHOLD:
            pix2tex_equations = run_pix2tex_on_page(pix2tex_model, page_image)
            final_result      = ensemble_pick(nougat_result, pix2tex_equations)
        else:
            nougat_result["source"] = "nougat"
            final_result            = nougat_result

        # skip pages where nothing was found
        total_equations = len(final_result["inline"]) + len(final_result["display"])
        if total_equations == 0:
            continue

        results[page_num_str] = {
            "equations_inline":  final_result["inline"],
            "equations_display": final_result["display"],
            "full_markdown":     final_result["full_markdown"],
            "source":            final_result["source"],
            "confidence":        round(confidence, 3),
            "equation_count":    total_equations,
        }

    doc.close()
    return results


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df             = pd.read_csv(INVENTORY_PATH)
    classification = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))

    df = df[df["source"].isin(SUPPORTED_SOURCES)].reset_index(drop=True)

    # filter to only docs that have at least one equation page
    def has_equation_pages(filename: str) -> bool:
        doc_id = filename.replace(".pdf", "")
        pages  = classification.get(doc_id, {}).get("pages", {})
        return any(
            "equations" in p.get("content_types", [])
            for p in pages.values()
        )

    df = df[df["filename"].apply(has_equation_pages)].reset_index(drop=True)
    print(f"Documents with equation pages: {len(df)}")
    print(f"Loading models on {DEVICE}...\n")

    nougat_model  = load_nougat()
    pix2tex_model = load_pix2tex()

    print("Models loaded. Starting extraction...\n")

    skipped = 0
    failed  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting equations"):
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
            result = extract_equations_from_pdf(
                pdf_path, page_types, nougat_model, pix2tex_model
            )
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        out_path = OUTPUT_DIR / f"{doc_id}_equations.json"
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