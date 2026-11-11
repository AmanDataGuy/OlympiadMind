"""
extract_figures_fast.py
-----------------------
Fast Figure Extraction

Extracts all figures from PDFs and classifies them with CLIP.
No Qwen2-VL description — that runs separately in describe_figures.py.

Runs in ~10 minutes on the full dataset.

Output: data/extracted/figures/{doc_id}_figures.json
        data/extracted/figures/images/{figure_id}.png
"""

import json
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
OUTPUT_DIR          = Path("data/extracted/figures")
IMAGES_DIR          = OUTPUT_DIR / "images"

SUPPORTED_SOURCES = {"olympiad", "cbse", "science_bowl", "multilingual"}

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

MIN_FIGURE_SIZE = 100

DIAGRAM_TYPES = [
    "force diagram with arrows",
    "electric circuit diagram",
    "graph or plot with axes",
    "geometric diagram with shapes",
    "experimental setup or apparatus",
    "wave or oscillation diagram",
    "optics ray diagram",
    "magnetic field diagram",
    "energy level diagram",
    "other physics diagram",
]


# ── path resolution ───────────────────────────────────────────────────────────

def resolve_path(row: pd.Series) -> Path:
    source   = row["source"]
    filename = row["filename"]
    root     = SOURCE_ROOTS[source]

    if source == "olympiad":
        subfolder = OLYMPIAD_SUBFOLDERS.get(row["doc_type"], "problems")
        return root / subfolder / filename

    return root / filename


# ── clip ──────────────────────────────────────────────────────────────────────

def load_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def classify_diagram(clip_model, preprocess, tokenizer, image: Image.Image) -> str:
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    text_tokens  = tokenizer(DIAGRAM_TYPES).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_features  = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze(0)
        best_idx   = similarity.argmax().item()

    return DIAGRAM_TYPES[best_idx]


# ── extraction ────────────────────────────────────────────────────────────────

def extract_figures_from_pdf(
    pdf_path: Path,
    doc_id: str,
    page_types: dict,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
) -> list:
    doc     = fitz.open(pdf_path)
    figures = []

    for page_num_str, page_info in page_types.items():
        content_types = page_info.get("content_types", [])

        if "figure" not in content_types:
            continue

        page_num   = int(page_num_str)
        page       = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref        = img[0]
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width       = base_image.get("width",  0)
            height      = base_image.get("height", 0)

            if width < MIN_FIGURE_SIZE or height < MIN_FIGURE_SIZE:
                continue

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            diagram_type = classify_diagram(
                clip_model, clip_preprocess, clip_tokenizer, pil_image
            )

            figure_id  = f"{doc_id}_p{page_num_str}_f{img_idx}"
            image_path = IMAGES_DIR / f"{figure_id}.png"
            pil_image.save(image_path)

            figures.append({
                "figure_id":   figure_id,
                "page":        page_num_str,
                "image_path":  str(image_path),
                "width":       width,
                "height":      height,
                "diagram_type": diagram_type,
                "description": None,  # filled in by describe_figures.py
            })

    doc.close()
    return figures


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df             = pd.read_csv(INVENTORY_PATH)
    classification = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))

    df = df[df["source"].isin(SUPPORTED_SOURCES)].reset_index(drop=True)

    def has_figure_pages(filename: str) -> bool:
        doc_id = filename.replace(".pdf", "")
        pages  = classification.get(doc_id, {}).get("pages", {})
        return any(
            "figure" in p.get("content_types", [])
            for p in pages.values()
        )

    df = df[df["filename"].apply(has_figure_pages)].reset_index(drop=True)
    print(f"Documents with figure pages: {len(df)}")
    print(f"Loading CLIP on {DEVICE}...\n")

    clip_model, clip_pre, clip_tok = load_clip()
    print("CLIP loaded. Starting extraction...\n")

    skipped = 0
    failed  = 0
    total_figures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting figures"):
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
            figures = extract_figures_from_pdf(
                pdf_path, doc_id, page_types,
                clip_model, clip_pre, clip_tok,
            )
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        total_figures += len(figures)

        out_path = OUTPUT_DIR / f"{doc_id}_figures.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":       doc_id,
                "source":       row["source"],
                "figure_count": len(figures),
                "figures":      figures,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Total figures : {total_figures}")
    print(f"  Skipped       : {skipped}")
    print(f"  Failed        : {failed}")
    print(f"  Output        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()