"""
figure_extractor.py
-------------------
Step 2.3 — Figure Extraction and Description

Two-stage pipeline per figure:
  Stage 1 — CLIP classifies the diagram type (force diagram, circuit, graph etc.)
  Stage 2 — Qwen2-VL generates a structured JSON description using a
             type-specific prompt based on CLIP's classification

Only processes pages classified as 'figure' in page_classification.json.
Skips figures smaller than MIN_FIGURE_SIZE pixels — too small to describe.

Output: data/extracted/figures/{doc_id}_figures.json
        data/extracted/figures/images/{figure_id}.png  — saved figure images
"""

import json
import io
import re
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

# skip figures smaller than this in either dimension — too small to describe
MIN_FIGURE_SIZE = 100

# CLIP diagram type candidates
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


# ── clip classifier ───────────────────────────────────────────────────────────

def load_clip():
    """Load CLIP model for diagram type classification."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def classify_diagram(clip_model, preprocess, tokenizer, image: Image.Image) -> str:
    """
    Use CLIP to classify the diagram type.
    Returns the most likely diagram type string from DIAGRAM_TYPES.
    """
    import open_clip

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


# ── qwen2-vl descriptor ───────────────────────────────────────────────────────

# type-specific prompts — more focused prompts give better descriptions
PROMPTS = {
    "force diagram with arrows": """Describe this physics force diagram in JSON:
{
  "diagram_type": "force diagram",
  "physics_concept": "the main concept shown",
  "objects": ["list of objects in the diagram"],
  "forces": ["list of forces with directions"],
  "description": "2 sentence description"
}
Return ONLY valid JSON.""",

    "electric circuit diagram": """Describe this electric circuit diagram in JSON:
{
  "diagram_type": "circuit diagram",
  "physics_concept": "the main concept shown",
  "components": ["list of circuit components"],
  "topology": "series / parallel / mixed",
  "description": "2 sentence description"
}
Return ONLY valid JSON.""",

    "graph or plot with axes": """Describe this physics graph in JSON:
{
  "diagram_type": "graph",
  "physics_concept": "the main concept shown",
  "x_axis": "x axis label and units",
  "y_axis": "y axis label and units",
  "curve_behavior": "description of the curve shape",
  "description": "2 sentence description"
}
Return ONLY valid JSON.""",

    "default": """You are analyzing a figure from a physics olympiad problem.
Describe this figure in JSON:
{
  "diagram_type": "force diagram | circuit diagram | graph | geometric | experimental setup | other",
  "physics_concept": "the main physics concept illustrated",
  "axes_labels": ["list of axis labels if any"],
  "annotations": ["list of labels, arrows, symbols visible"],
  "description": "2 sentence description"
}
Return ONLY valid JSON.""",
}

def load_qwen():
    """Load Qwen2-VL 2B — fits comfortably on 6GB VRAM."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor


def describe_figure(qwen_model, processor, image: Image.Image, diagram_type: str) -> dict:
    """
    Use Qwen2-VL to generate a structured description of a figure.
    Uses a type-specific prompt based on CLIP's classification.
    Returns parsed JSON dict or a fallback dict on parse failure.
    """
    prompt = PROMPTS.get(diagram_type, PROMPTS["default"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=300)

    # decode only the new tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response  = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # parse JSON from response
    try:
        # strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "diagram_type":    diagram_type,
            "physics_concept": "unknown",
            "description":     response[:200],
            "parse_error":     True,
        }


# ── core extraction ───────────────────────────────────────────────────────────

def extract_figures_from_pdf(
    pdf_path: Path,
    doc_id: str,
    page_types: dict,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    qwen_model,
    qwen_processor,
) -> list:
    """
    Extract and describe all figures from pages classified as 'figure'.

    For each figure:
    1. Extract image bytes from PDF
    2. Filter out tiny images
    3. CLIP classifies diagram type
    4. Qwen2-VL generates structured description
    5. Save image to disk
    """
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

            # skip tiny images — logos, decorations, not physics diagrams
            if width < MIN_FIGURE_SIZE or height < MIN_FIGURE_SIZE:
                continue

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # stage 1 — CLIP classification
            diagram_type = classify_diagram(
                clip_model, clip_preprocess, clip_tokenizer, pil_image
            )

            # stage 2 — Qwen2-VL description
            description = describe_figure(
                qwen_model, qwen_processor, pil_image, diagram_type
            )

            # save image
            figure_id  = f"{doc_id}_p{page_num_str}_f{img_idx}"
            image_path = IMAGES_DIR / f"{figure_id}.png"
            pil_image.save(image_path)

            figures.append({
                "figure_id":    figure_id,
                "page":         page_num_str,
                "image_path":   str(image_path),
                "width":        width,
                "height":       height,
                "diagram_type": diagram_type,
                **description,
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

    # filter to docs with at least one figure page
    def has_figure_pages(filename: str) -> bool:
        doc_id = filename.replace(".pdf", "")
        pages  = classification.get(doc_id, {}).get("pages", {})
        return any(
            "figure" in p.get("content_types", [])
            for p in pages.values()
        )

    df = df[df["filename"].apply(has_figure_pages)].reset_index(drop=True)
    print(f"Documents with figure pages: {len(df)}")
    print(f"Loading models on {DEVICE}...\n")

    clip_model, clip_preprocess, clip_tokenizer = load_clip()
    qwen_model, qwen_processor                  = load_qwen()

    print("Models loaded. Starting extraction...\n")

    skipped = 0
    failed  = 0

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
                clip_model, clip_preprocess, clip_tokenizer,
                qwen_model, qwen_processor,
            )
        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            failed += 1
            continue

        out_path = OUTPUT_DIR / f"{doc_id}_figures.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id":       doc_id,
                "source":       row["source"],
                "figure_count": len(figures),
                "figures":      figures,
            }, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"  Output  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()