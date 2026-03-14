import sys, json
sys.path.insert(0, ".")
from pathlib import Path
from src.extraction.figure_extractor import (
    load_clip, load_qwen, extract_figures_from_pdf
)

classification = json.loads(
    Path("data/manifests/page_classification.json").read_text(encoding="utf-8")
)
page_types = classification.get("IPhO_2022_Q1", {}).get("pages", {})

clip_model, clip_pre, clip_tok = load_clip()
qwen_model, qwen_proc          = load_qwen()

figures = extract_figures_from_pdf(
    Path("data/raw/olympiad/problems/IPhO_2022_Q1.pdf"),
    "IPhO_2022_Q1", page_types,
    clip_model, clip_pre, clip_tok,
    qwen_model, qwen_proc
)

print(f"Figures extracted: {len(figures)}")
print(json.dumps(figures[:1], indent=2))