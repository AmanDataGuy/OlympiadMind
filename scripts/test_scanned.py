import sys, json
sys.path.insert(0, ".")
from pathlib import Path
from src.extraction.scanned_ocr import load_ocr, extract_scanned_pages

classification = json.loads(
    Path("data/manifests/page_classification.json").read_text(encoding="utf-8")
)
page_types = classification.get("IPhO_1969_Q2", {}).get("pages", {})

ocr = load_ocr()
result = extract_scanned_pages(
    Path("data/raw/olympiad/problems/IPhO_1969_Q2.pdf"),
    page_types, ocr
)
print(json.dumps(result, indent=2)[:500])