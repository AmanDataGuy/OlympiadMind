import json
from pathlib import Path

with open("data/manifests/page_classification.json", encoding="utf-8") as f:
    classification = json.load(f)

pages = classification.get("IPhO_1969_Q2", {}).get("pages", {})
for p, info in pages.items():
    print(f"page {p}: {info['content_types']}")