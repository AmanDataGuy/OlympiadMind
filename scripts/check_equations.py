import json
from pathlib import Path

files = list(Path("data/extracted/equations").glob("*.json"))
print(f"Output files: {len(files)}")

data = json.load(open(files[0], encoding="utf-8"))
print(f"Doc: {data['doc_id']}")
print(f"Pages with equations: {len(data['pages'])}")

if data["pages"]:
    page = list(data["pages"].values())[0]
    print(f"Source: {page['source']}")
    print(f"Equations found: {page['equation_count']}")
    print(f"Sample inline: {page['equations_inline'][:2]}")
    print(f"Sample display: {page['equations_display'][:1]}")