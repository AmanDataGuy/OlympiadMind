import json
from pathlib import Path

files = list(Path("data/extracted/equations").glob("*.json"))

total_equations = 0
nougat_pages = 0
pix2tex_pages = 0
empty_docs = 0

for f in files:
    data = json.load(open(f, encoding="utf-8"))
    if not data["pages"]:
        empty_docs += 1
        continue
    for page in data["pages"].values():
        total_equations += page["equation_count"]
        if page["source"] == "nougat":
            nougat_pages += 1
        else:
            pix2tex_pages += 1

print(f"Total equations extracted : {total_equations}")
print(f"Pages sourced from Nougat : {nougat_pages}")
print(f"Pages sourced from pix2tex: {pix2tex_pages}")
print(f"Docs with no equations    : {empty_docs}")