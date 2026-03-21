import fitz

files = {
    "science_bowl": "data/raw/science_bowl/sciencebowl_en_mcq_1991.pdf",
    "cbse":         "data/raw/cbse/cbse_en_assertion_classxi.pdf",
    "nist":         "data/raw/tabular/nist_en_constants_table.pdf",
}

for name, path in files.items():
    doc  = fitz.open(path)
    page = doc[2]
    print(f"\n=== {name} ===")
    print(page.get_text()[:400])
    doc.close()