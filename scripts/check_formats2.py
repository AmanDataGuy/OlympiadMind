import fitz

# science bowl - already clear, check answer pattern
doc = fitz.open("data/raw/science_bowl/sciencebowl_en_mcq_1991.pdf")
print("=== SCIENCE BOWL page 2 ===")
print(doc[2].get_text()[:600])
doc.close()

# cbse - check page 5 for actual questions
doc = fitz.open("data/raw/cbse/cbse_en_assertion_classxi.pdf")
print("\n=== CBSE page 5 ===")
print(doc[5].get_text()[:600])
doc.close()

# nist - check page 0
doc = fitz.open("data/raw/tabular/nist_en_constants_table.pdf")
print("\n=== NIST page 0 ===")
print(doc[0].get_text()[:600])
doc.close()