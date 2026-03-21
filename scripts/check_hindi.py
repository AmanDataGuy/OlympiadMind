import fitz

doc = fitz.open("data/raw/hindi/rbse_hi_pyq_classxii.pdf")
for i in range(3):
    page = doc[i]
    text = page.get_text()
    print(f"Page {i}: {len(text)} chars")
    print(text[:200])
    print("---")
doc.close()