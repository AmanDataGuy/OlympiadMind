import json
from pathlib import Path

labels = [json.loads(l) for l in Path("data/corpus/concept_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Total labels: {len(labels)}")
print(f"Empty concepts: {sum(1 for l in labels if not l['concepts'])}")
print()
print("Sample:")
for l in labels[:5]:
    print(f"  {l['doc_id']}: {l['concepts']}")