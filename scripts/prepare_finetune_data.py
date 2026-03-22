import json
import random
from pathlib import Path

random.seed(42)

# load concept labels
labels = [json.loads(l) for l in Path("data/corpus/concept_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

# load problem texts
docs = {}
for line in Path("data/extracted/documents.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        doc = json.loads(line)
        docs[doc["doc_id"]] = doc
    except json.JSONDecodeError:
        continue

# build instruction pairs
pairs = []
for label in labels:
    doc_id   = label["doc_id"]
    concepts = label["concepts"]
    if not concepts or doc_id not in docs:
        continue
    doc  = docs[doc_id]
    text = doc.get("pages", [{}])[0].get("text", "").strip()[:600]
    if not text:
        continue
    pairs.append({
        "instruction": "List the physics concepts required to solve this problem.",
        "input":       text,
        "output":      json.dumps(concepts),
    })

# shuffle and split
random.shuffle(pairs)
n       = len(pairs)
n_train = int(n * 0.75)
n_val   = int(n * 0.125)

train = pairs[:n_train]
val   = pairs[n_train:n_train + n_val]
test  = pairs[n_train + n_val:]

Path("data/finetune").mkdir(parents=True, exist_ok=True)
Path("data/finetune/train.jsonl").write_text("\n".join(json.dumps(p) for p in train), encoding="utf-8")
Path("data/finetune/val.jsonl").write_text("\n".join(json.dumps(p) for p in val),   encoding="utf-8")
Path("data/finetune/test.jsonl").write_text("\n".join(json.dumps(p) for p in test), encoding="utf-8")

print(f"Total pairs : {len(pairs)}")
print(f"Train       : {len(train)}")
print(f"Val         : {len(val)}")
print(f"Test        : {len(test)}")
print(f"Sample:")
print(json.dumps(train[0], indent=2))