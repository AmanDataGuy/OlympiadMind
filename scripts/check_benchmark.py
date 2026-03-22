import json
from pathlib import Path

gt = [json.loads(l) for l in Path("data/evaluation/ground_truth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
olympiad = [r for r in gt if r.get("source") == "olympiad"]
print(f"Total ground truth: {len(gt)}")
print(f"Olympiad problems: {len(olympiad)}")
print()
for r in olympiad[:3]:
    print(f"  {r['doc_id']}: {r['typeset_text_sample'][:80]}")