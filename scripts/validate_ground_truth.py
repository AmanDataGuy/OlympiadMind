import json
from pathlib import Path

lines = Path("data/evaluation/ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
errors = 0
todo_count = 0

for i, line in enumerate(lines, 1):
    if not line.strip():
        continue
    try:
        r = json.loads(line)
        todos = str(r).count("TODO")
        if todos > 0:
            print(f"  Line {i} ({r['doc_id']}): {todos} TODO fields remaining")
            todo_count += todos
    except json.JSONDecodeError as e:
        print(f"  Line {i}: JSON ERROR — {e}")
        errors += 1

print(f"\nTotal records: {len([l for l in lines if l.strip()])}")
print(f"JSON errors: {errors}")
print(f"TODO fields remaining: {todo_count}")