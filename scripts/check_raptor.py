import json
from pathlib import Path

tree = json.loads(Path("data/raptor/tree.json").read_text(encoding="utf-8"))
level1 = tree["levels"]["1"]
print(f"Level 1 nodes: {len(level1)}")
print()
for node in level1[:3]:
    print(f"Cluster {node['cluster_id']} ({node['n_children']} children):")
    print(f"  {node['text'][:200]}")
    print() 