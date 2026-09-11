import json
from pathlib import Path

src = Path("data/graph_data/games.jsonl")
out = Path("data/graph_data/games.tsv")

rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]

with open(out, "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write(r["games_id"] + "\t" + r["name"].replace("\t", " ") + "\n")

print("Created:", out)
print("Rows:", len(rows))
