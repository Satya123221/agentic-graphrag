import json
import re
from pathlib import Path

CORPUS = Path("data/corpus.jsonl")
OUTPUT = Path("data/graph_data/sports.jsonl")

# Matches titles such as:
# Cycling at the 2008 Summer Olympics
# Fencing at the 1988 Summer Olympics
PATTERN = re.compile(
    r"^(.*?)\s+at the \d{4}\s+(?:Summer|Winter)\s+Olympics",
    re.IGNORECASE
)

sports = {}

with CORPUS.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        doc = json.loads(line)
        title = doc.get("title", "")

        match = PATTERN.search(title)

        if match:
            name = match.group(1).strip()

            if name:
                sport_id = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    name.lower()
                ).strip("_")

                sports[sport_id] = {
                    "sport_id": sport_id,
                    "name": name
                }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for sport in sorted(sports.values(), key=lambda x: x["name"].lower()):
        f.write(json.dumps(sport, ensure_ascii=False) + "\n")

print("SPORT EXTRACTION COMPLETE")
print("Unique sports:", len(sports))
print("Output:", OUTPUT)