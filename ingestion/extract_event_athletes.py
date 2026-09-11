import json
import re
from pathlib import Path

CORPUS = Path("data/corpus.jsonl")
ATHLETES = Path("data/graph_data/athletes.jsonl")
EVENTS = Path("data/graph_data/events.jsonl")
OUTPUT = Path("data/graph_data/event_athletes.tsv")

# Load athletes
athletes = []

with ATHLETES.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            athletes.append(json.loads(line))

# Load valid Olympic event IDs
event_ids = set()

with EVENTS.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            event_ids.add(json.loads(line)["event_id"])

edges = set()

# Search athlete names only inside their corresponding event documents
with CORPUS.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        doc = json.loads(line)
        doc_id = doc["doc_id"]

        if doc_id not in event_ids:
            continue

        text = doc.get("text", "")

        for athlete in athletes:
            name = athlete["name"]

            # Ignore single-word names to avoid false matches
            if len(name.split()) < 2:
                continue

            pattern = (
                r"(?<![A-Za-z])"
                + re.escape(name)
                + r"(?![A-Za-z])"
            )

            if re.search(pattern, text, re.IGNORECASE):
                edges.add(
                    (doc_id, athlete["athlete_id"])
                )

# Write TSV
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for event_id, athlete_id in sorted(edges):
        f.write(event_id + "\t" + athlete_id + "\n")

print("EVENT-ATHLETE EXTRACTION COMPLETE")
print("Candidate edges:", len(edges))
print("Output:", OUTPUT)