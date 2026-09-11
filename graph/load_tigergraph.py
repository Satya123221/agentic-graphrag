import json
from pathlib import Path

DATA_DIR = Path("data/graph_data")


def load_jsonl(filename):
    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


documents = load_jsonl("documents.jsonl")
events = load_jsonl("events.jsonl")
games = load_jsonl("games.jsonl")
sports = load_jsonl("sports.jsonl")
edges = load_jsonl("document_event_edges.jsonl")


print("TIGERGRAPH DATA READY")
print("=" * 60)
print("Documents:", len(documents))
print("Events:", len(events))
print("Games:", len(games))
print("Sports:", len(sports))
print("Document → Event edges:", len(edges))