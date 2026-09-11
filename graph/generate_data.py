import json
import re
import sys

sys.path.append(".")

from config import CORPUS_PATH, DATA_DIR

GRAPH_DATA_DIR = DATA_DIR / "graph_data"
GRAPH_DATA_DIR.mkdir(exist_ok=True)

events = []
games = {}
sports = {}
documents = []
edges = []


def clean(value):
    return value.strip() if value else ""


def make_id(value):
    return re.sub(
        r"[^a-zA-Z0-9]+", "_", value
    ).strip("_").lower()


with open(CORPUS_PATH, "r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        doc = json.loads(line)

        doc_id = doc["doc_id"]
        text = doc["text"]
        title = doc["title"]

        # Document
        documents.append({
            "doc_id": doc_id,
            "title": title,
            "url": doc["url"]
        })

        # Event
        event_id = doc_id

        events.append({
            "event_id": event_id,
            "name": title
        })

        edges.append({
            "from": doc_id,
            "to": event_id
        })

        # Olympic Games
        game_match = re.search(
            r"games:\s*([^\n]+)",
            text,
            re.IGNORECASE
        )

        if game_match:

            game_name = clean(game_match.group(1))
            game_id = make_id(game_name)

            games[game_id] = {
                "games_id": game_id,
                "name": game_name
            }

        # Sport
        sport_match = re.search(
            r"(?:sport|discipline):\s*([^\n]+)",
            text,
            re.IGNORECASE
        )

        if sport_match:

            sport_name = clean(sport_match.group(1))
            sport_id = make_id(sport_name)

            sports[sport_id] = {
                "sport_id": sport_id,
                "name": sport_name
            }


def write_jsonl(filename, data):

    path = GRAPH_DATA_DIR / filename

    with open(path, "w", encoding="utf-8") as f:

        for item in data:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    return path


write_jsonl("documents.jsonl", documents)
write_jsonl("events.jsonl", events)
write_jsonl("games.jsonl", list(games.values()))
write_jsonl("sports.jsonl", list(sports.values()))
write_jsonl("document_event_edges.jsonl", edges)


print("GRAPH DATA GENERATION COMPLETE")
print("=" * 60)
print("Documents:", len(documents))
print("Events:", len(events))
print("Games:", len(games))
print("Sports:", len(sports))
print("Output:", GRAPH_DATA_DIR)
print()
print("FILES CREATED:")
print("documents.jsonl")
print("events.jsonl")
print("games.jsonl")
print("sports.jsonl")
print("document_event_edges.jsonl")