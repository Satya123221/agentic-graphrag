"""Build a high-confidence Athlete catalog from Olympic infobox medalists.

Existing valid athlete IDs are preserved when the cleaned name matches.
Noisy non-person entries are dropped. Does not load TigerGraph.
"""

import json
import sys
from pathlib import Path

sys.path.append(".")

from config import CORPUS_PATH, DATA_DIR
from ingestion.name_utils import (
    clean_display_name,
    is_non_person_value,
    looks_like_person_name,
    make_id,
    normalize,
    extract_infobox_fields,
    split_medal_value,
)

EVENTS_PATH = DATA_DIR / "graph_data" / "events.jsonl"
ATHLETES_JSONL = DATA_DIR / "graph_data" / "athletes.jsonl"
ATHLETES_TSV = DATA_DIR / "graph_data" / "athletes.tsv"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_id_map(old_athletes):
    by_norm = {}
    for athlete in old_athletes:
        cleaned = clean_display_name(athlete.get("name", ""))
        if not looks_like_person_name(cleaned):
            continue
        key = normalize(cleaned)
        by_norm.setdefault(key, []).append(athlete["athlete_id"])
    return by_norm


def choose_id(name, by_norm):
    preferred = make_id(name)
    key = normalize(name)
    old_ids = by_norm.get(key, [])
    if preferred in old_ids:
        return preferred
    if len(old_ids) == 1:
        return old_ids[0]
    if old_ids:
        slug_matches = [athlete_id for athlete_id in old_ids if athlete_id == preferred]
        if slug_matches:
            return slug_matches[0]
        return sorted(old_ids, key=len)[0]
    return preferred


def main():
    old_athletes = load_jsonl(ATHLETES_JSONL)
    old_count = len(old_athletes)
    by_norm = existing_id_map(old_athletes)

    event_ids = {row["event_id"] for row in load_jsonl(EVENTS_PATH)}
    found_names = {}

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("doc_id") not in event_ids:
                continue

            for raw in extract_infobox_fields(doc.get("text", ""), gold_only=False):
                for name in split_medal_value(raw):
                    if not looks_like_person_name(name):
                        continue
                    key = normalize(name)
                    if not key or is_non_person_value(name):
                        continue
                    if key not in found_names:
                        found_names[key] = name

    athletes = {}
    for key, name in found_names.items():
        athlete_id = choose_id(name, by_norm)
        current = athletes.get(athlete_id)
        if current is None or len(name) < len(current["name"]):
            athletes[athlete_id] = {
                "athlete_id": athlete_id,
                "name": name,
            }

    rows = sorted(athletes.values(), key=lambda row: row["name"].casefold())
    ATHLETES_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with ATHLETES_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ATHLETES_TSV.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["athlete_id"] + "\t" + row["name"].replace("\t", " ") + "\n")

    old_ids = {row["athlete_id"] for row in old_athletes}
    preserved = sum(1 for row in rows if row["athlete_id"] in old_ids)

    print("ATHLETE CATALOG UPDATE COMPLETE")
    print("Old athlete count:", old_count)
    print("New athlete count:", len(rows))
    print("Preserved existing IDs:", preserved)
    print("Output:", ATHLETES_JSONL)


if __name__ == "__main__":
    main()
