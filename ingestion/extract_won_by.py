"""Extract true Event → Athlete WON_BY candidates from infobox gold fields.

This script does not load TigerGraph and does not modify event_athletes.tsv.
Athlete IDs come only from the existing athletes.jsonl catalog.
"""

import json
import sys

sys.path.append(".")

from config import CORPUS_PATH, DATA_DIR
from ingestion.name_utils import (
    clean_display_name,
    extract_infobox_fields,
    is_country_or_team,
    is_non_person_value,
    looks_like_person_name,
    normalize,
    split_medal_value,
)

EVENTS_PATH = DATA_DIR / "graph_data" / "events.jsonl"
ATHLETES_PATH = DATA_DIR / "graph_data" / "athletes.jsonl"
WON_BY_PATH = DATA_DIR / "graph_data" / "won_by.tsv"
UNRESOLVED_PATH = DATA_DIR / "graph_data" / "won_by_unresolved.tsv"
AUDIT_PATH = DATA_DIR / "graph_data" / "won_by_extraction.jsonl"


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_athletes(athletes):
    exact = {}
    for athlete in athletes:
        name = athlete.get("name", "")
        if not looks_like_person_name(name):
            continue
        keys = {normalize(name), normalize(clean_display_name(name))}
        for key in keys:
            if not key or is_non_person_value(key):
                continue
            exact.setdefault(key, [])
            if athlete not in exact[key]:
                exact[key].append(athlete)
    return exact


def resolve_name(name, index):
    key = normalize(name)
    if not key or is_country_or_team(name):
        return None
    matches = index.get(key, [])
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    event_ids = {row["event_id"] for row in load_jsonl(EVENTS_PATH)}
    athletes = load_jsonl(ATHLETES_PATH)
    index = index_athletes(athletes)

    resolved_rows = []
    unresolved_rows = []
    audit_rows = []

    gold_fields_found = 0
    events_with_gold = 0
    events_missing_gold = 0
    country_or_team_gold = 0

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            doc = json.loads(line)
            event_id = doc["doc_id"]
            if event_id not in event_ids:
                continue

            gold_fields = extract_infobox_fields(doc.get("text", ""), gold_only=True)
            parsed_names = []
            resolved = []
            unresolved = []

            if not gold_fields:
                events_missing_gold += 1
                unresolved.append({
                    "name": "",
                    "reason": "missing_gold",
                    "raw_gold": "",
                })
            else:
                events_with_gold += 1
                gold_fields_found += len(gold_fields)

                for raw_gold in gold_fields:
                    if is_country_or_team(raw_gold) or is_non_person_value(raw_gold):
                        reason = (
                            "country_or_team"
                            if is_country_or_team(raw_gold)
                            else "non_athlete"
                        )
                        if reason == "country_or_team":
                            country_or_team_gold += 1
                        unresolved.append({
                            "name": raw_gold,
                            "reason": reason,
                            "raw_gold": raw_gold,
                        })
                        continue

                    names = split_medal_value(raw_gold)
                    if not names:
                        unresolved.append({
                            "name": raw_gold,
                            "reason": "unparsed_gold",
                            "raw_gold": raw_gold,
                        })
                        continue

                    for name in names:
                        parsed_names.append(name)
                        athlete = resolve_name(name, index)
                        if athlete:
                            resolved.append({
                                "event_id": event_id,
                                "athlete_id": athlete["athlete_id"],
                                "athlete_name": athlete["name"],
                                "gold_name": name,
                                "raw_gold": raw_gold,
                            })
                        elif looks_like_person_name(name):
                            unresolved.append({
                                "name": name,
                                "reason": "unresolved_person",
                                "raw_gold": raw_gold,
                            })
                        else:
                            unresolved.append({
                                "name": name,
                                "reason": "non_athlete",
                                "raw_gold": raw_gold,
                            })

            unique_resolved = {}
            for row in resolved:
                unique_resolved[(row["event_id"], row["athlete_id"])] = row
            resolved = list(unique_resolved.values())

            resolved_rows.extend(resolved)
            for item in unresolved:
                unresolved_rows.append({
                    "event_id": event_id,
                    "raw_gold": item["raw_gold"],
                    "parsed_name": item["name"],
                    "reason": item["reason"],
                })

            audit_rows.append({
                "event_id": event_id,
                "title": doc.get("title", ""),
                "gold_fields": gold_fields,
                "parsed_names": parsed_names,
                "resolved": [
                    {
                        "athlete_id": row["athlete_id"],
                        "athlete_name": row["athlete_name"],
                        "gold_name": row["gold_name"],
                    }
                    for row in resolved
                ],
                "unresolved": unresolved,
            })

    WON_BY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with WON_BY_PATH.open("w", encoding="utf-8") as f:
        f.write("event_id\tathlete_id\tathlete_name\tgold_name\n")
        for row in sorted(resolved_rows, key=lambda r: (r["event_id"], r["athlete_id"])):
            f.write(
                f"{row['event_id']}\t{row['athlete_id']}\t"
                f"{row['athlete_name']}\t{row['gold_name']}\n"
            )

    with UNRESOLVED_PATH.open("w", encoding="utf-8") as f:
        f.write("event_id\traw_gold\tparsed_name\treason\n")
        for row in unresolved_rows:
            f.write(
                f"{row['event_id']}\t{row['raw_gold']}\t"
                f"{row['parsed_name']}\t{row['reason']}\n"
            )

    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    unresolved_people = [
        row for row in unresolved_rows if row["reason"] == "unresolved_person"
    ]

    print("WON_BY EXTRACTION COMPLETE")
    print("=" * 70)
    print("Valid Olympic events processed:", len(event_ids))
    print("Events with gold field:", events_with_gold)
    print("Gold fields found:", gold_fields_found)
    print("Country/team gold fields:", country_or_team_gold)
    print("Events missing gold field:", events_missing_gold)
    print("Resolved athlete winners:", len(resolved_rows))
    print("Unresolved person names:", len(unresolved_people))
    print("Unresolved rows including missing gold:", len(unresolved_rows))
    print("Output:", WON_BY_PATH)

    print("\nSAMPLE RESOLVED RELATIONSHIPS")
    for row in resolved_rows[:15]:
        print(
            f"{row['event_id']}\t{row['gold_name']} -> "
            f"{row['athlete_id']} ({row['athlete_name']})"
        )


if __name__ == "__main__":
    main()
