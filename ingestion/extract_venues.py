"""Extract Venue vertices and Event → Venue HELD_AT edges from corpus infoboxes.

Does not load TigerGraph. Processes only the valid Olympic events in events.jsonl.
Venue values come only from infobox venue / venues / location fields.
"""

import json
import re
import sys
from collections import Counter, defaultdict

sys.path.append(".")

from config import CORPUS_PATH, DATA_DIR
from ingestion.name_utils import (
    COUNTRIES_AND_TEAMS,
    NOC_CODES,
    SPORTS,
    is_country_or_team,
    make_id,
    normalize,
    split_camel_case,
)

EVENTS_PATH = DATA_DIR / "graph_data" / "events.jsonl"
VENUES_JSONL = DATA_DIR / "graph_data" / "venues.jsonl"
VENUES_TSV = DATA_DIR / "graph_data" / "venues.tsv"
HELD_AT_PATH = DATA_DIR / "graph_data" / "event_venues.tsv"
UNRESOLVED_PATH = DATA_DIR / "graph_data" / "held_at_unresolved.tsv"
AUDIT_PATH = DATA_DIR / "graph_data" / "held_at_extraction.jsonl"

INFOBOX_START_RE = re.compile(r"^\[infobox", re.IGNORECASE)
FIELD_RE = re.compile(r"^\s+([A-Za-z0-9_]+):\s*(.*)$")
TITLE_GAMES_RE = re.compile(
    r"\bat the (\d{4}\s+(?:Summer|Winter) Olympics)\b",
    re.IGNORECASE,
)
CJK_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af]+")
GLUED_DIGIT_RE = re.compile(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])(?=\d)")
TRAILING_DISTANCE_RE = re.compile(r"\s+\d+(?:\.\d+)?$")

VENUE_END_WORDS = {
    "stadium", "stadia", "arena", "centre", "center", "oval", "palace",
    "park", "dome", "hall", "gymnasium", "complex", "coliseum", "colosseum",
    "velodrome", "rink", "pool", "pavilion", "bowl", "forum", "waterway",
    "barracks", "academy", "institute", "lagoon", "place", "hippodrome",
    "natatorium", "gardens", "garden", "theatre", "theater", "velodrom",
}

HOST_CITIES = {
    "albertville", "amsterdam", "athens", "atlanta", "barcelona", "beijing",
    "berlin", "busan", "calgary", "gangneung", "grenoble", "helsinki",
    "hong kong", "innsbruck", "lake placid", "lillehammer", "london",
    "los angeles", "melbourne", "mexico city", "montreal", "moscow",
    "munich", "nagano", "paris", "pusan", "pyeongchang", "rio",
    "rio de janeiro", "rome", "salt lake city", "sarajevo", "savannah",
    "seoul", "sochi", "stockholm", "sydney", "tokyo", "torino", "turin",
    "vancouver",
}

GEO_TAILS = HOST_CITIES | COUNTRIES_AND_TEAMS | {
    "alberta", "british columbia", "california", "canada", "china",
    "copacabana", "cypress provincial park", "florida", "kananaskis village",
    "krasnaya polyana", "ontario", "quebec", "russia", "south korea",
    "utah", "west vancouver",
}

GEOGRAPHIC_PHRASES = {
    "london and surrey",
    "central and southwest london and north surrey",
}

COURSE_RE = re.compile(
    r"\b(marathon to|public roads)\b",
    re.IGNORECASE,
)
GAMES_RE = re.compile(r"(\d{4})\s+(Summer|Winter)", re.IGNORECASE)
GLUED_IN_RE = re.compile(r"(?<=[a-z])in(?=[A-Z])")
CONNECTOR_WORDS = {
    "and", "of", "de", "del", "da", "do", "du", "des", "di",
    "the", "in", "at", "la", "le",
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_infobox(text):
    fields = {}
    infobox_type = ""
    in_box = False
    for line in text.splitlines()[:100]:
        if INFOBOX_START_RE.match(line):
            in_box = True
            infobox_type = line.strip()
            continue
        if not in_box:
            continue
        if line.strip() == "":
            break
        match = FIELD_RE.match(line)
        if match:
            key = match.group(1).lower()
            value = match.group(2).strip()
            if value:
                fields.setdefault(key, []).append(value)
            continue
        if line.startswith("["):
            continue
        break
    return infobox_type, fields


def normalize_games(label):
    match = GAMES_RE.search(label or "")
    if match:
        return f"{match.group(1)} {match.group(2).title()}"
    return ""


def games_label(fields, title):
    games = normalize_games((fields.get("games") or [""])[0])
    if games:
        return games
    match = TITLE_GAMES_RE.search(title or "")
    if match:
        return normalize_games(match.group(1))
    return ""


def raw_venue_entries(fields):
    if fields.get("venue"):
        return "venue", list(fields["venue"])
    if fields.get("venues"):
        return "venues", list(fields["venues"])
    if fields.get("location"):
        return "location", list(fields["location"])
    return "", []


def is_geo_name(name):
    key = normalize(name)
    if not key:
        return True
    if key in GEO_TAILS or key in HOST_CITIES or key in COUNTRIES_AND_TEAMS:
        return True
    if key.upper() in NOC_CODES and " " not in key:
        return True
    return is_country_or_team(name)


def token_is_end(token):
    key = normalize(token).rstrip("s")
    return key in VENUE_END_WORDS or normalize(token) in VENUE_END_WORDS


def has_venue_signal(name):
    return any(token_is_end(tok) for tok in name.split())


def clean_fragment(value):
    value = value.replace("\n", " ").replace("\u00a0", " ")
    value = CJK_RE.sub(" ", value)
    value = re.sub(r"\([^)]*\)", ",", value)
    value = GLUED_IN_RE.sub(" in ", value)
    value = split_camel_case(value)
    value = GLUED_DIGIT_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;./")
    return value


def strip_trailing_geo_tokens(name):
    while True:
        match = re.search(r"\s+in\s+(.+)$", name, re.IGNORECASE)
        if match and is_geo_name(match.group(1)):
            name = name[: match.start()].strip()
            continue
        break

    tokens = name.split()
    while len(tokens) >= 2:
        last = tokens[-1]
        rest = tokens[:-1]
        rest_name = " ".join(rest)
        if normalize(last) in CONNECTOR_WORDS and has_venue_signal(rest_name):
            tokens = rest
            continue
        if is_geo_name(last):
            if rest and normalize(rest[-1]) in CONNECTOR_WORDS:
                break
            if has_venue_signal(rest_name):
                tokens = rest
                continue
        if len(tokens) >= 3 and is_geo_name(" ".join(tokens[-2:])):
            leftover = tokens[:-2]
            if leftover and normalize(leftover[-1]) not in CONNECTOR_WORDS:
                tokens = leftover
                continue
        if len(tokens) >= 4 and is_geo_name(" ".join(tokens[-3:])):
            leftover = tokens[:-3]
            if leftover and normalize(leftover[-1]) not in CONNECTOR_WORDS:
                tokens = leftover
                continue
        break
    return " ".join(tokens)


def strip_trailing_geo(name):
    parts = [p.strip(" ,;") for p in re.split(r"\s*,\s*", name) if p.strip(" ,;")]
    kept = []
    for part in parts:
        part = strip_trailing_geo_tokens(part)
        if not part or is_geo_name(part):
            continue
        kept.append(part)
    return ", ".join(kept)


def strip_trailing_distance(name):
    if TRAILING_DISTANCE_RE.search(name):
        number = float(TRAILING_DISTANCE_RE.search(name).group().strip())
        if number >= 20 or not float(number).is_integer():
            name = TRAILING_DISTANCE_RE.sub("", name).strip(" ,;")
    return name


def split_concatenated(name):
    """Split only at glued CamelCase boundaries, e.g. PlaceUBC or DomeShayba."""
    parts = []
    buf = []
    for i, ch in enumerate(name):
        if i > 0 and ch.isupper() and name[i - 1].islower():
            k = i - 1
            while k >= 0 and not name[k].isupper() and name[k].isalpha():
                k -= 1
            prev_word_len = i - k if k >= 0 and name[k].isupper() else i
            if prev_word_len >= 4:
                parts.append("".join(buf).strip())
                buf = []
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return [name.strip()] if name.strip() else []

    chunks = []
    current = parts[0]
    for right in parts[1:]:
        left = current.strip()
        right_tokens = right.split()
        should_split = (
            has_venue_signal(left)
            and has_venue_signal(right)
            and len(right_tokens) >= 2
            and normalize(right_tokens[0]) not in CONNECTOR_WORDS
        )
        if should_split:
            chunks.append(left)
            current = right
        else:
            current = f"{left} {right}".strip()
    chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def split_raw_value(raw):
    prepared = raw.replace("\n", " ").replace("\u00a0", " ")
    prepared = CJK_RE.sub(" ", prepared)
    prepared = re.sub(r"\([^)]*\)", ",", prepared)
    prepared = GLUED_IN_RE.sub(" in ", prepared)
    prepared = GLUED_DIGIT_RE.sub(" ", prepared)
    names = []
    for piece in re.split(r"\s*[,;/]\s*", prepared):
        piece = piece.strip(" ,;./")
        if not piece:
            continue
        for chunk in split_concatenated(piece):
            chunk = split_camel_case(chunk)
            chunk = strip_trailing_geo_tokens(chunk)
            chunk = strip_trailing_distance(chunk)
            chunk = re.sub(r"\s+", " ", chunk).strip(" ,;./")
            if chunk and not is_geo_name(chunk):
                names.append(chunk)
    return names


def is_host_city(name):
    return normalize(name) in HOST_CITIES


def is_geographic(name):
    key = normalize(name)
    if key in GEOGRAPHIC_PHRASES:
        return True
    if key.startswith("central and southwest london"):
        return True
    if COURSE_RE.search(name):
        return True
    if " and " in key:
        left, right = key.split(" and ", 1)
        if left in HOST_CITIES and right in HOST_CITIES:
            return True
    return False


def classify_name(name):
    if re.fullmatch(r"\d+", name.strip()):
        return "venue_count"
    if not name or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", name):
        return "non_venue"
    key = normalize(name)
    if not key or len(key) < 3:
        return "non_venue"
    if key.startswith("and "):
        return "unparsed"
    if key in SPORTS:
        return "sport"
    if is_geo_name(name) or key in GEOGRAPHIC_PHRASES:
        return "country_or_city"
    if is_host_city(name) or is_geographic(name):
        return "country_or_city"
    tokens = name.split()
    if tokens and normalize(tokens[-1]) in {"village", "county", "province", "prefecture", "municipality"} and not has_venue_signal(name):
        return "country_or_city"
    return "venue"


def display_name(name, games, needs_games):
    name = re.sub(r"\s+", " ", name).strip(" ,;")
    if needs_games and games:
        return f"{name} ({games})"
    return name


def choose_id(name, games, needs_games):
    if needs_games and games:
        return make_id(f"{name} {games}")
    return make_id(name)


def main():
    event_ids = {row["event_id"] for row in load_jsonl(EVENTS_PATH)}

    parsed_events = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            event_id = doc.get("doc_id")
            if event_id not in event_ids:
                continue
            infobox_type, fields = parse_infobox(doc.get("text", ""))
            field_name, raw_values = raw_venue_entries(fields)
            parsed_events.append({
                "event_id": event_id,
                "title": doc.get("title", ""),
                "infobox_type": infobox_type,
                "games": games_label(fields, doc.get("title", "")),
                "field": field_name,
                "raw_values": raw_values,
            })

    name_games = defaultdict(set)
    extracted = []
    for event in parsed_events:
        names = []
        unresolved = []
        if not event["raw_values"]:
            unresolved.append({
                "name": "",
                "raw": "",
                "reason": "missing_venue",
            })
        for raw in event["raw_values"]:
            fragments = split_raw_value(raw)
            if not fragments:
                preview = CJK_RE.sub(" ", raw)
                preview = re.sub(r"\s+", " ", preview).strip()
                if re.fullmatch(r"\d+", raw.strip()):
                    reason = "venue_count"
                else:
                    reason = classify_name(preview or raw)
                    if reason == "venue":
                        reason = "unparsed"
                unresolved.append({"name": preview or raw, "raw": raw, "reason": reason})
                continue
            for name in fragments:
                reason = classify_name(name)
                if reason != "venue":
                    unresolved.append({"name": name, "raw": raw, "reason": reason})
                    continue
                names.append(name)
                if event["games"]:
                    name_games[normalize(name)].add(event["games"])
        extracted.append((event, names, unresolved))

    multi_games_names = {
        key for key, games in name_games.items() if len(games) > 1
    }

    venues = {}
    held_at = {}
    unresolved_rows = []
    audit_rows = []
    reason_counts = Counter()

    for event, names, unresolved in extracted:
        resolved = []
        for name in names:
            needs_games = normalize(name) in multi_games_names
            venue_id = choose_id(name, event["games"], needs_games)
            if not venue_id:
                unresolved.append({
                    "name": name,
                    "raw": name,
                    "reason": "unparsed",
                })
                continue
            label = display_name(name, event["games"], needs_games)
            current = venues.get(venue_id)
            if current is None or len(label) < len(current["name"]):
                venues[venue_id] = {
                    "venue_id": venue_id,
                    "name": label,
                }
            held_at[(event["event_id"], venue_id)] = {
                "event_id": event["event_id"],
                "venue_id": venue_id,
                "venue_name": venues[venue_id]["name"],
            }
            resolved.append({
                "venue_id": venue_id,
                "venue_name": venues[venue_id]["name"],
                "source_name": name,
            })

        for item in unresolved:
            reason_counts[item["reason"]] += 1
            unresolved_rows.append({
                "event_id": event["event_id"],
                "raw_venue": item["raw"],
                "parsed_name": item["name"],
                "reason": item["reason"],
            })

        audit_rows.append({
            "event_id": event["event_id"],
            "title": event["title"],
            "games": event["games"],
            "field": event["field"],
            "raw_values": event["raw_values"],
            "resolved": resolved,
            "unresolved": unresolved,
        })

    venue_rows = sorted(venues.values(), key=lambda row: row["name"].casefold())
    held_at_rows = sorted(
        held_at.values(),
        key=lambda row: (row["event_id"], row["venue_id"]),
    )

    VENUES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with VENUES_JSONL.open("w", encoding="utf-8") as f:
        for row in venue_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with VENUES_TSV.open("w", encoding="utf-8") as f:
        for row in venue_rows:
            f.write(row["venue_id"] + "\t" + row["name"].replace("\t", " ") + "\n")

    with HELD_AT_PATH.open("w", encoding="utf-8") as f:
        for row in held_at_rows:
            f.write(row["event_id"] + "\t" + row["venue_id"] + "\n")

    with UNRESOLVED_PATH.open("w", encoding="utf-8") as f:
        f.write("event_id\traw_venue\tparsed_name\treason\n")
        for row in unresolved_rows:
            f.write(
                f"{row['event_id']}\t{row['raw_venue']}\t"
                f"{row['parsed_name']}\t{row['reason']}\n"
            )

    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    events_with_edge = {row["event_id"] for row in held_at_rows}
    unknown_events = [
        row["event_id"] for row in held_at_rows if row["event_id"] not in event_ids
    ]
    unknown_venues = [
        row["venue_id"] for row in held_at_rows if row["venue_id"] not in venues
    ]

    print("VENUE / HELD_AT EXTRACTION COMPLETE")
    print("=" * 70)
    print("Valid Olympic events processed:", len(event_ids))
    print("Unique venues:", len(venue_rows))
    print("Event → Venue relationships:", len(held_at_rows))
    print("Events with at least one HELD_AT:", len(events_with_edge))
    print("Events without HELD_AT:", len(event_ids) - len(events_with_edge))
    print("Unresolved rows:", len(unresolved_rows))
    print("Unresolved reasons:", dict(reason_counts))
    print("Names disambiguated by Games:", len(multi_games_names))
    print("Held-at event IDs missing from event catalog:", len(unknown_events))
    print("Held-at venue IDs missing from venue catalog:", len(unknown_venues))
    print("Venues:", VENUES_JSONL)
    print("Edges:", HELD_AT_PATH)

    print("\nSAMPLE VENUES")
    for row in venue_rows[:15]:
        print(f"{row['venue_id']}\t{row['name']}")

    print("\nSAMPLE HELD_AT")
    for row in held_at_rows[:12]:
        print(f"{row['event_id']}\t{row['venue_id']}\t{row['venue_name']}")

    print("\nDISAMBIGUATED GENERIC NAMES")
    for key in sorted(multi_games_names):
        print(f"{key}: {sorted(name_games[key])}")


if __name__ == "__main__":
    main()
