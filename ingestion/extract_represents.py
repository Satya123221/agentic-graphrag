"""Extract Athlete → Country REPRESENTS candidates from corpus infoboxes.

Does not load TigerGraph. Uses only valid Olympic events and the athlete catalog.
NOC codes are never stored as countries; they are mapped to country names only
when this corpus provides enough evidence.
"""

import json
import re
import sys
from collections import Counter, defaultdict

sys.path.append(".")

from config import CORPUS_PATH, DATA_DIR
from ingestion.extract_venues import HOST_CITIES
from ingestion.extract_won_by import index_athletes, load_jsonl, resolve_name
from ingestion.name_utils import (
    COUNTRIES_AND_TEAMS,
    is_country_or_team,
    is_non_person_value,
    looks_like_person_name,
    make_id,
    normalize,
    split_medal_value,
)

EVENTS_PATH = DATA_DIR / "graph_data" / "events.jsonl"
ATHLETES_PATH = DATA_DIR / "graph_data" / "athletes.jsonl"
COUNTRIES_JSONL = DATA_DIR / "graph_data" / "countries.jsonl"
COUNTRIES_TSV = DATA_DIR / "graph_data" / "countries.tsv"
REPRESENTS_PATH = DATA_DIR / "graph_data" / "athlete_countries.tsv"
UNRESOLVED_PATH = DATA_DIR / "graph_data" / "represents_unresolved.tsv"
AUDIT_PATH = DATA_DIR / "graph_data" / "represents_extraction.jsonl"

INFOBOX_START_RE = re.compile(r"^\[infobox", re.IGNORECASE)
FIELD_RE = re.compile(r"^\s+([A-Za-z0-9_]+):\s*(.*)$")
LINK_RE = re.compile(
    r"^(?:\s+[Aa]nd\s+[A-ZÀ-ÖØ-öø-ÿ][\w.'\-]*"
    r"(?:\s+(?:van|von|de|den|der|di|da|del|la|le|bin|al|"
    r"[A-ZÀ-ÖØ-öø-ÿ][\w.'\-]*)){0,3})?"
    r"\s+(?:from|of the|of|representing)\s+"
)
PAREN_COUNTRY_RE = re.compile(r"^\s*\(([^)]{3,40})\)")

MEDAL_SLOTS = (
    ("gold", "goldnoc"),
    ("gold2", "goldnoc2"),
    ("silver", "silvernoc"),
    ("silver2", "silvernoc2"),
    ("silver3", "silvernoc3"),
    ("bronze", "bronzenoc"),
    ("bronze2", "bronzenoc2"),
)

# Olympic designations that are teams / independent athletes, not countries.
TEAM_NOCS = {"EUN", "ROC", "OAR", "IOA", "IOP", "MIX", "AIN", "ANZ", "EUA"}

COUNTRY_DISPLAY = {
    "united states": "United States",
    "united kingdom": "United Kingdom",
    "great britain": "Great Britain",
    "czech republic": "Czech Republic",
    "south korea": "South Korea",
    "north korea": "North Korea",
    "south africa": "South Africa",
    "new zealand": "New Zealand",
    "east germany": "East Germany",
    "west germany": "West Germany",
    "soviet union": "Soviet Union",
    "chinese taipei": "Chinese Taipei",
    "the netherlands": "Netherlands",
    "netherlands": "Netherlands",
    "hong kong": "Hong Kong",
    "puerto rico": "Puerto Rico",
    "costa rica": "Costa Rica",
    "saudi arabia": "Saudi Arabia",
    "ivory coast": "Ivory Coast",
    "trinidad and tobago": "Trinidad and Tobago",
    "dominican republic": "Dominican Republic",
    "united arab emirates": "United Arab Emirates",
    "serbia and montenegro": "Serbia and Montenegro",
}

# Extra country names attested as athlete nationalities in this Olympic corpus.
# Used only as a recognition whitelist; a country vertex is created only when
# a name from this set is actually matched in corpus text.
EXTRA_COUNTRIES = {
    "algeria", "argentina", "armenia", "azerbaijan", "bahamas", "bahrain",
    "bermuda", "chile", "colombia", "costa rica", "dominican republic",
    "ecuador", "fiji", "georgia", "grenada", "guatemala", "indonesia",
    "ivory coast", "jordan", "kosovo", "kyrgyzstan", "luxembourg",
    "malaysia", "mongolia", "morocco", "mozambique", "panama", "peru",
    "philippines", "qatar", "romania", "saudi arabia", "serbia and montenegro",
    "singapore", "suriname", "tajikistan", "thailand", "trinidad and tobago",
    "tunisia", "uganda", "united arab emirates", "uzbekistan", "venezuela",
    "vietnam", "zimbabwe", "bohemia", "taiwan",
}

CANON_ALIASES = {
    "the netherlands": "netherlands",
    "holland": "netherlands",
    "the united states": "united states",
    "united states of america": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "ussr": "soviet union",
    "the soviet union": "soviet union",
    "the united kingdom": "great britain",
    "united kingdom": "great britain",
    "the czech republic": "czech republic",
    "the dominican republic": "dominican republic",
    "republic of korea": "south korea",
    "the republic of korea": "south korea",
    "people's republic of china": "china",
    "republic of china": "chinese taipei",
    "chinese taipei": "chinese taipei",
    "taiwan": "chinese taipei",
}

CITY_TAILS = (
    " city", " province", " county", " state", " island", " islands",
)

# Host cities must never become Country vertices. Hong Kong is an Olympic NOC.
CITY_DENY = (HOST_CITIES | {"surrey"}) - {"hong kong"}


def country_display(norm):
    if norm in COUNTRY_DISPLAY:
        return COUNTRY_DISPLAY[norm]
    return " ".join(part.capitalize() for part in norm.split())


def country_name_set():
    names = set(COUNTRIES_AND_TEAMS) | set(EXTRA_COUNTRIES)
    names.discard("roc")
    names.discard("unified team")
    names.discard("united team")
    names.discard("olympic athletes from russia")
    names.discard("ussr")
    return names


def canonicalize_country(norm):
    norm = (norm or "").strip()
    if not norm:
        return None
    if norm in CANON_ALIASES:
        return CANON_ALIASES[norm]
    if norm.startswith("the ") and norm[4:] in COUNTRY_NORMS:
        return norm[4:]
    if norm in COUNTRY_NORMS:
        return norm
    return None


def build_aliases():
    aliases = {}
    for norm in country_name_set():
        canon = canonicalize_country(norm) or norm
        if canon in {"unified team", "united team", "olympic athletes from russia"}:
            continue
        aliases[norm] = canon
        aliases[country_display(norm).lower()] = canon
        aliases[country_display(canon).lower()] = canon
    for alias, canon in CANON_ALIASES.items():
        aliases[alias] = canon
    aliases["the netherlands"] = "netherlands"
    aliases["netherlands"] = "netherlands"
    return tuple(
        sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True)
    )


COUNTRY_NORMS = country_name_set()
COUNTRY_ALIASES = build_aliases()


def is_city_name(norm):
    return norm in CITY_DENY


def match_country_prefix(text):
    if not text:
        return None
    low = text.lower().replace("’", "'").replace("‘", "'")
    low = low.lstrip(" \t")
    for alias, canon in COUNTRY_ALIASES:
        if not low.startswith(alias):
            continue
        end = len(alias)
        nxt = low[end:end + 1]
        if nxt and nxt.isalpha():
            continue
        rest = low[end:]
        if any(rest.startswith(tail) for tail in CITY_TAILS):
            continue
        if is_city_name(canon) or is_city_name(alias):
            continue
        if canon not in COUNTRY_NORMS and canon not in {
            "netherlands", "soviet union", "chinese taipei",
        }:
            continue
        return canon
    return None


def match_country_suffix(text):
    if not text:
        return None
    low = text.lower().replace("’", "'").replace("‘", "'").rstrip(" \t,;:-")
    for alias, canon in COUNTRY_ALIASES:
        if not low.endswith(alias):
            continue
        start = len(low) - len(alias)
        prev = low[start - 1:start]
        if prev and prev.isalpha():
            continue
        if is_city_name(canon) or is_city_name(alias):
            continue
        return canon
    return None


def name_spans(text, name):
    if not text or not name:
        return
    hay = text.lower()
    needle = name.lower()
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return
        prev = hay[i - 1] if i else ""
        nxt = hay[i + len(needle)] if i + len(needle) < len(hay) else ""
        if (not prev or not prev.isalpha()) and (not nxt or not nxt.isalpha()):
            yield i, i + len(name)
        start = i + 1


def match_country_for_name(text, person_name):
    if not text or not person_name:
        return None
    hits = []
    for start, end in name_spans(text, person_name):
        after = text[end:]
        link = LINK_RE.match(after)
        if link:
            country = match_country_prefix(after[link.end():])
            if country:
                hits.append(country)
        paren = PAREN_COUNTRY_RE.match(after)
        if paren:
            country = match_country_prefix(paren.group(1).strip())
            if country:
                hits.append(country)
        before = text[:start]
        chunk = before[-120:].replace("’", "'").replace("‘", "'")
        stripped = chunk.rstrip()
        if stripped.endswith("'s") or stripped.endswith("'"):
            stem = stripped[:-2] if stripped.endswith("'s") else stripped[:-1]
            country = match_country_suffix(stem)
            if country:
                hits.append(country)
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    return None


def parse_infobox(text):
    fields = {}
    in_box = False
    for line in text.splitlines()[:100]:
        if INFOBOX_START_RE.match(line):
            in_box = True
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
    return fields


def body_text(text):
    lines = text.splitlines()
    start = 0
    in_box = False
    for i, line in enumerate(lines[:120]):
        if INFOBOX_START_RE.match(line):
            in_box = True
            continue
        if in_box and line.strip() == "":
            start = i + 1
            break
    body = " ".join(line.strip() for line in lines[start:] if line.strip())
    return re.sub(r"\s+", " ", body)[:15000]


def classify_noc(raw):
    value = (raw or "").strip()
    if not value:
        return "missing_noc", ""
    if re.fullmatch(r"\d+", value):
        return "non_country", value
    compact = value.upper().replace(".", "")
    if compact in TEAM_NOCS:
        return "team_or_noc", compact
    if re.fullmatch(r"[A-Z]{3}", compact):
        return "noc", compact
    key = canonicalize_country(normalize(value))
    if key:
        return "country", key
    if is_country_or_team(value):
        return "country", normalize(value)
    return "unparsed_noc", value


def choose_mapping(votes):
    mapping = {}
    for noc, counts in votes.items():
        total = sum(counts.values())
        if total < 2:
            continue
        country, n = counts.most_common(1)[0]
        second = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        if n >= 2 and n / total >= 0.75 and n > second:
            mapping[noc] = country
    return mapping


def person_names_from_medal(raw_medal):
    if not raw_medal or is_non_person_value(raw_medal) or is_country_or_team(raw_medal):
        return []
    names = split_medal_value(raw_medal)
    return [n for n in names if looks_like_person_name(n)] or names


def main():
    event_ids = {row["event_id"] for row in load_jsonl(EVENTS_PATH)}
    athletes = load_jsonl(ATHLETES_PATH)
    catalog_ids = {a["athlete_id"] for a in athletes}
    index = index_athletes(athletes)

    parsed = []
    noc_votes = defaultdict(Counter)
    host_country_values = Counter()

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            event_id = doc.get("doc_id")
            if event_id not in event_ids:
                continue
            text = doc.get("text", "")
            fields = parse_infobox(text)
            body = body_text(text)
            for val in fields.get("country") or []:
                host_country_values[val] += 1
            slots = []
            for medal_key, noc_key in MEDAL_SLOTS:
                medal_vals = fields.get(medal_key) or []
                noc_vals = fields.get(noc_key) or []
                if not medal_vals and not noc_vals:
                    continue
                noc_raw = noc_vals[0] if noc_vals else ""
                for raw_medal in medal_vals or [""]:
                    slots.append((medal_key, raw_medal, noc_raw))
                    kind, noc = classify_noc(noc_raw)
                    if kind != "noc" or not raw_medal:
                        continue
                    if is_country_or_team(raw_medal) or is_non_person_value(raw_medal):
                        continue
                    for name in person_names_from_medal(raw_medal):
                        country = match_country_for_name(body, name)
                        if country:
                            noc_votes[noc][country] += 1
            parsed.append({
                "event_id": event_id,
                "title": doc.get("title", ""),
                "slots": slots,
                "body": body,
            })

    noc_to_country = choose_mapping(noc_votes)

    countries = {}
    represents = {}
    unresolved_rows = []
    audit_rows = []
    reason_counts = Counter()
    conflicts = []

    def add_country(norm):
        name = country_display(norm)
        country_id = make_id(name)
        countries[country_id] = {"country_id": country_id, "name": name}
        return country_id, name

    for event in parsed:
        resolved = []
        unresolved = []
        if not event["slots"]:
            unresolved.append({
                "name": "",
                "noc": "",
                "reason": "missing_noc",
                "raw": "",
            })
        for medal_key, raw_medal, noc_raw in event["slots"]:
            kind, noc_key = classify_noc(noc_raw)
            if is_non_person_value(raw_medal) or is_country_or_team(raw_medal):
                unresolved.append({
                    "name": raw_medal,
                    "noc": noc_raw,
                    "reason": "team_or_country_medal",
                    "raw": raw_medal,
                })
                continue
            if kind == "missing_noc":
                unresolved.append({
                    "name": raw_medal,
                    "noc": "",
                    "reason": "missing_noc",
                    "raw": raw_medal,
                })
                continue
            if kind == "team_or_noc":
                unresolved.append({
                    "name": raw_medal,
                    "noc": noc_key,
                    "reason": "team_or_noc",
                    "raw": raw_medal,
                })
                continue
            if kind not in {"noc", "country"}:
                unresolved.append({
                    "name": raw_medal,
                    "noc": noc_raw,
                    "reason": kind,
                    "raw": raw_medal,
                })
                continue

            names = split_medal_value(raw_medal)
            if not names:
                unresolved.append({
                    "name": raw_medal,
                    "noc": noc_key,
                    "reason": "unparsed_medal",
                    "raw": raw_medal,
                })
                continue

            for name in names:
                athlete = resolve_name(name, index)
                if not athlete:
                    reason = (
                        "unresolved_person"
                        if looks_like_person_name(name)
                        else "non_athlete"
                    )
                    unresolved.append({
                        "name": name,
                        "noc": noc_key,
                        "reason": reason,
                        "raw": raw_medal,
                    })
                    continue

                if kind == "country":
                    local = noc_key
                    mapped = noc_key
                else:
                    local = match_country_for_name(event["body"], name)
                    if not local:
                        local = match_country_for_name(event["body"], athlete["name"])
                    mapped = noc_to_country.get(noc_key)

                if local and mapped and local != mapped:
                    unresolved.append({
                        "name": name,
                        "noc": noc_key,
                        "reason": "conflicting_country",
                        "raw": raw_medal,
                    })
                    conflicts.append({
                        "athlete_id": athlete["athlete_id"],
                        "name": name,
                        "noc": noc_key,
                        "local": local,
                        "mapped": mapped,
                        "event_id": event["event_id"],
                    })
                    continue

                country_norm = local or mapped
                if not country_norm:
                    unresolved.append({
                        "name": name,
                        "noc": noc_key or noc_raw,
                        "reason": "unmapped_noc",
                        "raw": raw_medal,
                    })
                    continue
                if country_norm in {
                    "unified team",
                    "united team",
                    "olympic athletes from russia",
                } or is_city_name(country_norm):
                    unresolved.append({
                        "name": name,
                        "noc": noc_key,
                        "reason": "team_or_noc",
                        "raw": raw_medal,
                    })
                    continue

                country_id, country_name = add_country(country_norm)
                key = (athlete["athlete_id"], country_id)
                represents[key] = {
                    "athlete_id": athlete["athlete_id"],
                    "country_id": country_id,
                    "athlete_name": athlete["name"],
                    "country_name": country_name,
                }
                resolved.append({
                    "athlete_id": athlete["athlete_id"],
                    "athlete_name": athlete["name"],
                    "country_id": country_id,
                    "country_name": country_name,
                    "medal": medal_key,
                    "noc": noc_key,
                    "source_name": name,
                    "via": "local" if local else "noc_map",
                })

        for item in unresolved:
            reason_counts[item["reason"]] += 1
            unresolved_rows.append({
                "event_id": event["event_id"],
                "raw_medal": item["raw"],
                "parsed_name": item["name"],
                "noc": item["noc"],
                "reason": item["reason"],
            })
        audit_rows.append({
            "event_id": event["event_id"],
            "title": event["title"],
            "resolved": resolved,
            "unresolved": unresolved,
        })

    country_rows = sorted(countries.values(), key=lambda r: r["name"].casefold())
    rel_rows = sorted(
        represents.values(),
        key=lambda r: (r["athlete_id"], r["country_id"]),
    )

    COUNTRIES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with COUNTRIES_JSONL.open("w", encoding="utf-8") as f:
        for row in country_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with COUNTRIES_TSV.open("w", encoding="utf-8") as f:
        for row in country_rows:
            f.write(row["country_id"] + "\t" + row["name"].replace("\t", " ") + "\n")
    with REPRESENTS_PATH.open("w", encoding="utf-8") as f:
        f.write("athlete_id\tcountry_id\tathlete_name\tcountry_name\n")
        for row in rel_rows:
            f.write(
                f"{row['athlete_id']}\t{row['country_id']}\t"
                f"{row['athlete_name']}\t{row['country_name']}\n"
            )
    with UNRESOLVED_PATH.open("w", encoding="utf-8") as f:
        f.write("event_id\traw_medal\tparsed_name\tnoc\treason\n")
        for row in unresolved_rows:
            f.write(
                f"{row['event_id']}\t{row['raw_medal']}\t{row['parsed_name']}\t"
                f"{row['noc']}\t{row['reason']}\n"
            )
    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    unknown_athletes = [
        r["athlete_id"] for r in rel_rows if r["athlete_id"] not in catalog_ids
    ]
    unknown_countries = [
        r["country_id"] for r in rel_rows if r["country_id"] not in countries
    ]
    noc_like_countries = [
        r for r in country_rows
        if re.fullmatch(r"[A-Za-z]{3}", r["name"]) and r["name"].upper() in TEAM_NOCS
        or r["name"].upper() in TEAM_NOCS
    ]
    city_countries = [
        r for r in country_rows if normalize(r["name"]) in CITY_DENY
    ]
    multi = defaultdict(list)
    for row in rel_rows:
        multi[row["athlete_id"]].append(row["country_name"])
    multi_athletes = {k: v for k, v in multi.items() if len(v) > 1}

    print("REPRESENTS EXTRACTION COMPLETE")
    print("=" * 70)
    print("Unique countries:", len(country_rows))
    print("Athlete → Country relationships:", len(rel_rows))
    print("Athletes with a country:", len({r["athlete_id"] for r in rel_rows}))
    print("Unresolved rows:", len(unresolved_rows))
    print("Unresolved reasons:", dict(reason_counts))
    print("NOC mappings accepted:", len(noc_to_country))
    print("Host infobox country: values (not used):", host_country_values.most_common(8))
    print("Unknown athlete IDs:", len(unknown_athletes))
    print("Unknown country IDs:", len(unknown_countries))
    print("NOC/team-like country vertices:", noc_like_countries)
    print("City-like country vertices:", city_countries)
    print("Athletes with multiple countries:", len(multi_athletes))
    print("Local vs mapped conflicts skipped:", len(conflicts))
    print("Countries:", COUNTRIES_JSONL)
    print("Edges:", REPRESENTS_PATH)

    print("\nNOC → COUNTRY MAPPING")
    for noc, country in sorted(noc_to_country.items(), key=lambda kv: kv[0]):
        print(f"{noc}\t{country_display(country)}\t{dict(noc_votes[noc])}")

    print("\nUNMAPPED NOCs WITH VOTES < THRESHOLD")
    for noc, counts in sorted(noc_votes.items()):
        if noc not in noc_to_country:
            print(f"{noc}\t{dict(counts)}")

    print("\nCOUNTRY LIST")
    for row in country_rows:
        print(f"{row['country_id']}\t{row['name']}")

    print("\nSAMPLE RELATIONSHIPS")
    wanted = {
        "rudolf_dombi", "roland_kokeny", "kristin_armstrong", "dayron_robles",
        "sven_kramer", "aaron_egbele", "simone_biles", "felix_sanchez",
    }
    for row in rel_rows:
        if row["athlete_id"] in wanted:
            print(f"{row['athlete_id']}\t{row['athlete_name']}\t{row['country_name']}")
    for row in rel_rows[:12]:
        print(f"{row['athlete_id']}\t{row['athlete_name']}\t{row['country_name']}")

    if multi_athletes:
        print("\nMULTI-COUNTRY ATHLETES (up to 15)")
        for athlete_id, names in list(sorted(multi_athletes.items()))[:15]:
            print(athlete_id, names)


if __name__ == "__main__":
    main()
