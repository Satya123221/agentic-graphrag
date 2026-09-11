"""GraphRAG retrieval over TigerGraph relationships and corpus text.

Does not modify the graph. Does not use the TF-IDF RAG baseline.
"""

import json
import re
import sys
from collections import OrderedDict

sys.path.append(".")

from config import CORPUS_PATH
from rag.graph_client import (
    events_held_at_venue,
    fetch_edges,
    fetch_vertex,
    fetch_vertices,
)

STOPWORDS = {
    "a", "an", "and", "at", "according", "before", "corpus", "did", "event",
    "events", "for", "gold", "held", "how", "immediately", "in", "just",
    "many", "medal", "of", "olympics", "olympic", "on", "prior", "provided",
    "question", "the", "to", "was", "what", "which", "who", "won", "year",
}

EVIDENCE_CHARS = 1500
DEFAULT_TOP_K = 8

YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
BEFORE_RE = re.compile(
    r"(?:immediately|just)?\s*(?:before|prior to)\s+(?:the\s+)?(\d{4})",
    re.IGNORECASE,
)
AFTER_RE = re.compile(
    r"(?:immediately|just)?\s*(?:after|following)\s+(?:the\s+)?(\d{4})",
    re.IGNORECASE,
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
MONTH_NAMES = {n: name for name, n in MONTHS.items()}
DATE_DMY_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
DATE_MDY_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{1,2})(?:,)?\s+(\d{4})\b",
    re.IGNORECASE,
)

_corpus = None


def _normalize(text):
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    text = text.replace("kilometres", "kilometer").replace("kilometers", "kilometer")
    text = re.sub(r"\bkm\b", "kilometer", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text):
    return [tok for tok in _normalize(text).split() if tok and tok not in STOPWORDS]


def load_corpus():
    global _corpus
    if _corpus is None:
        docs = {}
        with CORPUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                docs[row["doc_id"]] = {
                    "doc_id": row["doc_id"],
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
        _corpus = docs
    return _corpus


def _parse_games_vertex(vertex):
    name = vertex.get("name") or ""
    match = re.match(r"^(\d{4})\s+(Summer|Winter)$", name.strip(), re.IGNORECASE)
    if not match:
        return None
    return {
        "id": vertex["id"],
        "name": name,
        "year": int(match.group(1)),
        "season": match.group(2).lower(),
    }


def _detect_season(question):
    q = _normalize(question)
    if "winter" in q.split():
        return "winter"
    if "summer" in q.split():
        return "summer"
    return None


def _detect_gender(question):
    q = _normalize(question)
    if re.search(r"\bwomen s\b|\bwomen\b|\bfemale\b", q):
        return "women"
    if re.search(r"\bmen s\b|\bmen\b|\bmale\b", q):
        return "men"
    return None


def _match_named_vertices(question, vertices, min_len=4):
    q = _normalize(question)
    hits = []
    for vertex in vertices:
        name = _normalize(vertex["name"])
        if len(name) < min_len:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", q):
            hits.append(vertex)
    hits.sort(key=lambda v: (-len(v["name"]), v["id"]))
    return hits


def _parse_question_dates(question):
    dates = []
    for match in DATE_DMY_RE.finditer(question):
        dates.append({
            "day": int(match.group(1)),
            "month": MONTHS[match.group(2).lower()],
            "year": int(match.group(3)),
        })
    for match in DATE_MDY_RE.finditer(question):
        dates.append({
            "day": int(match.group(2)),
            "month": MONTHS[match.group(1).lower()],
            "year": int(match.group(3)),
        })
    uniq = []
    seen = set()
    for row in dates:
        key = (row["year"], row["month"], row["day"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def _drop_subsumed_venues(hits):
    kept = []
    norms = [(_normalize(v["name"]), v) for v in hits]
    for name, vertex in norms:
        if any(name != other and name in other for other, _ in norms):
            continue
        kept.append(vertex)
    return kept


def _date_match_score(event, dates):
    if not dates:
        return 0.0
    corpus = load_corpus()
    row = corpus.get(event["id"]) or {}
    blob = _normalize((row.get("text") or "")[:2500] + " " + (event.get("name") or ""))
    score = 0.0
    for item in dates:
        month_name = MONTH_NAMES[item["month"]]
        day = str(item["day"])
        year = str(item["year"])
        day_month = (
            re.search(rf"\b{re.escape(day)} {re.escape(month_name)}\b", blob)
            or re.search(rf"\b{re.escape(month_name)} {re.escape(day)}\b", blob)
        )
        year_hit = re.search(rf"\b{re.escape(year)}\b", blob)
        if day_month:
            score += 0.8
            if year_hit:
                score += 0.2
        elif year_hit:
            score += 0.15
    return score


def match_entities(question):
    games = [_parse_games_vertex(v) for v in fetch_vertices("OlympicGames")]
    games = [g for g in games if g]
    sports = fetch_vertices("Sport")
    venues = fetch_vertices("Venue")
    countries = fetch_vertices("Country")

    season = _detect_season(question)
    years = [int(y) for y in YEAR_RE.findall(question)]
    before = BEFORE_RE.search(question)
    after = AFTER_RE.search(question)
    dates = _parse_question_dates(question)

    matched_games = []
    for game in games:
        if years and game["year"] in years:
            if season is None or game["season"] == season:
                matched_games.append(game)
    if years and season is None:
        matched_games = [g for g in games if g["year"] in years]

    target_games = list(matched_games)
    temporal = None
    if before and matched_games:
        temporal = "previous_same_season"
        for anchor in matched_games:
            earlier = [
                g for g in games
                if g["season"] == anchor["season"] and g["year"] < anchor["year"]
            ]
            earlier.sort(key=lambda g: (-g["year"], g["id"]))
            if earlier:
                target_games.append(earlier[0])
    elif after and matched_games:
        temporal = "next_same_season"
        for anchor in matched_games:
            later = [
                g for g in games
                if g["season"] == anchor["season"] and g["year"] > anchor["year"]
            ]
            later.sort(key=lambda g: (g["year"], g["id"]))
            if later:
                target_games.append(later[0])

    uniq_games = list(OrderedDict((g["id"], g) for g in target_games).values())
    sport_hits = _match_named_vertices(question, sports, min_len=4)
    venue_hits = _drop_subsumed_venues(
        _match_named_vertices(question, venues, min_len=8)
    )
    country_hits = _match_named_vertices(question, countries, min_len=4)

    return {
        "question_tokens": _tokens(question),
        "gender": _detect_gender(question),
        "season": season,
        "years": years,
        "dates": dates,
        "temporal": temporal,
        "games": uniq_games,
        "anchor_games": matched_games,
        "sports": sport_hits,
        "venues": venue_hits,
        "countries": country_hits,
    }


def _event_token_score(event_name, question_tokens, gender):
    name_toks = set(_tokens(event_name))
    q_toks = [t for t in question_tokens if t not in {"summer", "winter"}]
    if not q_toks:
        return 0.0, 0
    overlap = [t for t in q_toks if t in name_toks]
    score = len(overlap) / len(q_toks)
    if gender == "men" and "women" in name_toks:
        return 0.0, len(overlap)
    if gender == "women" and "men" in name_toks and "women" not in name_toks:
        return 0.0, len(overlap)
    if gender and gender in name_toks:
        score += 0.15
    return score, len(overlap)


def select_events(entities, top_k=DEFAULT_TOP_K):
    events = fetch_vertices("Event")
    events_by_id = {event["id"]: event for event in events}
    paths = []
    games_ids = {g["id"] for g in entities["games"]}
    sport_ids = {s["id"] for s in entities["sports"]}
    target_ids = {
        g["id"] for g in entities["games"]
        if g["id"] not in {a["id"] for a in entities["anchor_games"]}
    }
    anchor_ids = {g["id"] for g in entities["anchor_games"]}
    venue_ids = [v["id"] for v in entities["venues"]]
    dates = entities.get("dates") or []

    for gid in sorted(games_ids):
        paths.append({
            "path": "Event-(PART_OF)->OlympicGames",
            "from_id": None,
            "to_id": gid,
        })
    for sid in sorted(sport_ids):
        paths.append({
            "path": "Event-(IN_SPORT)->Sport",
            "from_id": None,
            "to_id": sid,
        })

    candidates = {}

    def add_candidate(event, token_score, overlap, from_venue):
        eid = event["id"]
        prev = candidates.get(eid)
        if prev is None:
            candidates[eid] = {
                "token_score": token_score,
                "overlap": overlap,
                "event": event,
                "from_venue": from_venue,
            }
            return
        prev["from_venue"] = prev["from_venue"] or from_venue
        if token_score > prev["token_score"]:
            prev["token_score"] = token_score
            prev["overlap"] = overlap

    lexical = []
    for event in events:
        token_score, overlap = _event_token_score(
            event["name"], entities["question_tokens"], entities["gender"]
        )
        if overlap < 2 and token_score < 0.5:
            continue
        lexical.append((token_score, overlap, event["id"], event))
    lexical.sort(key=lambda row: (-row[0], -row[1], row[2]))
    for token_score, overlap, eid, event in lexical[: max(top_k * 6, 24)]:
        add_candidate(event, token_score, overlap, False)

    venue_event_links = []
    for vid in venue_ids:
        paths.append({
            "path": "Venue<-(HELD_AT)-Event",
            "from_id": vid,
            "to_id": None,
        })
        for eid in events_held_at_venue(vid):
            event = events_by_id.get(eid)
            if not event:
                continue
            token_score, overlap = _event_token_score(
                event["name"], entities["question_tokens"], entities["gender"]
            )
            add_candidate(event, token_score, overlap, True)
            venue_event_links.append((vid, eid))

    ranked = []
    event_games = {}
    event_sport = {}
    date_scores = {}
    for eid, row in candidates.items():
        event = row["event"]
        part = fetch_edges("Event", eid, "PART_OF")
        sport = fetch_edges("Event", eid, "IN_SPORT")
        gid = part[0]["to_id"] if part else None
        sid = sport[0]["to_id"] if sport else None
        event_games[eid] = gid
        event_sport[eid] = sid
        from_venue = row["from_venue"]
        token_score = row["token_score"]
        overlap = row["overlap"]
        dscore = _date_match_score(event, dates) if dates else 0.0
        date_scores[eid] = dscore
        score = token_score
        if from_venue:
            score += 0.35
            score += dscore
        if gid and gid in target_ids:
            score += 0.5
        elif gid and gid in anchor_ids:
            score += 0.25
        elif gid and gid in games_ids:
            score += 0.4
        if sid and sid in sport_ids:
            score += 0.3
        if sport_ids and sid not in sport_ids:
            continue
        if target_ids and gid not in games_ids:
            continue
        if from_venue and games_ids and gid not in games_ids:
            continue
        ranked.append((score, overlap, eid, event, gid, sid, from_venue, dscore))

    if dates:
        dated_venue = [row for row in ranked if row[6] and row[7] >= 0.8]
        if dated_venue:
            ranked = dated_venue

    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected = [
        (score, overlap, eid, event, gid, sid)
        for score, overlap, eid, event, gid, sid, _from_venue, _dscore in ranked[:top_k]
    ]
    selected_ids = {row[2] for row in selected}
    for vid, eid in venue_event_links:
        if eid in selected_ids:
            paths.append({
                "path": "Venue<-(HELD_AT)-Event",
                "from_id": vid,
                "to_id": eid,
            })
    return selected, paths, event_games, event_sport


def expand_event(event, event_games, event_sport):
    eid = event["id"]
    traversals = []
    games_edges = fetch_edges("Event", eid, "PART_OF")
    sport_edges = fetch_edges("Event", eid, "IN_SPORT")
    won_by = fetch_edges("Event", eid, "WON_BY")
    held_at = fetch_edges("Event", eid, "HELD_AT")
    describes = fetch_edges("Document", eid, "DESCRIBES")
    athletes = []
    countries = []
    for edge in won_by:
        traversals.append({
            "path": "Event-(WON_BY)->Athlete",
            "from_id": eid,
            "to_id": edge["to_id"],
        })
        athlete = fetch_vertex("Athlete", edge["to_id"])
        if athlete:
            athletes.append(athlete)
            for cre in fetch_edges("Athlete", athlete["id"], "REPRESENTS"):
                traversals.append({
                    "path": "Athlete-(REPRESENTS)->Country",
                    "from_id": athlete["id"],
                    "to_id": cre["to_id"],
                })
                country = fetch_vertex("Country", cre["to_id"])
                if country:
                    countries.append(country)
    venues = []
    for edge in held_at:
        traversals.append({
            "path": "Event-(HELD_AT)->Venue",
            "from_id": eid,
            "to_id": edge["to_id"],
        })
        venue = fetch_vertex("Venue", edge["to_id"])
        if venue:
            venues.append(venue)
    for edge in games_edges:
        traversals.append({
            "path": "Event-(PART_OF)->OlympicGames",
            "from_id": eid,
            "to_id": edge["to_id"],
        })
    for edge in sport_edges:
        traversals.append({
            "path": "Event-(IN_SPORT)->Sport",
            "from_id": eid,
            "to_id": edge["to_id"],
        })
    documents = []
    for edge in describes:
        if edge["to_id"] == eid:
            traversals.append({
                "path": "Document-(DESCRIBES)->Event",
                "from_id": edge["from_id"],
                "to_id": eid,
            })
            doc = fetch_vertex("Document", edge["from_id"])
            if doc:
                documents.append(doc)
    if not documents:
        doc = fetch_vertex("Document", eid)
        if doc:
            documents.append(doc)
            traversals.append({
                "path": "Document-(DESCRIBES)->Event",
                "from_id": eid,
                "to_id": eid,
            })
    return {
        "athletes": athletes,
        "countries": countries,
        "venues": venues,
        "documents": documents,
        "traversals": traversals,
        "won_by_count": len(won_by),
        "held_at_count": len(held_at),
    }


def _evidence_for_doc(doc_id):
    corpus = load_corpus()
    row = corpus.get(doc_id)
    if not row:
        return None
    text = row["text"]
    if len(text) > EVIDENCE_CHARS:
        text = text[:EVIDENCE_CHARS].rsplit(" ", 1)[0]
    return {
        "doc_id": doc_id,
        "title": row["title"],
        "text": text,
    }


def retrieve_graph(question, top_k=DEFAULT_TOP_K):
    entities = match_entities(question)
    selected, seed_paths, event_games, event_sport = select_events(
        entities, top_k=top_k
    )

    matched = {
        "games": [
            {"id": g["id"], "name": g["name"], "year": g["year"], "season": g["season"]}
            for g in entities["games"]
        ],
        "sports": [{"id": s["id"], "name": s["name"]} for s in entities["sports"]],
        "venues": [{"id": v["id"], "name": v["name"]} for v in entities["venues"]],
        "countries": [{"id": c["id"], "name": c["name"]} for c in entities["countries"]],
        "gender": entities["gender"],
        "temporal": entities["temporal"],
        "events": [],
        "athletes": [],
    }

    traversals = list(seed_paths)
    evidence = []
    document_ids = []
    ranked_events = []
    athlete_seen = set()

    for score, overlap, eid, event, gid, sid in selected:
        expansion = expand_event(event, event_games, event_sport)
        traversals.extend(expansion["traversals"])
        matched["events"].append({
            "id": event["id"],
            "name": event["name"],
            "games_id": gid,
            "sport_id": sid,
            "score": round(float(score), 4),
            "token_overlap": overlap,
        })
        ranked_events.append({
            "event_id": event["id"],
            "name": event["name"],
            "score": float(score),
        })
        for athlete in expansion["athletes"]:
            if athlete["id"] not in athlete_seen:
                athlete_seen.add(athlete["id"])
                matched["athletes"].append({
                    "id": athlete["id"],
                    "name": athlete["name"],
                    "event_id": event["id"],
                })
        for doc in expansion["documents"]:
            if doc["id"] not in document_ids:
                document_ids.append(doc["id"])
                item = _evidence_for_doc(doc["id"])
                if item:
                    item["score"] = float(score)
                    evidence.append(item)

    evidence.sort(key=lambda r: (-r.get("score", 0.0), r["doc_id"]))
    unique_paths = []
    seen_paths = set()
    for row in traversals:
        key = (row["path"], row.get("from_id"), row.get("to_id"))
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique_paths.append(row)

    return {
        "matched_entities": matched,
        "graph_paths": unique_paths,
        "document_ids": document_ids,
        "evidence": evidence,
        "scores": [
            {"event_id": row["id"], "name": row["name"], "score": row["score"]}
            for row in matched["events"]
        ],
        "stats": {
            "n_matched_games": len(matched["games"]),
            "n_matched_sports": len(matched["sports"]),
            "n_matched_events": len(matched["events"]),
            "n_matched_athletes": len(matched["athletes"]),
            "n_documents": len(document_ids),
            "n_graph_paths": len(unique_paths),
            "temporal": entities["temporal"],
            "gender": entities["gender"],
        },
    }


if __name__ == "__main__":
    question = input("Enter your question: ")
    result = retrieve_graph(question)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
