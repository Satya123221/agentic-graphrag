"""State-driven Agentic GraphRAG orchestrator.

Chooses one read-only tool per iteration from the current investigation
gaps. Does not use a fixed search → graph → documents pipeline.
Does not modify TigerGraph, RAG, or GraphRAG.
"""

import json
import re
import sys

sys.path.append(".")

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from agent.state import (
    STATUS_ANSWERED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    InvestigationState,
)
from agent.tools import TOOLS, call_tool
from rag.graph_client import fetch_vertices
from rag.graph_retriever import load_corpus

DEFAULT_MAX_ITERATIONS = 14
AGGREGATION_MAX_ITERATIONS = 48
PLANNER_TIMEOUT_SEC = 120
PRIORITY_UNIQUE_GAP = 12
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
GOLD_RE = re.compile(r"gold:\s*([^\n]+)", re.IGNORECASE)
INFOBOX_RE = re.compile(
    r"\b(nations|competitors|participants):\s*(\d+)", re.IGNORECASE
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
BEFORE_RE = re.compile(
    r"(?:immediately|just)?\s*(?:before|prior to|preceding)\s+(?:the\s+)?(\d{4})",
    re.IGNORECASE,
)
AFTER_RE = re.compile(
    r"(?:immediately|just)?\s*(?:after|following)\s+(?:the\s+)?(\d{4})",
    re.IGNORECASE,
)
PREVIOUS_RE = re.compile(
    r"\b(?:previous|preceding|prior)\b.{0,24}\b(?:olympics|olympic|games)\b",
    re.IGNORECASE,
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
MONTH_NAMES = {n: name for name, n in MONTHS.items()}
MONTH_ALT = (
    r"January|February|March|April|May|June|July|"
    r"August|September|October|November|December"
)
DASH_RE = re.compile(r"[\u2013\u2014\u2212]")
WEIGHT_CLASS_RE = re.compile(r"([+\-])?(\d+)\s*kg\b", re.IGNORECASE)
DATE_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({MONTH_ALT})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
DATE_MDY_RE = re.compile(
    rf"\b({MONTH_ALT})\s+(\d{{1,2}})(?:,)?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
DATE_RANGE_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})\s*(?:-|to)\s*(\d{{1,2}})\s+({MONTH_ALT})(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
DATE_RANGE_MDY_RE = re.compile(
    rf"\b({MONTH_ALT})\s+(\d{{1,2}})\s*(?:-|to)\s*(\d{{1,2}})(?:,)?(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
DATE_RANGE_CROSS_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({MONTH_ALT})\s*(?:-|to)\s*(\d{{1,2}})\s+({MONTH_ALT})(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
DATE_RANGE_CROSS_MDY_RE = re.compile(
    rf"\b({MONTH_ALT})\s+(\d{{1,2}})\s*(?:-|to)\s*({MONTH_ALT})\s+(\d{{1,2}})(?:,)?(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
DATE_DMY_FLEX_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({MONTH_ALT})(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
DATE_MDY_FLEX_RE = re.compile(
    rf"\b({MONTH_ALT})\s+(\d{{1,2}})(?:,)?(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
INFOBOX_DATE_RE = re.compile(r"(?im)^\s*date:\s*(.+)$")
INFOBOX_GAMES_RE = re.compile(
    r"(?im)^\s*games:\s*((?:19|20)\d{2})\s+(Summer|Winter)\b"
)
COMPARE_RE = re.compile(
    r"\b(more than|greater than|over|fewer than|less than|at least)\s+(\d+)\b",
    re.IGNORECASE,
)
SUPERLATIVE_RE = re.compile(
    r"\b(highest|lowest|most|fewest|least)\b",
    re.IGNORECASE,
)
WHICH_EVENT_RE = re.compile(
    r"\bwhich\s+((?:[a-z]+(?:\s+[a-z]+){0,3}))\s+events?\b",
    re.IGNORECASE,
)
HOW_MANY_EVENTS_SPORT_RE = re.compile(
    r"\b(?:how many|number of)\s+(.+?)\s+events?\b",
    re.IGNORECASE,
)
GAMES_NAME_RE = re.compile(r"^(\d{4})\s+(Summer|Winter)\b", re.IGNORECASE)
EVENT_HINT_STOPWORDS = {
    "a", "an", "and", "at", "for", "in", "of", "on", "the", "to", "that",
    "which", "whose", "with", "had", "have", "has",
    "olympic", "olympics", "games", "summer", "winter",
    "according", "provided", "corpus",
    "competitor", "competitors", "nation", "nations",
    "participant", "participants",
    "more", "than", "greater", "over", "fewer", "less", "least",
    "number", "highest", "lowest", "most", "fewest", "count",
}
COUNTRY_CONSTRAINT_STOP = {
    "in", "at", "for", "of", "on", "during", "who", "which", "that",
    "event", "events", "sport", "olympic", "olympics", "games",
    "participated", "participate", "athlete", "athletes",
}

PLANNER_PROMPT = """You are a GraphRAG investigation planner.
Pick exactly ONE next read-only tool from the candidate list.
Honor the investigation kind:
- temporal: resolve previous/next Olympic Games before choosing the final Event
- aggregation: collect evidence for the full candidate Event set, then count
- superlative: collect evidence for the full candidate Event set, then return the Event name with the highest/lowest infobox value
- venue_date: use Venue reverse HELD_AT, then filter Events by date
- lookup: retrieve corpus evidence for the specific Event
Never stop while listed gaps remain. Do not use web search.
Reply with JSON only:
{"decision":"act","tool":"<tool_name>","reason":"<short reason>"}
or
{"decision":"stop","tool":null,"reason":"<short reason>"}
/no_think"""


def _strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text):
    stop = {
        "a", "an", "and", "at", "did", "for", "in", "of", "on", "the", "to",
        "who", "what", "which", "how", "many", "won", "gold", "medal",
        "event", "events", "according", "provided", "corpus", "had",
    }
    return [tok for tok in _norm(text).split() if tok and tok not in stop]


def _entity_rows(state, vtype):
    return list((state.identified_entities or {}).get(vtype) or [])


def _normalize_dashes(text):
    return DASH_RE.sub("-", text or "")


def _spans_overlap(span, occupied):
    start, end = span
    for other_start, other_end in occupied:
        if start < other_end and other_start < end:
            return True
    return False


def _range_record(start_day, end_day, start_month, year, end_month=None, end_year=None):
    if end_month is None:
        end_month = start_month
    if end_year is None:
        end_year = year
    start = {"day": int(start_day), "month": int(start_month), "year": year}
    end = {"day": int(end_day), "month": int(end_month), "year": end_year}
    return {"start": start, "end": end}


def _parse_dates(question, fallback_years=None):
    """Parse exact dates and day-month ranges into range records.

    Exact dates are represented as a range whose start and end are equal.
    Year may be attached to the date or taken from nearby/fallback years.
    Cross-month ranges such as '18 September to 1 October 2000' are one
    continuous range, not two independent point dates.
    """
    text = _normalize_dashes(question or "")
    years = [int(y) for y in YEAR_RE.findall(text)]
    for year in fallback_years or []:
        year = int(year)
        if year not in years:
            years.append(year)

    def fill_year(raw):
        if raw:
            return int(raw)
        if years:
            return int(years[0])
        return None

    ranges = []
    occupied = []

    def add(rng, span):
        if _spans_overlap(span, occupied):
            return
        occupied.append(span)
        ranges.append(rng)

    for match in DATE_RANGE_CROSS_DMY_RE.finditer(text):
        add(
            _range_record(
                match.group(1),
                match.group(3),
                MONTHS[match.group(2).lower()],
                fill_year(match.group(5)),
                end_month=MONTHS[match.group(4).lower()],
            ),
            match.span(),
        )
    for match in DATE_RANGE_CROSS_MDY_RE.finditer(text):
        add(
            _range_record(
                match.group(2),
                match.group(4),
                MONTHS[match.group(1).lower()],
                fill_year(match.group(5)),
                end_month=MONTHS[match.group(3).lower()],
            ),
            match.span(),
        )
    for match in DATE_RANGE_DMY_RE.finditer(text):
        month = MONTHS[match.group(3).lower()]
        add(
            _range_record(match.group(1), match.group(2), month, fill_year(match.group(4))),
            match.span(),
        )
    for match in DATE_RANGE_MDY_RE.finditer(text):
        month = MONTHS[match.group(1).lower()]
        add(
            _range_record(match.group(2), match.group(3), month, fill_year(match.group(4))),
            match.span(),
        )
    for match in DATE_DMY_FLEX_RE.finditer(text):
        month = MONTHS[match.group(2).lower()]
        day = int(match.group(1))
        add(_range_record(day, day, month, fill_year(match.group(3))), match.span())
    for match in DATE_MDY_FLEX_RE.finditer(text):
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        add(_range_record(day, day, month, fill_year(match.group(3))), match.span())
    return ranges


def _parse_games_name(name):
    match = GAMES_NAME_RE.match((name or "").strip())
    if not match:
        return None
    return {"year": int(match.group(1)), "season": match.group(2).lower()}


def _clean_event_hint(phrase):
    """Keep the sport/domain noun phrase; drop years, seasons, and filler."""
    kept = []
    for tok in _norm(phrase).split():
        if not tok or tok in EVENT_HINT_STOPWORDS:
            continue
        if YEAR_RE.fullmatch(tok) or tok.isdigit():
            continue
        kept.append(tok)
    if not kept:
        return None
    return " ".join(kept)


def _event_hint_from_question(q, which_event=None):
    """Prefer the noun phrase attached to 'which/how many ... events'."""
    captured = (which_event.group(1) if which_event else "") or ""
    hint = _clean_event_hint(captured)
    if hint:
        return hint
    how_many_sport = HOW_MANY_EVENTS_SPORT_RE.search(q or "")
    if how_many_sport:
        hint = _clean_event_hint(how_many_sport.group(1))
        if hint:
            return hint
    lexicon = re.search(
        r"\b(biathlon|shooting|judo|sailing|athletics|weightlifting|"
        r"taekwondo|cycling|rowing|fencing|swimming|boxing|wrestling|"
        r"gymnastics|speed skating|figure skating|alpine skiing)\b",
        q or "",
    )
    return lexicon.group(1).lower() if lexicon else None


def _country_hint_from_question(q):
    """Extract a nationality constraint such as 'from India' or 'represented India'."""
    match = re.search(
        r"\b(?:from|represented|representing)\s+(?:the\s+)?(.+)$",
        q or "",
    )
    if not match:
        return None
    kept = []
    for tok in match.group(1).split():
        if tok in COUNTRY_CONSTRAINT_STOP or tok in EVENT_HINT_STOPWORDS:
            break
        if tok:
            kept.append(tok)
    if not kept:
        return None
    return " ".join(kept)


def question_intent(question):
    raw = question or ""
    q = _norm(raw)
    years = [int(y) for y in YEAR_RE.findall(raw)]
    before = BEFORE_RE.search(raw)
    after = AFTER_RE.search(raw)
    previous = bool(PREVIOUS_RE.search(raw))
    dates = _parse_dates(raw)
    venue_centric = bool(re.search(r"\bheld at\b", q))
    how_many = bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q))
    how_many_events = bool(re.search(r"\bhow many\b.{0,80}\bevents?\b", q))
    compare = COMPARE_RE.search(raw)
    asks_who = bool(re.search(r"\bwho\b|\bwhich athletes?\b", q))
    which_event = WHICH_EVENT_RE.search(q)
    superlative_word = SUPERLATIVE_RE.search(q)
    is_superlative = bool(
        which_event and superlative_word
        or re.search(r"\b(highest|lowest|most|fewest|least)\s+number of\b", q)
    )
    season = None
    if "winter" in q.split():
        season = "winter"
    elif "summer" in q.split():
        season = "summer"
    gender = None
    if re.search(r"\bwomen s\b|\bwomen\b|\bfemale\b", q):
        gender = "women"
    elif re.search(r"\bmen s\b|\bmen\b|\bmale\b", q):
        gender = "men"
    temporal = None
    anchor_year = None
    if before:
        temporal = "previous"
        anchor_year = int(before.group(1))
    elif after:
        temporal = "next"
        anchor_year = int(after.group(1))
    elif previous and years:
        temporal = "previous"
        anchor_year = years[0]
    field = "nations" if re.search(r"\bnations?\b", q) and not re.search(
        r"\bcompetitors?\b", q
    ) else "competitors"
    agg_spec = None
    if how_many_events and compare:
        phrase = compare.group(1).lower()
        value = int(compare.group(2))
        if "fewer" in phrase or "less" in phrase:
            op = "<"
        elif "at least" in phrase:
            op = ">="
        else:
            op = ">"
        agg_spec = {"field": field, "op": op, "value": value}
    superlative_spec = None
    if is_superlative:
        word = (superlative_word.group(1).lower() if superlative_word else "highest")
        direction = "lowest" if word in {"lowest", "fewest", "least"} else "highest"
        superlative_spec = {"field": field, "direction": direction}
    if is_superlative:
        kind = "superlative"
    elif how_many_events or (how_many and compare and re.search(r"\bevents?\b", q)):
        kind = "aggregation"
    elif venue_centric and (dates or years):
        kind = "venue_date"
    elif temporal:
        kind = "temporal"
    elif how_many:
        kind = "lookup"
    elif asks_who:
        kind = "who_won"
    else:
        kind = "multi_hop"
    has_specific_event = bool(re.search(
        r"\b(\d+\s*kg|metres|meters|walk|judo|sailing|biathlon|"
        r"shooting|weightlifting|rs:?x|sprint|pursuit)\b",
        q,
    ))
    event_hint = _event_hint_from_question(q, which_event)
    country_hint = _country_hint_from_question(q)
    return {
        "kind": kind,
        "asks_who": asks_who,
        "asks_how_many": how_many and kind != "superlative",
        "asks_where": bool(re.search(r"\bwhere\b", q)),
        "asks_country": bool(re.search(r"\b(country|nation|noc)\b", q)) and not how_many,
        "country_hint": country_hint,
        "venue_centric": venue_centric,
        "has_year": bool(years),
        "has_date": bool(dates),
        "dates": dates,
        "years": years,
        "season": season,
        "gender": gender,
        "temporal": temporal,
        "anchor_year": anchor_year,
        "aggregation": agg_spec,
        "superlative": superlative_spec,
        "has_specific_event": has_specific_event,
        "event_hint": event_hint,
    }


def parse_weight_classes(text):
    """Signed Olympic weight classes, e.g. ('', 80), ('+', 80), ('-', 60)."""
    blob = DASH_RE.sub("-", text or "")
    found = []
    seen = set()
    for match in WEIGHT_CLASS_RE.finditer(blob):
        sign = match.group(1) or ""
        kg = int(match.group(2))
        item = (sign, kg)
        if item in seen:
            continue
        seen.add(item)
        found.append(item)
    return found


def weight_class_exact_match(query, candidate):
    """True only when a question class matches a candidate class including sign."""
    query_classes = parse_weight_classes(query)
    candidate_classes = parse_weight_classes(candidate)
    if not query_classes or not candidate_classes:
        return False
    return any(item in candidate_classes for item in query_classes)


def _weight_class_conflict(event_name, question):
    """Same numeric class with a different sign is not an exact match."""
    query_classes = parse_weight_classes(question)
    event_classes = parse_weight_classes(event_name)
    if not query_classes or not event_classes:
        return 0
    if any(item in event_classes for item in query_classes):
        return 0
    query_kg = {kg for _, kg in query_classes}
    event_kg = {kg for _, kg in event_classes}
    if query_kg & event_kg:
        return 1
    return 0


def _event_overlap(event_name, question):
    name_toks = set(_tokens(event_name))
    q_toks = _tokens(question)
    if not q_toks:
        return 0
    score = sum(1 for tok in q_toks if tok in name_toks)
    if weight_class_exact_match(question, event_name):
        score += 10
    return score


def ranked_events(state, question, limit=5, pool=None):
    events = pool if pool is not None else _entity_rows(state, "Event")
    scored = []
    for event in events:
        name = event.get("name") or ""
        scored.append((
            _event_overlap(name, question),
            _weight_class_conflict(name, question),
            event.get("id") or "",
            event,
        ))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [row[3] for row in scored[:limit] if row[3].get("id")]


def parsed_games(state):
    rows = []
    for vertex in _entity_rows(state, "OlympicGames"):
        parsed = _parse_games_name(vertex.get("name") or "")
        if not parsed:
            continue
        rows.append({
            "id": vertex.get("id"),
            "name": vertex.get("name"),
            "year": parsed["year"],
            "season": parsed["season"],
        })
    return rows


def resolve_target_games(state, intent=None):
    intent = intent or question_intent(state.original_question)
    games = parsed_games(state)
    if not games:
        return []
    season = intent.get("season")
    if intent.get("temporal") == "previous" and intent.get("anchor_year"):
        earlier = [
            g for g in games
            if g["year"] < intent["anchor_year"]
            and (season is None or g["season"] == season)
        ]
        earlier.sort(key=lambda g: (-g["year"], g["id"] or ""))
        if earlier:
            latest = earlier[0]["year"]
            return [g for g in earlier if g["year"] == latest]
    if intent.get("temporal") == "next" and intent.get("anchor_year"):
        later = [
            g for g in games
            if g["year"] > intent["anchor_year"]
            and (season is None or g["season"] == season)
        ]
        later.sort(key=lambda g: (g["year"], g["id"] or ""))
        if later:
            soonest = later[0]["year"]
            return [g for g in later if g["year"] == soonest]
    years = intent.get("years") or []
    if years:
        matched = [g for g in games if g["year"] in years]
        if season:
            seasonal = [g for g in matched if g["season"] == season]
            if seasonal:
                return seasonal
        if matched:
            return matched
    return games


def aggregation_required_tokens(intent, state=None):
    """Whole-word tokens that define the aggregation Event universe."""
    tokens = []
    if intent.get("event_hint"):
        tokens.append(str(intent["event_hint"]).lower())
    if state is not None:
        for row in _entity_rows(state, "Sport"):
            name = _norm(row.get("name") or "")
            if name and name not in tokens:
                tokens.append(name)
                break
    if intent.get("temporal"):
        target = resolve_target_games(state, intent) if state is not None else []
        if target:
            tokens.append(str(target[0]["year"]))
            tokens.append(target[0]["season"])
    else:
        years = intent.get("years") or []
        if years:
            tokens.append(str(years[0]))
        if intent.get("season"):
            tokens.append(intent["season"])
    seen = []
    for tok in tokens:
        if tok and tok not in seen:
            seen.append(tok)
    return seen


def aggregation_search_query(intent, state=None):
    tokens = aggregation_required_tokens(intent, state)
    if tokens:
        return " ".join(tokens)
    return event_search_query(state, intent) if state is not None else ""


def _needs_complete_event_set(intent):
    return intent.get("kind") in {"aggregation", "superlative"}


def _aggregation_discovery_complete(state, intent=None):
    """True after a non-truncated sport/year/season Event search."""
    intent = intent or question_intent(state.original_question)
    if not _needs_complete_event_set(intent):
        return True
    required = aggregation_required_tokens(intent, state)
    for row in state.actions:
        if row.get("tool") != "search_events":
            continue
        args = row.get("args") or {}
        got = [
            str(tok).lower()
            for tok in (args.get("required_tokens") or [])
            if str(tok).strip()
        ]
        if required:
            if not got or any(tok not in got for tok in required):
                continue
        elif not got:
            continue
        try:
            top_k = int(args.get("top_k"))
        except (TypeError, ValueError):
            top_k = 8
        if top_k <= 0:
            return True
    return False


def event_search_query(state, intent=None):
    intent = intent or question_intent(state.original_question)
    question = state.original_question
    target = resolve_target_games(state, intent)
    if target and intent.get("temporal"):
        name = target[0]["name"]
        rewritten = BEFORE_RE.sub(name, question)
        rewritten = AFTER_RE.sub(name, rewritten)
        rewritten = re.sub(
            r"\b(?:previous|preceding)\s+(?:olympics|olympic games|games)\b",
            name,
            rewritten,
            flags=re.IGNORECASE,
        )
        if name.lower() not in rewritten.lower():
            rewritten = f"{rewritten} {name}"
        return rewritten
    if _needs_complete_event_set(intent):
        scoped = aggregation_search_query(intent, state)
        if scoped:
            return scoped
    sports = [row.get("name") for row in _entity_rows(state, "Sport") if row.get("name")]
    if _needs_complete_event_set(intent) and (sports or target):
        parts = [sports[0] if sports else ""]
        if target:
            parts.append(target[0]["name"])
        extra = " ".join(p for p in parts if p)
        if extra:
            return f"{extra} {question}"
    return question


def _infobox_value(text, field):
    for match in INFOBOX_RE.finditer(text or ""):
        if match.group(1).lower().startswith(field[:6]):
            return int(match.group(2))
    return None


def _is_point_range(rng):
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    return (
        start.get("day") == end.get("day")
        and start.get("month") == end.get("month")
        and start.get("year") == end.get("year")
    )


def _filled_ymd(part, years):
    if not part:
        return None
    year = part.get("year")
    if year is None and years:
        year = years[0]
    if part.get("month") is None or part.get("day") is None:
        return None
    return (year, int(part["month"]), int(part["day"]))


def _parse_corpus_date_ranges(text, intent=None):
    intent = intent or {}
    fallback = list(intent.get("years") or [])
    games = INFOBOX_GAMES_RE.search(text or "")
    if games:
        fallback = [int(games.group(1))] + [y for y in fallback if int(y) != int(games.group(1))]
    date_line = INFOBOX_DATE_RE.search(text or "")
    blob = date_line.group(1) if date_line else (text or "")
    ranges = _parse_dates(blob, fallback_years=fallback)
    if not ranges and date_line:
        extra = " ".join(str(year) for year in fallback)
        ranges = _parse_dates(f"{date_line.group(1)} {extra}", fallback_years=fallback)
    return ranges


def _question_date_constraint(intent):
    return bool(intent.get("dates"))


def _ranges_compatible(question_rng, corpus_rng, years):
    q_start = _filled_ymd(question_rng.get("start"), years)
    q_end = _filled_ymd(question_rng.get("end"), years)
    c_start = _filled_ymd(corpus_rng.get("start"), years)
    c_end = _filled_ymd(corpus_rng.get("end"), years)
    if not q_start or not q_end or not c_start or not c_end:
        return False
    if _is_point_range(question_rng):
        return c_start <= q_start <= c_end
    return q_start == c_start and q_end == c_end


def _evidence_matches_date(text, intent):
    intent = intent or {}
    q_ranges = intent.get("dates") or []
    years = list(intent.get("years") or [])
    if not q_ranges:
        if intent.get("kind") == "venue_date" and years:
            blob = _norm(text)
            return any(re.search(rf"\b{year}\b", blob) for year in years)
        if intent.get("kind") == "venue_date":
            return False
        return True
    corpus_ranges = _parse_corpus_date_ranges(text, intent)
    if not corpus_ranges:
        return False
    for question_rng in q_ranges:
        for corpus_rng in corpus_ranges:
            if _ranges_compatible(question_rng, corpus_rng, years):
                return True
    return False


def _event_corpus_text(event):
    corpus = load_corpus()
    row = corpus.get(event.get("id") or "") or {}
    return row.get("text") or ""


def _expand_related_venues(venues):
    """Include more-specific venues whose names contain a matched venue name."""
    if not venues:
        return list(venues or [])
    known = []
    seen = set()
    for venue in venues:
        vid = venue.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        known.append(venue)
    names = [_norm(v.get("name") or "") for v in known if _norm(v.get("name") or "")]
    extras = []
    for vertex in fetch_vertices("Venue"):
        vid = vertex.get("id")
        if not vid or vid in seen:
            continue
        name = _norm(vertex.get("name") or "")
        if not name:
            continue
        if any(
            base and len(base) >= 6 and name != base and name.startswith(base + " ")
            for base in names
        ):
            seen.add(vid)
            extras.append({
                "id": vid,
                "type": "Venue",
                "name": vertex.get("name"),
            })
    extras.sort(key=lambda row: (-len(row.get("name") or ""), row.get("id") or ""))
    return known + extras


def _venue_date_status(event, intent):
    """Classify a venue Event against the question date constraint."""
    text = _event_corpus_text(event)
    if not _question_date_constraint(intent):
        if intent.get("kind") == "venue_date" and intent.get("years"):
            return "match" if _evidence_matches_date(text, intent) else "no_match"
        return "unknown"
    corpus_ranges = _parse_corpus_date_ranges(text, intent)
    if not corpus_ranges:
        return "unknown"
    if _evidence_matches_date(text, intent):
        return "match"
    return "no_match"


def _evidenced_ids(state):
    ids = set(state.document_ids or [])
    for item in state.evidence or []:
        if item.get("doc_id"):
            ids.add(item["doc_id"])
    return ids


def _retrieve_attempted_ids(state):
    """Event/document ids already passed to retrieve_documents."""
    ids = set()
    for row in state.actions or []:
        if row.get("tool") != "retrieve_documents":
            continue
        args = row.get("args") or {}
        event_id = args.get("event_id")
        if event_id:
            ids.add(event_id)
        for doc_id in args.get("doc_ids") or []:
            if doc_id:
                ids.add(doc_id)
    return ids


def scope_events(state, intent=None):
    intent = intent or question_intent(state.original_question)
    events = _entity_rows(state, "Event")
    if not events:
        return []
    sports = [
        _norm(row.get("name") or "")
        for row in _entity_rows(state, "Sport")
        if row.get("name")
    ]
    if intent.get("event_hint"):
        sports.append(_norm(intent["event_hint"]))
    target = resolve_target_games(state, intent)
    years = [g["year"] for g in target] if target else list(intent.get("years") or [])
    seasons = [g["season"] for g in target] if target else (
        [intent["season"]] if intent.get("season") else []
    )
    kept = []
    for event in events:
        name = _norm(event.get("name") or "")
        if sports and not any(sport in name for sport in sports if len(sport) >= 4):
            continue
        if years and not any(str(year) in name for year in years):
            continue
        if seasons and not any(season in name for season in seasons):
            continue
        if intent.get("gender") and intent["gender"] not in name and (
            intent.get("kind") in {"lookup", "who_won", "temporal"}
        ):
            continue
        kept.append(event)
    if not kept:
        if intent.get("kind") == "venue_date" and (intent.get("event_hint") or sports):
            return []
        year_only = []
        for event in events:
            name = _norm(event.get("name") or "")
            if years and not any(str(year) in name for year in years):
                continue
            year_only.append(event)
        kept = year_only or events
    return kept


def unevidenced_events(state, intent=None):
    intent = intent or question_intent(state.original_question)
    scoped = scope_events(state, intent)
    have = _evidenced_ids(state)
    leftover = [event for event in scoped if event.get("id") not in have]
    if intent.get("kind") == "superlative":
        attempted = _retrieve_attempted_ids(state)
        leftover = [
            event for event in leftover if event.get("id") not in attempted
        ]
    if intent.get("kind") == "venue_date":
        ranked = []
        for event in leftover:
            status = _venue_date_status(event, intent)
            if status == "no_match":
                continue
            score = 2 if status == "match" else 0
            if any(str(year) in (event.get("name") or "") for year in (intent.get("years") or [])):
                score += 1
            ranked.append((score, event.get("id") or "", event))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        leftover = [row[2] for row in ranked]
    return leftover


def date_matched_doc_ids(state, intent=None):
    intent = intent or question_intent(state.original_question)
    return [
        item.get("doc_id")
        for item in (state.evidence or [])
        if item.get("doc_id") and _evidence_matches_date(item.get("text") or "", intent)
    ]


def count_qualifying_events(state, intent=None):
    intent = intent or question_intent(state.original_question)
    spec = intent.get("aggregation")
    if not spec:
        return None
    scoped_ids = {event.get("id") for event in scope_events(state, intent) if event.get("id")}
    count = 0
    for item in state.evidence or []:
        doc_id = item.get("doc_id")
        if scoped_ids and doc_id not in scoped_ids:
            continue
        value = _infobox_value(item.get("text") or "", spec["field"])
        if value is None:
            continue
        if spec["op"] == ">" and value > spec["value"]:
            count += 1
        elif spec["op"] == ">=" and value >= spec["value"]:
            count += 1
        elif spec["op"] == "<" and value < spec["value"]:
            count += 1
    return count


def pick_superlative_event(state, intent=None):
    """Return the Event name with the highest or lowest infobox value."""
    intent = intent or question_intent(state.original_question)
    spec = intent.get("superlative")
    if not spec:
        return None
    scoped = {
        event.get("id"): event
        for event in scope_events(state, intent)
        if event.get("id")
    }
    best_by_event = {}
    for item in state.evidence or []:
        doc_id = item.get("doc_id")
        if not doc_id or (scoped and doc_id not in scoped):
            continue
        value = _infobox_value(item.get("text") or "", spec["field"])
        if value is None:
            continue
        name = item.get("title") or (scoped.get(doc_id) or {}).get("name")
        prev = best_by_event.get(doc_id)
        if prev is None or value > prev[0]:
            best_by_event[doc_id] = (value, name)
    if not best_by_event:
        return None
    rows = list(best_by_event.values())
    if spec.get("direction") == "lowest":
        winner = min(rows, key=lambda row: (row[0], row[1] or ""))
    else:
        winner = max(rows, key=lambda row: (row[0], row[1] or ""))
    return winner[1]


def _needs_country_filter(intent):
    return bool((intent or {}).get("country_hint"))


def _country_hint_norm(intent):
    return _norm((intent or {}).get("country_hint") or "")


def _country_event_pool(state, intent):
    events = _entity_rows(state, "Event")
    hint = _norm((intent or {}).get("event_hint") or "")
    if not hint:
        return events
    matched = [
        event for event in events
        if hint in _norm(event.get("name") or "")
    ]
    return matched or events


def _country_matched_athletes(state, intent):
    """Athletes linked by Athlete-(REPRESENTS)->Country to the question country."""
    hint = _country_hint_norm(intent)
    if not hint:
        return []
    country_ids = set()
    for row in _entity_rows(state, "Country"):
        name = _norm(row.get("name") or "")
        cid = _norm(row.get("id") or "")
        if hint == name or hint == cid or hint in name.split():
            country_ids.add(row["id"])
    athlete_ids = []
    seen = set()
    for path in state.graph_paths or []:
        if (path.get("path") or "") != "Athlete-(REPRESENTS)->Country":
            continue
        to_id = path.get("to_id")
        from_id = path.get("from_id")
        if not from_id or from_id in seen:
            continue
        if to_id in country_ids or _norm(to_id) == hint:
            seen.add(from_id)
            athlete_ids.append(from_id)
    by_id = {row["id"]: row for row in _entity_rows(state, "Athlete") if row.get("id")}
    return [by_id[aid] for aid in athlete_ids if aid in by_id]


def inspect_gaps(state, intent=None):
    intent = intent or question_intent(state.original_question)
    events = _entity_rows(state, "Event")
    venues = _entity_rows(state, "Venue")
    sports = _entity_rows(state, "Sport")
    games = _entity_rows(state, "OlympicGames")
    athletes = _entity_rows(state, "Athlete")
    countries = _entity_rows(state, "Country")
    tools_used = [row.get("tool") for row in state.actions]
    kind = intent["kind"]
    target = resolve_target_games(state, intent)
    gaps = []
    if kind == "temporal" and not target:
        gaps.append("target_games")
    if kind == "venue_date" and not venues:
        gaps.append("venues")
    if not (venues or sports or games or events) and "target_games" not in gaps:
        gaps.append("named_entities")
    if kind == "venue_date" and venues and not events:
        gaps.append("venue_events")
    elif not events:
        gaps.append("events")
    if _needs_complete_event_set(intent):
        leftover = unevidenced_events(state, intent)
        if not _aggregation_discovery_complete(state, intent):
            gaps.append("incomplete_candidate_set")
        if events and leftover:
            gaps.append("candidate_event_evidence")
        elif (
            events
            and not leftover
            and _aggregation_discovery_complete(state, intent)
        ):
            if kind == "aggregation" and intent.get("aggregation"):
                gaps.append("aggregation_count")
            elif kind == "superlative" and intent.get("superlative"):
                gaps.append("superlative_select")
    if kind == "venue_date" and events and (
        _question_date_constraint(intent) or intent.get("years")
    ):
        leftover = unevidenced_events(state, intent)
        if leftover:
            gaps.append("date_evidence")
        elif not date_matched_doc_ids(state, intent) and state.evidence:
            gaps.append("date_filter")
    if intent["asks_who"] and not athletes:
        gold_rows = state.evidence
        if kind == "venue_date" and _question_date_constraint(intent):
            matched = set(date_matched_doc_ids(state, intent))
            gold_rows = [
                item for item in gold_rows if item.get("doc_id") in matched
            ]
        gold_hit = any(GOLD_RE.search(item.get("text") or "") for item in gold_rows)
        if not gold_hit:
            gaps.append("athletes")
    if _needs_country_filter(intent) and not _country_matched_athletes(state, intent):
        gaps.append("country_filter")
    if intent["asks_country"] and not countries:
        gaps.append("countries")
    if intent["asks_where"] and not venues:
        gaps.append("venues")
    if not state.evidence:
        gaps.append("corpus_evidence")
    return {
        "gaps": gaps,
        "intent": intent,
        "kind": kind,
        "counts": {
            "events": len(events),
            "venues": len(venues),
            "sports": len(sports),
            "games": len(games),
            "athletes": len(athletes),
            "countries": len(countries),
            "documents": len(state.document_ids),
            "evidence": len(state.evidence),
            "graph_paths": len(state.graph_paths),
            "unevidenced_events": len(unevidenced_events(state, intent)),
            "target_games": len(target),
        },
        "tools_used": tools_used,
    }


def evidence_is_sufficient(state, intent=None):
    intent = intent or question_intent(state.original_question)
    blob = "\n".join((item.get("text") or "")[:1500] for item in state.evidence)
    kind = intent["kind"]
    if _needs_complete_event_set(intent):
        if not _aggregation_discovery_complete(state, intent):
            return False
        leftover = unevidenced_events(state, intent)
        if leftover:
            return False
        if not state.evidence:
            return False
        if kind == "aggregation":
            return count_qualifying_events(state, intent) is not None
        return pick_superlative_event(state, intent) is not None
    if kind == "lookup":
        return bool(INFOBOX_RE.search(blob))
    if kind == "venue_date":
        leftover = unevidenced_events(state, intent)
        if leftover:
            return False
        matched = date_matched_doc_ids(state, intent)
        if not matched:
            return False
        if len(set(matched)) > 1 and _question_date_constraint(intent):
            return False
        if intent["asks_who"]:
            matched_text = "\n".join(
                item.get("text") or ""
                for item in state.evidence
                if item.get("doc_id") in matched
            )
            return bool(GOLD_RE.search(matched_text) or _entity_rows(state, "Athlete"))
        return True
    if _needs_country_filter(intent):
        return bool(_country_matched_athletes(state, intent))
    if kind in {"who_won", "temporal", "multi_hop"} and intent["asks_who"]:
        if not state.evidence:
            return False
        return bool(GOLD_RE.search(blob) or _entity_rows(state, "Athlete"))
    if intent["asks_where"]:
        return bool(_entity_rows(state, "Venue") and state.evidence)
    return bool(state.evidence and state.document_ids)


def extract_answer(state, intent=None):
    intent = intent or question_intent(state.original_question)
    kind = intent["kind"]
    if kind == "aggregation":
        if not _aggregation_discovery_complete(state, intent):
            return None
        if unevidenced_events(state, intent):
            return None
        count = count_qualifying_events(state, intent)
        if count is not None and state.evidence:
            return str(count)
        return None
    if kind == "superlative":
        if not _aggregation_discovery_complete(state, intent):
            return None
        if unevidenced_events(state, intent):
            return None
        name = pick_superlative_event(state, intent)
        if name:
            return name
        return None
    evidence_rows = list(state.evidence or [])
    if kind == "venue_date" and (
        _question_date_constraint(intent) or intent.get("years")
    ):
        matched = set(date_matched_doc_ids(state, intent))
        filtered = [
            item for item in evidence_rows if item.get("doc_id") in matched
        ]
        if filtered:
            evidence_rows = filtered
    if kind == "lookup" or (
        intent["asks_how_many"] and kind not in {"aggregation", "superlative"}
    ):
        field = "nations" if "nation" in _norm(state.original_question) else "competitors"
        ranked = sorted(
            evidence_rows,
            key=lambda item: -_event_overlap(item.get("title") or "", state.original_question),
        )
        for item in ranked:
            value = _infobox_value(item.get("text") or "", field)
            if value is not None:
                return str(value)
    if intent["asks_who"]:
        if _needs_country_filter(intent):
            names = []
            seen = set()
            for row in _country_matched_athletes(state, intent):
                name = (row.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
            if names:
                return ", ".join(names)
            return None
        question = state.original_question
        ranked = sorted(
            evidence_rows,
            key=lambda item: (
                -_event_overlap(item.get("title") or "", question),
                _weight_class_conflict(item.get("title") or "", question),
            ),
        )
        for item in ranked:
            match = GOLD_RE.search(item.get("text") or "")
            if match:
                return match.group(1).strip()
        athletes = [
            row.get("name") for row in _entity_rows(state, "Athlete") if row.get("name")
        ]
        if athletes:
            return athletes[0]
    if intent["asks_where"]:
        venues = [row.get("name") for row in _entity_rows(state, "Venue") if row.get("name")]
        if venues:
            return venues[0]
    return None


def _fingerprint(tool, args):
    return (tool, json.dumps(args or {}, sort_keys=True, default=str))


def _used_fingerprints(state):
    used = set()
    for row in state.actions:
        used.add(_fingerprint(row.get("tool"), row.get("args")))
    return used


def _action(tool, args, reason, priority):
    return {
        "tool": tool,
        "args": dict(args or {}),
        "reason": reason,
        "priority": priority,
    }


def _traversed(state):
    return {
        (
            row.get("args", {}).get("src_type"),
            row.get("args", {}).get("src_id"),
            row.get("args", {}).get("edge_type"),
            row.get("args", {}).get("direction", "out"),
        )
        for row in state.actions
        if row.get("tool") == "traverse_graph"
    }


def propose_actions(state, max_candidates=6):
    """Build next-action candidates from current gaps only."""
    intent = question_intent(state.original_question)
    question = state.original_question
    used = _used_fingerprints(state)
    events = _entity_rows(state, "Event")
    venues = _entity_rows(state, "Venue")
    athletes = _entity_rows(state, "Athlete")
    countries = _entity_rows(state, "Country")
    tools_used = {row.get("tool") for row in state.actions}
    traversed = _traversed(state)
    retrieved_events = {
        row.get("args", {}).get("event_id")
        for row in state.actions
        if row.get("tool") == "retrieve_documents"
    }
    looked_up = {
        row.get("args", {}).get("doc_id") or row.get("args", {}).get("event_id")
        for row in state.actions
        if row.get("tool") == "lookup_evidence"
    }
    kind = intent["kind"]
    target = resolve_target_games(state, intent)
    candidates = []

    if _needs_complete_event_set(intent) and not _aggregation_discovery_complete(state, intent):
        tokens = aggregation_required_tokens(intent, state)
        if tokens:
            reason = (
                "Superlative needs the complete sport/year Event set, not a truncated ranking."
                if kind == "superlative"
                else "Aggregation needs the complete sport/year Event set, not a truncated ranking."
            )
            candidates.append(_action(
                "search_events",
                {
                    "query": aggregation_search_query(intent, state),
                    "top_k": 0,
                    "required_tokens": tokens,
                },
                reason,
                112,
            ))
            return _finalize_candidates(candidates, used)

    if kind == "temporal" and not target and "search_entities" not in tools_used:
        candidates.append(_action(
            "search_entities",
            {"query": question, "limit": 20},
            "Temporal question; resolve the relevant Olympic Games first.",
            110,
        ))
        return _finalize_candidates(candidates, used)

    if kind == "venue_date" and not venues and "search_entities" not in tools_used:
        candidates.append(_action(
            "search_entities",
            {"query": question, "limit": 20},
            "Venue+date question; identify the venue before walking HELD_AT.",
            110,
        ))
        return _finalize_candidates(candidates, used)

    if kind == "venue_date" and venues:
        leftover_here = unevidenced_events(state, intent)
        have_date_match = bool(date_matched_doc_ids(state, intent))
        walk_pool = _expand_related_venues(venues)
        if not leftover_here and (not events or not have_date_match):
            ranked_venues = ranked_events(
                state,
                question,
                limit=len(walk_pool),
                pool=walk_pool,
            ) or walk_pool
            for venue in ranked_venues:
                key = ("Venue", venue["id"], "HELD_AT", "in")
                if key in traversed:
                    continue
                candidates.append(_action(
                    "traverse_graph",
                    {
                        "src_type": "Venue",
                        "src_id": venue["id"],
                        "edge_type": "HELD_AT",
                        "direction": "in",
                        "limit": 40,
                    },
                    "Venue is known; reverse-walk HELD_AT to candidate Events.",
                    108,
                ))
                break
            if candidates:
                return _finalize_candidates(candidates, used)

    if not (events or venues or _entity_rows(state, "Sport") or _entity_rows(state, "OlympicGames")):
        if intent["has_specific_event"] and "search_events" not in tools_used:
            if _needs_complete_event_set(intent):
                top_k = 24
            elif _needs_country_filter(intent):
                top_k = 12
            else:
                top_k = 5
            search_args = {
                "query": event_search_query(state, intent),
                "top_k": top_k,
            }
            if _needs_country_filter(intent) and intent.get("event_hint"):
                search_args["required_tokens"] = [intent["event_hint"]]
            candidates.append(_action(
                "search_events",
                search_args,
                "Question already names an event; search Event vertices.",
                95,
            ))
        if "search_entities" not in tools_used:
            candidates.append(_action(
                "search_entities",
                {"query": question, "limit": 20},
                "No graph entities yet; identify games/sport/venue mentions.",
                80,
            ))
        if "search_events" not in tools_used and not intent["has_specific_event"]:
            top_k = 24 if _needs_complete_event_set(intent) else 5
            candidates.append(_action(
                "search_events",
                {"query": event_search_query(state, intent), "top_k": top_k},
                "No events yet; search Event vertices from the question tokens.",
                70,
            ))

    if venues and not events and kind != "venue_date":
        for venue in venues[:2]:
            key = ("Venue", venue["id"], "HELD_AT", "in")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Venue",
                    "src_id": venue["id"],
                    "edge_type": "HELD_AT",
                    "direction": "in",
                    "limit": 40,
                },
                "Venue is known but events are not; reverse-walk HELD_AT.",
                98,
            ))
            break

    if not events and "search_events" not in tools_used and (
        _entity_rows(state, "Sport") or target or _entity_rows(state, "OlympicGames")
    ) and not (kind == "venue_date" and venues):
        top_k = 24 if _needs_complete_event_set(intent) else 8
        candidates.append(_action(
            "search_events",
            {"query": event_search_query(state, intent), "top_k": top_k},
            "Named/target entities exist; search Event vertices next.",
            90,
        ))

    leftover = unevidenced_events(state, intent)
    if leftover and kind == "superlative" and not _needs_country_filter(intent):
        pending_ids = [
            event["id"] for event in leftover
            if event.get("id") and event["id"] not in retrieved_events
        ]
        if pending_ids:
            candidates.append(_action(
                "retrieve_documents",
                {"doc_ids": pending_ids, "limit": len(pending_ids)},
                "Superlative needs competitor counts for the complete candidate Event set.",
                96,
            ))
            return _finalize_candidates(candidates, used)

    if leftover and kind != "superlative" and not _needs_country_filter(intent) and (
        kind in {"aggregation", "lookup", "venue_date"}
        or not state.evidence
        or intent["asks_who"]
        or intent["asks_how_many"]
    ):
        pool = leftover
        if kind == "venue_date":
            year_hits = leftover
            if intent.get("years"):
                year_hits = [
                    event for event in leftover
                    if any(str(year) in (event.get("name") or "") for year in intent["years"])
                ] or leftover
            matches = [
                event for event in year_hits
                if _venue_date_status(event, intent) == "match"
            ]
            unknown = [
                event for event in year_hits
                if _venue_date_status(event, intent) == "unknown"
            ]
            pool = matches or unknown or year_hits
        for event in ranked_events(state, question, limit=3, pool=pool):
            if event["id"] in retrieved_events:
                continue
            reason = "Corpus evidence is still missing for a known Event."
            if kind == "aggregation":
                reason = "Aggregation needs evidence for the remaining candidate Events."
            elif kind == "venue_date":
                reason = "Retrieve Event documents so date/year can filter venue candidates."
            candidates.append(_action(
                "retrieve_documents",
                {"event_id": event["id"], "limit": 2},
                reason,
                96 if kind in {"aggregation", "superlative", "venue_date", "lookup"} else 93,
            ))
            break

    if athletes and _needs_country_filter(intent) and not _country_matched_athletes(state, intent):
        for athlete in athletes:
            key = ("Athlete", athlete["id"], "REPRESENTS", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Athlete",
                    "src_id": athlete["id"],
                    "edge_type": "REPRESENTS",
                    "direction": "out",
                    "limit": 4,
                },
                "Apply the country constraint via Athlete REPRESENTS.",
                97,
            ))
            return _finalize_candidates(candidates, used)

    if (
        events
        and _needs_country_filter(intent)
        and not _country_matched_athletes(state, intent)
    ):
        pool = _country_event_pool(state, intent)
        for event in ranked_events(state, question, limit=8, pool=pool):
            key = ("Event", event["id"], "WON_BY", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Event",
                    "src_id": event["id"],
                    "edge_type": "WON_BY",
                    "direction": "out",
                    "limit": 8,
                },
                "Country constraint; follow Event WON_BY then Athlete REPRESENTS.",
                94,
            ))
            return _finalize_candidates(candidates, used)

    if (
        kind == "venue_date"
        and not unevidenced_events(state, intent)
        and date_matched_doc_ids(state, intent)
        and intent["asks_who"]
    ):
        matched_ids = set(date_matched_doc_ids(state, intent))
        for event_id in matched_ids:
            key = ("Event", event_id, "WON_BY", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Event",
                    "src_id": event_id,
                    "edge_type": "WON_BY",
                    "direction": "out",
                    "limit": 8,
                },
                "Date-filtered Event is known; follow WON_BY after collecting evidence.",
                88,
            ))
            break

    if (
        events
        and intent["asks_who"]
        and kind not in {"venue_date"}
        and state.evidence
        and not athletes
        and not any(GOLD_RE.search(item.get("text") or "") for item in state.evidence)
    ):
        pool = scope_events(state, intent) or events
        for event in ranked_events(state, question, limit=3, pool=pool):
            key = ("Event", event["id"], "WON_BY", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Event",
                    "src_id": event["id"],
                    "edge_type": "WON_BY",
                    "direction": "out",
                    "limit": 8,
                },
                "Who-question still lacks a gold name in evidence; follow WON_BY.",
                86,
            ))
            break

    if events and intent["asks_where"] and not venues:
        for event in ranked_events(state, question, limit=2):
            key = ("Event", event["id"], "HELD_AT", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Event",
                    "src_id": event["id"],
                    "edge_type": "HELD_AT",
                    "direction": "out",
                    "limit": 8,
                },
                "Where-question; follow Event HELD_AT to the venue.",
                84,
            ))
            break

    if athletes and intent["asks_country"] and not countries:
        for athlete in athletes[:3]:
            key = ("Athlete", athlete["id"], "REPRESENTS", "out")
            if key in traversed:
                continue
            candidates.append(_action(
                "traverse_graph",
                {
                    "src_type": "Athlete",
                    "src_id": athlete["id"],
                    "edge_type": "REPRESENTS",
                    "direction": "out",
                    "limit": 4,
                },
                "Country is needed; follow Athlete REPRESENTS.",
                83,
            ))
            break

    if state.document_ids and not state.evidence:
        for doc_id in state.document_ids:
            if doc_id in looked_up:
                continue
            candidates.append(_action(
                "lookup_evidence",
                {"doc_id": doc_id},
                "Document IDs exist without corpus text; look up evidence.",
                72,
            ))
            break

    return _finalize_candidates(candidates, used, max_candidates)


def _finalize_candidates(candidates, used, max_candidates=6):
    unique = []
    seen = set()
    for action in candidates:
        fp = _fingerprint(action["tool"], action["args"])
        if fp in used or fp in seen:
            continue
        if action["tool"] not in TOOLS:
            continue
        seen.add(fp)
        unique.append(action)
    unique.sort(key=lambda row: (-row["priority"], row["tool"]))
    return unique[:max_candidates]


def _compact_result(result):
    evidence = []
    for item in (result or {}).get("evidence") or []:
        text = item.get("text") or ""
        evidence.append({
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "snippet": text[:240],
        })
    return {
        "ok": (result or {}).get("ok"),
        "tool": (result or {}).get("tool"),
        "error": (result or {}).get("error"),
        "n_entities": len((result or {}).get("entities") or []),
        "n_events": len((result or {}).get("events") or []),
        "n_documents": len((result or {}).get("document_ids") or []),
        "n_evidence": len((result or {}).get("evidence") or []),
        "n_graph_paths": len((result or {}).get("graph_paths") or []),
        "document_ids": list((result or {}).get("document_ids") or []),
        "entities": [
            {"id": row.get("id"), "type": row.get("type"), "name": row.get("name")}
            for row in ((result or {}).get("entities") or [])[:12]
        ],
        "events": [
            {"id": row.get("id"), "name": row.get("name")}
            for row in ((result or {}).get("events") or [])[:8]
        ],
        "graph_paths": list((result or {}).get("graph_paths") or [])[:12],
        "evidence": evidence[:4],
    }


def _plan_with_ollama(question, inspection, candidates):
    payload_actions = [
        {"tool": row["tool"], "args": row["args"], "reason": row["reason"]}
        for row in candidates
    ]
    user = (
        f"Question:\n{question}\n\n"
        f"Investigation kind:\n{inspection.get('kind')}\n"
        f"State counts:\n{json.dumps(inspection['counts'], indent=2)}\n"
        f"Missing:\n{json.dumps(inspection['gaps'])}\n"
        f"Tools already used:\n{json.dumps(inspection['tools_used'])}\n\n"
        f"Candidate actions:\n{json.dumps(payload_actions, indent=2)}\n"
    )
    body = {
        "model": OLLAMA_MODEL,
        "prompt": f"{PLANNER_PROMPT}\n\n{user}\n\nJSON:",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": 160,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=body,
        timeout=PLANNER_TIMEOUT_SEC,
    )
    if response.status_code == 400:
        body.pop("think", None)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=body,
            timeout=PLANNER_TIMEOUT_SEC,
        )
    response.raise_for_status()
    text = _strip_think((response.json() or {}).get("response") or "")
    match = JSON_RE.search(text)
    if not match:
        raise ValueError("planner returned no JSON")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("planner JSON is not an object")
    return parsed


def choose_action(state, candidates, inspection):
    if not candidates:
        return {
            "decision": "stop",
            "tool": None,
            "args": {},
            "reason": "No useful next action remains.",
            "planner": "deterministic",
        }
    chosen = candidates[0]
    unique_top = (
        len(candidates) == 1
        or chosen["priority"] >= candidates[1]["priority"] + PRIORITY_UNIQUE_GAP
    )
    if unique_top:
        return {
            "decision": "act",
            "tool": chosen["tool"],
            "args": chosen["args"],
            "reason": chosen["reason"],
            "planner": "deterministic",
        }
    try:
        parsed = _plan_with_ollama(
            state.original_question, inspection, candidates
        )
        decision = (parsed.get("decision") or "act").lower()
        if decision == "stop":
            if inspection.get("gaps"):
                return {
                    "decision": "act",
                    "tool": chosen["tool"],
                    "args": chosen["args"],
                    "reason": chosen["reason"],
                    "planner": "deterministic",
                }
            return {
                "decision": "stop",
                "tool": None,
                "args": {},
                "reason": parsed.get("reason") or "Planner stopped.",
                "planner": "ollama",
            }
        tool = parsed.get("tool")
        match = next((row for row in candidates if row["tool"] == tool), None)
        if match:
            return {
                "decision": "act",
                "tool": match["tool"],
                "args": match["args"],
                "reason": parsed.get("reason") or match["reason"],
                "planner": "ollama",
            }
    except Exception:
        pass
    return {
        "decision": "act",
        "tool": chosen["tool"],
        "args": chosen["args"],
        "reason": chosen["reason"],
        "planner": "fallback",
    }


def _new_entities(before, after):
    before_keys = {
        (row.get("type"), row.get("id"))
        for rows in (before or {}).values()
        for row in rows
    }
    discovered = []
    for vtype, rows in (after or {}).items():
        for row in rows:
            key = (row.get("type") or vtype, row.get("id"))
            if key in before_keys or not row.get("id"):
                continue
            discovered.append({
                "id": row.get("id"),
                "type": row.get("type") or vtype,
                "name": row.get("name"),
            })
    return discovered


def _new_documents(before_ids, after_ids):
    seen = set(before_ids or [])
    return [doc_id for doc_id in (after_ids or []) if doc_id not in seen]


def _new_paths(before_paths, after_paths):
    seen = {
        (row.get("path"), row.get("from_id"), row.get("to_id"))
        for row in (before_paths or [])
    }
    fresh = []
    for row in after_paths or []:
        key = (row.get("path"), row.get("from_id"), row.get("to_id"))
        if key in seen:
            continue
        fresh.append(dict(row))
    return fresh


def investigate(question, max_iterations=DEFAULT_MAX_ITERATIONS):
    """Run a state-driven investigation. Returns state plus a per-step trace."""
    state = InvestigationState.create(question)
    intent = question_intent(question)
    trace = []
    stop_reason = None

    def iteration_cap():
        cap = max_iterations
        if _needs_complete_event_set(intent):
            n_scoped = len(scope_events(state, intent))
            cap = min(AGGREGATION_MAX_ITERATIONS, max(cap, n_scoped + 2))
        if intent.get("kind") == "venue_date":
            n_events = len(_entity_rows(state, "Event"))
            n_venues = len(_entity_rows(state, "Venue"))
            cap = min(
                AGGREGATION_MAX_ITERATIONS,
                max(cap, n_events + n_venues + 3, 24),
            )
        return cap

    while True:
        if state.stopping_status != STATUS_RUNNING:
            break
        if state.iteration_count >= iteration_cap():
            stop_reason = "maximum iteration limit"
            state.stop(STATUS_STOPPED, answer=extract_answer(state, intent))
            break
        if evidence_is_sufficient(state, intent):
            stop_reason = "sufficient evidence for an answer"
            state.stop(STATUS_ANSWERED, answer=extract_answer(state, intent))
            break

        inspection = inspect_gaps(state, intent)
        candidates = propose_actions(state)
        decision = choose_action(state, candidates, inspection)
        if decision["decision"] != "act" or not decision.get("tool"):
            stop_reason = decision.get("reason") or "no useful next action"
            status = (
                STATUS_ANSWERED
                if evidence_is_sufficient(state, intent)
                else STATUS_STOPPED
            )
            if not state.document_ids and not _entity_rows(state, "Event"):
                status = STATUS_FAILED if state.iteration_count == 0 else STATUS_STOPPED
            state.stop(status, answer=extract_answer(state, intent))
            trace.append({
                "iteration": state.iteration_count + 1,
                "chosen_action": "stop",
                "tool_arguments": {},
                "tool_result": None,
                "reason": stop_reason,
                "planner": decision.get("planner"),
                "newly_discovered_entities": [],
                "newly_discovered_documents": [],
                "graph_paths": [],
                "missing_before": inspection["gaps"],
                "state_counts": inspection["counts"],
            })
            break

        before_entities = {
            vtype: [dict(row) for row in rows]
            for vtype, rows in state.identified_entities.items()
        }
        before_docs = list(state.document_ids)
        before_paths = [dict(row) for row in state.graph_paths]
        result = call_tool(decision["tool"], **decision["args"])
        state.ingest_tool_result(result)
        inspection_after = inspect_gaps(state, intent)
        step = {
            "iteration": state.iteration_count,
            "chosen_action": decision["tool"],
            "tool_arguments": dict(decision["args"]),
            "tool_result": _compact_result(result),
            "reason": decision["reason"],
            "planner": decision.get("planner"),
            "newly_discovered_entities": _new_entities(
                before_entities, state.identified_entities
            ),
            "newly_discovered_documents": _new_documents(
                before_docs, state.document_ids
            ),
            "graph_paths": _new_paths(before_paths, state.graph_paths),
            "missing_before": inspection["gaps"],
            "missing_after": inspection_after["gaps"],
            "state_counts": inspection_after["counts"],
        }
        trace.append(step)

        if evidence_is_sufficient(state, intent):
            stop_reason = "sufficient evidence for an answer"
            state.stop(STATUS_ANSWERED, answer=extract_answer(state, intent))
            break
        if not result.get("ok") and not propose_actions(state):
            stop_reason = "no useful next action"
            state.stop(STATUS_STOPPED, answer=extract_answer(state, intent))
            break

    if state.stopping_status == STATUS_RUNNING:
        stop_reason = stop_reason or "no useful next action"
        state.stop(STATUS_STOPPED, answer=extract_answer(state, intent))

    return {
        "question": question,
        "investigation_kind": intent["kind"],
        "initial_empty": {
            "iteration_count": 0,
            "document_ids": [],
            "evidence": [],
            "graph_paths": [],
            "actions": [],
            "identified_entities": {
                vtype: []
                for vtype in state.identified_entities
            },
        },
        "state": state.to_dict(),
        "trace": trace,
        "stopping_status": state.stopping_status,
        "stop_reason": stop_reason,
        "final_answer": state.final_answer,
        "sufficient_evidence": evidence_is_sufficient(state, intent),
        "supporting_evidence": bool(state.evidence),
        "tools_used": [step["chosen_action"] for step in trace if step["chosen_action"] != "stop"],
    }


if __name__ == "__main__":
    question = input("Enter your question: ")
    result = investigate(question)
    print(json.dumps({
        "investigation_kind": result["investigation_kind"],
        "stopping_status": result["stopping_status"],
        "stop_reason": result["stop_reason"],
        "final_answer": result["final_answer"],
        "tools_used": result["tools_used"],
        "supporting_evidence": result["supporting_evidence"],
        "trace": result["trace"],
        "sufficient_evidence": result["sufficient_evidence"],
    }, ensure_ascii=False, indent=2)[:8000])
