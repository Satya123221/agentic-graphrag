"""Read-only investigation tools over TigerGraph and the local corpus.

Does not modify the graph, the RAG baseline, or GraphRAG ranking.
Does not call web search.
"""

import re
import sys

sys.path.append(".")

from rag.graph_client import (
    events_held_at_venue,
    fetch_edges,
    fetch_edges_optional,
    fetch_vertex,
    fetch_vertices,
)
from rag.graph_retriever import EVIDENCE_CHARS, load_corpus, match_entities

TOOL_RESULT_FIELDS = (
    "ok",
    "tool",
    "args",
    "data",
    "entities",
    "events",
    "document_ids",
    "evidence",
    "graph_paths",
    "error",
)

OUTGOING_EDGES = {
    ("Document", "DESCRIBES"): "Event",
    ("Event", "PART_OF"): "OlympicGames",
    ("Event", "WON_BY"): "Athlete",
    ("Event", "IN_SPORT"): "Sport",
    ("Event", "HELD_AT"): "Venue",
    ("Athlete", "REPRESENTS"): "Country",
}

REVERSE_EDGES = {
    ("Venue", "HELD_AT"): "Event",
}


def tool_result(
    tool,
    args,
    data=None,
    entities=None,
    events=None,
    document_ids=None,
    evidence=None,
    graph_paths=None,
    error=None,
):
    return {
        "ok": error is None,
        "tool": tool,
        "args": dict(args or {}),
        "data": data if data is not None else {},
        "entities": list(entities or []),
        "events": list(events or []),
        "document_ids": list(document_ids or []),
        "evidence": list(evidence or []),
        "graph_paths": list(graph_paths or []),
        "error": error,
    }


def _entity(vertex, vtype=None):
    if not vertex:
        return None
    return {
        "id": vertex.get("id"),
        "type": vtype or vertex.get("type"),
        "name": vertex.get("name"),
    }


def _path(label, from_id, to_id):
    return {"path": label, "from_id": from_id, "to_id": to_id}


def _evidence_item(doc_id):
    corpus = load_corpus()
    row = corpus.get(doc_id)
    if not row:
        return None
    text = row.get("text") or ""
    if len(text) > EVIDENCE_CHARS:
        text = text[:EVIDENCE_CHARS].rsplit(" ", 1)[0]
    return {
        "doc_id": doc_id,
        "title": row.get("title") or "",
        "text": text,
    }


def _tokens(text):
    return [tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if tok]


VENUE_ALIAS_STOP = {
    "olympic", "olympics", "summer", "winter", "national", "international",
    "stadium", "centre", "center", "arena", "park", "field", "complex",
    "pavilion", "hall", "court", "ground", "the", "and",
}


def _match_venues_by_alias(query):
    """Match Venue vertices from a distinctive leading name token such as ExCeL."""
    q_tokens = set(_tokens(query))
    aliases = {
        tok for tok in q_tokens
        if len(tok) >= 5 and tok not in VENUE_ALIAS_STOP
    }
    if not aliases:
        return []
    hits = []
    seen = set()
    for vertex in fetch_vertices("Venue"):
        vid = vertex.get("id")
        if not vid or vid in seen:
            continue
        name_tokens = _tokens(vertex.get("name") or "")
        if name_tokens and name_tokens[0] in aliases:
            seen.add(vid)
            hits.append(vertex)
    hits.sort(key=lambda row: (-len(row.get("name") or ""), row.get("id") or ""))
    return hits


def search_entities(query, limit=20):
    """Find named graph entities mentioned in a question or phrase."""
    args = {"query": query, "limit": limit}
    if not query or not str(query).strip():
        return tool_result("search_entities", args, error="query is required")
    matched = match_entities(query)
    entities = []
    for vertex in matched.get("games") or []:
        entities.append(_entity(vertex, "OlympicGames"))
    for vtype, key in (
        ("Sport", "sports"),
        ("Venue", "venues"),
        ("Country", "countries"),
    ):
        for vertex in matched.get(key) or []:
            entities.append(_entity(vertex, vtype))
    seen_venues = {
        row.get("id") for row in entities if row and row.get("type") == "Venue"
    }
    for vertex in _match_venues_by_alias(query):
        if vertex.get("id") in seen_venues:
            continue
        row = _entity(vertex, "Venue")
        if row:
            entities.append(row)
            seen_venues.add(row["id"])
    entities = [row for row in entities if row]
    return tool_result(
        "search_entities",
        args,
        data={
            "gender": matched.get("gender"),
            "season": matched.get("season"),
            "years": list(matched.get("years") or []),
            "dates": list(matched.get("dates") or []),
            "temporal": matched.get("temporal"),
            "question_tokens": list(matched.get("question_tokens") or []),
        },
        entities=entities[:limit],
    )


def search_events(query, top_k=8, required_tokens=None):
    """Find Event vertices whose names overlap the query tokens.

    If required_tokens is set, keep only events whose names contain every
    token as a whole word. top_k <= 0 returns the full matching set.
    """
    req = [str(tok).lower() for tok in (required_tokens or []) if str(tok).strip()]
    args = {"query": query, "top_k": top_k, "required_tokens": req}
    if not query or not str(query).strip():
        return tool_result("search_events", args, error="query is required")
    tokens = [tok for tok in _tokens(query) if tok not in {"the", "at", "of", "in"}]
    if not tokens and not req:
        return tool_result("search_events", args, error="query has no searchable tokens")
    ranked = []
    for event in fetch_vertices("Event"):
        name = (event.get("name") or "").lower()
        if req:
            if not all(re.search(r"\b" + re.escape(tok) + r"\b", name) for tok in req):
                continue
        overlap = sum(1 for tok in tokens if tok in name) if tokens else len(req)
        if not req and overlap < 1:
            continue
        ranked.append((overlap, event["id"], event))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    try:
        cap = int(top_k)
    except (TypeError, ValueError):
        cap = 8
    if cap <= 0:
        picked = ranked
    else:
        picked = ranked[: max(cap, 1)]
    events = []
    entities = []
    for overlap, eid, event in picked:
        row = {
            "id": eid,
            "type": "Event",
            "name": event.get("name"),
            "score": overlap,
        }
        events.append(row)
        entities.append(_entity(event, "Event"))
    return tool_result(
        "search_events",
        args,
        data={
            "n_candidates": len(ranked),
            "n_returned": len(picked),
            "complete": len(picked) == len(ranked),
        },
        entities=entities,
        events=events,
    )


def traverse_graph(src_type, src_id, edge_type, direction="out", limit=50):
    """Read-only one-hop traversal along a schema edge."""
    args = {
        "src_type": src_type,
        "src_id": src_id,
        "edge_type": edge_type,
        "direction": direction,
        "limit": limit,
    }
    if not src_type or not src_id or not edge_type:
        return tool_result(
            "traverse_graph", args, error="src_type, src_id, and edge_type are required"
        )
    direction = (direction or "out").lower()
    key = (src_type, edge_type)
    entities = []
    events = []
    document_ids = []
    graph_paths = []
    neighbors = []

    if direction == "in":
        expected = REVERSE_EDGES.get(key)
        if expected is None:
            return tool_result(
                "traverse_graph",
                args,
                error=f"reverse traversal not supported for {src_type}-{edge_type}",
            )
        if key == ("Venue", "HELD_AT"):
            event_ids = events_held_at_venue(src_id)[: max(int(limit), 1)]
            label = "Venue<-(HELD_AT)-Event"
            for eid in event_ids:
                vertex = fetch_vertex("Event", eid)
                graph_paths.append(_path(label, src_id, eid))
                if vertex:
                    entities.append(_entity(vertex, "Event"))
                    events.append({
                        "id": vertex["id"],
                        "type": "Event",
                        "name": vertex.get("name"),
                    })
                    neighbors.append(_entity(vertex, "Event"))
        else:
            return tool_result(
                "traverse_graph",
                args,
                error=f"reverse traversal not supported for {src_type}-{edge_type}",
            )
    else:
        expected = OUTGOING_EDGES.get(key)
        if expected is None:
            return tool_result(
                "traverse_graph",
                args,
                error=f"outgoing traversal not supported for {src_type}-{edge_type}",
            )
        edges = fetch_edges_optional(src_type, src_id, edge_type)[: max(int(limit), 1)]
        label = f"{src_type}-({edge_type})->{expected}"
        for edge in edges:
            to_id = edge.get("to_id")
            to_type = edge.get("to_type") or expected
            graph_paths.append(_path(label, src_id, to_id))
            vertex = fetch_vertex(to_type, to_id)
            if not vertex:
                neighbors.append({"id": to_id, "type": to_type, "name": None})
                continue
            row = _entity(vertex, to_type)
            neighbors.append(row)
            entities.append(row)
            if to_type == "Event":
                events.append({
                    "id": vertex["id"],
                    "type": "Event",
                    "name": vertex.get("name"),
                })
            if to_type == "Document":
                document_ids.append(vertex["id"])

    return tool_result(
        "traverse_graph",
        args,
        data={"n_neighbors": len(neighbors), "neighbors": neighbors},
        entities=entities,
        events=events,
        document_ids=document_ids,
        graph_paths=graph_paths,
    )


def retrieve_documents(doc_ids=None, event_id=None, limit=8):
    """Fetch corpus documents by id or via Document-(DESCRIBES)->Event."""
    args = {"doc_ids": list(doc_ids or []), "event_id": event_id, "limit": limit}
    ids = []
    graph_paths = []
    entities = []
    for doc_id in doc_ids or []:
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
    if event_id:
        describes = fetch_edges_optional("Document", event_id, "DESCRIBES")
        for edge in describes:
            if edge.get("to_id") == event_id:
                graph_paths.append(
                    _path("Document-(DESCRIBES)->Event", edge.get("from_id"), event_id)
                )
                if edge.get("from_id") and edge["from_id"] not in ids:
                    ids.append(edge["from_id"])
        if event_id not in ids:
            doc = fetch_vertex("Document", event_id)
            if doc:
                ids.append(event_id)
                graph_paths.append(
                    _path("Document-(DESCRIBES)->Event", event_id, event_id)
                )
    ids = ids[: max(int(limit), 1)]
    evidence = []
    document_ids = []
    for doc_id in ids:
        item = _evidence_item(doc_id)
        if not item:
            continue
        document_ids.append(doc_id)
        evidence.append(item)
        vertex = fetch_vertex("Document", doc_id)
        if vertex:
            entities.append(_entity(vertex, "Document"))
        else:
            entities.append({
                "id": doc_id,
                "type": "Document",
                "name": item.get("title"),
            })
    if not document_ids:
        return tool_result(
            "retrieve_documents",
            args,
            graph_paths=graph_paths,
            error="no documents found",
        )
    return tool_result(
        "retrieve_documents",
        args,
        data={"n_documents": len(document_ids)},
        entities=entities,
        document_ids=document_ids,
        evidence=evidence,
        graph_paths=graph_paths,
    )


def lookup_evidence(doc_id=None, event_id=None):
    """Look up corpus evidence text for a document or its event page."""
    args = {"doc_id": doc_id, "event_id": event_id}
    target = doc_id or event_id
    if not target:
        return tool_result(
            "lookup_evidence", args, error="doc_id or event_id is required"
        )
    item = _evidence_item(target)
    if not item:
        return tool_result(
            "lookup_evidence", args, error=f"no corpus evidence for {target}"
        )
    return tool_result(
        "lookup_evidence",
        args,
        data={"title": item["title"]},
        entities=[{"id": target, "type": "Document", "name": item["title"]}],
        document_ids=[target],
        evidence=[item],
    )


TOOLS = {
    "search_entities": search_entities,
    "search_events": search_events,
    "traverse_graph": traverse_graph,
    "retrieve_documents": retrieve_documents,
    "lookup_evidence": lookup_evidence,
}


def call_tool(name, **kwargs):
    fn = TOOLS.get(name)
    if fn is None:
        return tool_result(name, kwargs, error=f"unknown tool: {name}")
    return fn(**kwargs)
