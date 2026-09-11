"""Read-only TigerGraph 4.2 REST helpers for GraphRAG."""

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(".")

from config import TIGERGRAPH_GRAPH, TIGERGRAPH_REST_URL

_vertex_cache = {}


def _get_json(url, timeout=60):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_vertices(vtype, limit=20000):
    if vtype in _vertex_cache:
        return _vertex_cache[vtype]
    url = (
        f"{TIGERGRAPH_REST_URL}/graph/{TIGERGRAPH_GRAPH}/vertices/"
        f"{vtype}?limit={limit}"
    )
    raw = _get_json(url)
    if raw.get("error"):
        raise RuntimeError(f"TigerGraph vertex fetch failed for {vtype}: {raw}")
    rows = []
    for item in raw.get("results") or []:
        attrs = item.get("attributes") or {}
        name = attrs.get("name") or attrs.get("title") or item.get("v_id")
        rows.append({
            "id": item.get("v_id"),
            "type": item.get("v_type") or vtype,
            "name": name,
            "attributes": attrs,
        })
    rows.sort(key=lambda r: r["id"])
    _vertex_cache[vtype] = rows
    return rows


def fetch_edges(src_type, src_id, edge_type):
    encoded = urllib.parse.quote(str(src_id), safe="")
    url = (
        f"{TIGERGRAPH_REST_URL}/graph/{TIGERGRAPH_GRAPH}/edges/"
        f"{src_type}/{encoded}/{edge_type}"
    )
    raw = _get_json(url)
    if raw.get("error"):
        raise RuntimeError(
            f"TigerGraph edge fetch failed for {src_type}/{src_id}/{edge_type}: {raw}"
        )
    edges = []
    for item in raw.get("results") or []:
        edges.append({
            "e_type": item.get("e_type") or edge_type,
            "from_id": str(item.get("from_id")),
            "from_type": item.get("from_type") or src_type,
            "to_id": str(item.get("to_id")),
            "to_type": item.get("to_type"),
        })
    edges.sort(key=lambda e: (e["from_id"], e["to_id"]))
    return edges


def fetch_edges_optional(src_type, src_id, edge_type):
    try:
        return fetch_edges(src_type, src_id, edge_type)
    except Exception:
        return []


_held_at_by_venue = None


def events_held_at_venue(venue_id):
    """Return Event IDs with HELD_AT into this venue (read-only).

    Tries the venue endpoint first; if directed REST returns nothing, builds a
    reverse index from Event -(HELD_AT)-> Venue.
    """
    global _held_at_by_venue
    incoming = fetch_edges_optional("Venue", venue_id, "HELD_AT")
    event_ids = []
    for edge in incoming:
        if edge.get("from_type") == "Event":
            event_ids.append(edge["from_id"])
        elif edge.get("to_type") == "Event":
            event_ids.append(edge["to_id"])
        elif edge.get("from_id") and edge.get("from_id") != str(venue_id):
            event_ids.append(edge["from_id"])
    event_ids = sorted(set(event_ids))
    if event_ids:
        return event_ids
    if _held_at_by_venue is None:
        index = {}
        events = fetch_vertices("Event")

        def _held_at(event):
            try:
                edges = fetch_edges("Event", event["id"], "HELD_AT")
            except Exception:
                return event["id"], []
            return event["id"], [edge["to_id"] for edge in edges]

        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(_held_at, event) for event in events]
            for fut in as_completed(futures):
                eid, venue_ids = fut.result()
                for vid in venue_ids:
                    index.setdefault(vid, []).append(eid)
        for key in list(index):
            index[key] = sorted(set(index[key]))
        _held_at_by_venue = index
    return list(_held_at_by_venue.get(venue_id, []))


def fetch_vertex(vtype, vid):
    encoded = urllib.parse.quote(str(vid), safe="")
    url = (
        f"{TIGERGRAPH_REST_URL}/graph/{TIGERGRAPH_GRAPH}/vertices/"
        f"{vtype}/{encoded}"
    )
    raw = _get_json(url)
    if raw.get("error"):
        return None
    rows = raw.get("results") or []
    if not rows:
        return None
    item = rows[0]
    attrs = item.get("attributes") or {}
    return {
        "id": item.get("v_id"),
        "type": item.get("v_type") or vtype,
        "name": attrs.get("name") or attrs.get("title") or item.get("v_id"),
        "attributes": attrs,
    }
