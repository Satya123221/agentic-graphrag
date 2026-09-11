"""Thin application adapter over existing pipeline interfaces.

Does not reimplement RAG, GraphRAG retrieval, planning, or graph access.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

from config import CORPUS_PATH
from agent.orchestrator import investigate
from backend.router import (
    CURRENT_WEB_ANSWER,
    ROUTE_AGENTIC,
    ROUTE_CALCULATION,
    ROUTE_CURRENT,
    ROUTE_GENERAL,
    ROUTE_LABELS,
    ask_general_llm,
    classify,
    evaluate_calculation,
)
from rag.graph_retriever import retrieve_graph
from rag.pipeline import (
    INSUFFICIENT_EVIDENCE,
    _build_context,
    _count_tokens,
    ask_ollama,
    rag_answer,
)

INSUFFICIENT_DISPLAY = "Insufficient evidence"
PIPELINE_ERROR_DISPLAY = "Pipeline error"


@lru_cache(maxsize=1)
def _source_urls():
    urls = {}
    path = Path(CORPUS_PATH)
    if not path.is_file():
        return urls
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id")
            url = row.get("url")
            if doc_id and url:
                urls[doc_id] = url
    return urls


def _flatten_entities(identified):
    rows = []
    for vtype, items in (identified or {}).items():
        for item in items or []:
            rows.append({
                "id": item.get("id"),
                "type": item.get("type") or vtype,
                "name": item.get("name"),
            })
    return rows


def _format_trace(raw_trace):
    formatted = []
    for step in raw_trace or []:
        compact = step.get("tool_result") or {}
        added = int(compact.get("n_evidence") or 0) > 0
        formatted.append({
            "step": step.get("iteration"),
            "action": step.get("chosen_action"),
            "arguments": step.get("tool_arguments") or {},
            "reason": step.get("reason"),
            "planner": step.get("planner"),
            "result_summary": {
                "ok": compact.get("ok"),
                "n_entities": compact.get("n_entities"),
                "n_events": compact.get("n_events"),
                "n_documents": compact.get("n_documents"),
                "n_evidence": compact.get("n_evidence"),
                "error": compact.get("error"),
            },
            "added_evidence": added,
            "new_documents": step.get("newly_discovered_documents") or [],
            "new_entities": step.get("newly_discovered_entities") or [],
            "unresolved": step.get("missing_after") if step.get("missing_after") is not None else (
                step.get("missing_before") or []
            ),
        })
    return formatted


def _format_evidence(items):
    urls = _source_urls()
    evidence = []
    for item in items or []:
        doc_id = item.get("doc_id")
        evidence.append({
            "doc_id": doc_id,
            "title": item.get("title") or "",
            "text": item.get("text") or "",
            "url": urls.get(doc_id),
        })
    return evidence


def format_investigation(result):
    """Map orchestrator output to the application response schema."""
    result = result or {}
    state = result.get("state") or {}
    answer = (result.get("final_answer") or "").strip()
    if not answer:
        answer = INSUFFICIENT_DISPLAY
    trace = result.get("trace") or []
    llm_used = any((step or {}).get("planner") == "ollama" for step in trace)
    return {
        "question": result.get("question") or "",
        "answer": answer,
        "investigation_kind": result.get("investigation_kind"),
        "trace": _format_trace(trace),
        "evidence": _format_evidence(state.get("evidence") or []),
        "entities": _flatten_entities(state.get("identified_entities") or {}),
        "graph_paths": list(state.get("graph_paths") or []),
        "steps": int(state.get("iteration_count") or 0),
        "tools_used": list(result.get("tools_used") or []),
        "stopping_reason": result.get("stop_reason"),
        "stopping_status": result.get("stopping_status"),
        "supporting_evidence": bool(result.get("supporting_evidence")),
        "llm_used": bool(llm_used),
    }


def ask_question(question, investigate_fn=None):
    """Run one Agentic investigation through the existing orchestrator."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question is required")
    runner = investigate_fn or investigate
    result = runner(question)
    return format_investigation(result)


def _routed_payload(question, route, answer, llm_used=False, stopping_reason=None):
    return {
        "question": question,
        "answer": answer,
        "route": route,
        "route_label": ROUTE_LABELS[route],
        "investigation_kind": None,
        "trace": [],
        "evidence": [],
        "entities": [],
        "graph_paths": [],
        "steps": 0,
        "tools_used": [],
        "stopping_reason": stopping_reason,
        "stopping_status": "answered",
        "supporting_evidence": False,
        "llm_used": bool(llm_used),
    }


def route_and_answer(
    question,
    investigate_fn=None,
    general_llm_fn=None,
):
    """Classify the question, then run only the selected pipeline."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question is required")
    decision = classify(question)
    route = decision["route"]
    if route == ROUTE_CALCULATION:
        try:
            answer = evaluate_calculation(decision["expression"])
        except ZeroDivisionError:
            answer = "Division by zero is undefined."
        except (ValueError, SyntaxError, TypeError):
            answer = "I could not evaluate that expression."
        return _routed_payload(
            question,
            route,
            answer,
            stopping_reason=decision["reason"],
        )
    if route == ROUTE_CURRENT:
        return _routed_payload(
            question,
            route,
            CURRENT_WEB_ANSWER,
            stopping_reason=decision["reason"],
        )
    if route == ROUTE_GENERAL:
        llm = general_llm_fn or ask_general_llm
        answer = (llm(question) or "").strip() or "I could not generate an answer."
        return _routed_payload(
            question,
            route,
            answer,
            llm_used=True,
            stopping_reason=decision["reason"],
        )
    payload = dict(ask_question(question, investigate_fn=investigate_fn))
    payload["route"] = ROUTE_AGENTIC
    payload["route_label"] = ROUTE_LABELS[ROUTE_AGENTIC]
    return payload


def _display_answer(answer):
    text = (answer or "").strip()
    if not text or text == INSUFFICIENT_EVIDENCE:
        return INSUFFICIENT_DISPLAY
    return text


def _format_rag_evidence(chunks):
    urls = _source_urls()
    evidence = []
    for chunk in chunks or []:
        doc_id = chunk.get("doc_id")
        evidence.append({
            "doc_id": doc_id,
            "title": chunk.get("title") or "",
            "text": chunk.get("text") or "",
            "score": chunk.get("score"),
            "chunk_id": chunk.get("chunk_id"),
            "url": urls.get(doc_id),
        })
    return evidence


def _flatten_graph_entities(matched):
    rows = []
    for vtype, items in (matched or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            rows.append({
                "id": item.get("id"),
                "type": item.get("type") or vtype,
                "name": item.get("name"),
            })
    return rows


def _failed_pipeline(name, question, latency, error):
    payload = {
        "pipeline": name,
        "ok": False,
        "error": error,
        "question": question,
        "answer": PIPELINE_ERROR_DISPLAY,
        "latency_sec": latency,
        "evidence": [],
        "document_ids": [],
        "llm_used": False,
    }
    if name == "rag":
        payload["similarity_scores"] = []
        payload["context_char_count"] = 0
        payload["context_token_count"] = 0
    elif name == "graphrag":
        payload["entities"] = []
        payload["matched_entities"] = {}
        payload["graph_paths"] = []
        payload["scores"] = []
        payload["stats"] = {}
        payload["context_char_count"] = 0
        payload["context_token_count"] = 0
    elif name == "agentic":
        payload.update({
            "investigation_kind": None,
            "trace": [],
            "entities": [],
            "graph_paths": [],
            "steps": 0,
            "tools_used": [],
            "stopping_reason": None,
            "stopping_status": None,
            "supporting_evidence": False,
        })
    return payload


def run_rag_pipeline(question, rag_fn=None):
    """Invoke existing rag_answer once and normalize the application payload."""
    runner = rag_fn or rag_answer
    t0 = time.perf_counter()
    try:
        raw = runner(question) or {}
    except Exception as exc:
        return _failed_pipeline("rag", question, round(time.perf_counter() - t0, 2), str(exc))
    latency = round(time.perf_counter() - t0, 2)
    chunks = raw.get("retrieved_chunks") or []
    return {
        "pipeline": "rag",
        "ok": True,
        "error": None,
        "question": question,
        "answer": _display_answer(raw.get("answer")),
        "latency_sec": latency,
        "evidence": _format_rag_evidence(chunks),
        "document_ids": list(raw.get("document_ids") or []),
        "similarity_scores": list(raw.get("similarity_scores") or []),
        "context_char_count": int(raw.get("context_char_count") or 0),
        "context_token_count": int(raw.get("context_token_count") or 0),
        "llm_used": bool(chunks),
    }


def run_graphrag_pipeline(question, retrieve_fn=None, generate_fn=None, build_context_fn=None):
    """Invoke existing retrieve_graph plus the existing RAG generation helpers."""
    retrieve = retrieve_fn or retrieve_graph
    generate = generate_fn or ask_ollama
    build_context = build_context_fn or _build_context
    t0 = time.perf_counter()
    try:
        retrieved = retrieve(question) or {}
        evidence_raw = retrieved.get("evidence") or []
        chunks = [
            {
                "doc_id": item.get("doc_id"),
                "title": item.get("title") or "",
                "text": item.get("text") or "",
            }
            for item in evidence_raw
        ]
        context = build_context(chunks) if chunks else ""
        if chunks:
            answer = generate(question, context) or INSUFFICIENT_EVIDENCE
            llm_used = True
        else:
            answer = INSUFFICIENT_EVIDENCE
            llm_used = False
    except Exception as exc:
        return _failed_pipeline(
            "graphrag", question, round(time.perf_counter() - t0, 2), str(exc)
        )
    latency = round(time.perf_counter() - t0, 2)
    matched = retrieved.get("matched_entities") or {}
    return {
        "pipeline": "graphrag",
        "ok": True,
        "error": None,
        "question": question,
        "answer": _display_answer(answer),
        "latency_sec": latency,
        "entities": _flatten_graph_entities(matched),
        "matched_entities": matched,
        "graph_paths": list(retrieved.get("graph_paths") or []),
        "evidence": [
            {**item, "score": raw.get("score")}
            for item, raw in zip(_format_evidence(evidence_raw), evidence_raw)
        ],
        "document_ids": list(retrieved.get("document_ids") or []),
        "scores": list(retrieved.get("scores") or []),
        "stats": dict(retrieved.get("stats") or {}),
        "context_char_count": len(context),
        "context_token_count": _count_tokens(context) if context else 0,
        "llm_used": llm_used,
    }


def run_agentic_pipeline(question, investigate_fn=None):
    """Invoke existing investigate via ask_question and keep the full trace."""
    t0 = time.perf_counter()
    try:
        payload = dict(ask_question(question, investigate_fn=investigate_fn))
    except Exception as exc:
        return _failed_pipeline(
            "agentic", question, round(time.perf_counter() - t0, 2), str(exc)
        )
    payload["pipeline"] = "agentic"
    payload["ok"] = True
    payload["error"] = None
    payload["latency_sec"] = round(time.perf_counter() - t0, 2)
    payload["document_ids"] = [
        item.get("doc_id") for item in (payload.get("evidence") or []) if item.get("doc_id")
    ]
    return payload


def _normalize_agree(text):
    return " ".join((text or "").lower().split())


def _comparison_summary(rag, graphrag, agentic):
    answers = {
        "rag": rag.get("answer") or INSUFFICIENT_DISPLAY,
        "graphrag": graphrag.get("answer") or INSUFFICIENT_DISPLAY,
        "agentic": agentic.get("answer") or INSUFFICIENT_DISPLAY,
    }
    groups = {}
    for name, payload in (("rag", rag), ("graphrag", graphrag), ("agentic", agentic)):
        if not payload.get("ok"):
            continue
        key = _normalize_agree(payload.get("answer"))
        groups.setdefault(key, []).append(name)
    agreeing = []
    for names in groups.values():
        if len(names) >= 2:
            agreeing.extend(names)
    agreeing = list(dict.fromkeys(agreeing))
    latencies = {
        "rag": rag.get("latency_sec"),
        "graphrag": graphrag.get("latency_sec"),
        "agentic": agentic.get("latency_sec"),
    }
    numeric = {
        name: value for name, value in latencies.items() if isinstance(value, (int, float))
    }
    fastest = min(numeric, key=numeric.get) if numeric else None
    tools = agentic.get("tools_used") or []
    return {
        "rag_answer": answers["rag"],
        "graphrag_answer": answers["graphrag"],
        "agentic_answer": answers["agentic"],
        "agreeing_pipelines": agreeing,
        "agentic_used_multiple_tools": len(tools) >= 2,
        "latencies_sec": latencies,
        "fastest": fastest,
    }


def compare_pipelines(
    question,
    rag_fn=None,
    retrieve_fn=None,
    generate_fn=None,
    investigate_fn=None,
):
    """Run RAG, GraphRAG, and Agentic once each and return a side-by-side payload."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question is required")
    rag = run_rag_pipeline(question, rag_fn=rag_fn)
    graphrag = run_graphrag_pipeline(
        question,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
    )
    agentic = run_agentic_pipeline(question, investigate_fn=investigate_fn)
    return {
        "question": question,
        "rag": rag,
        "graphrag": graphrag,
        "agentic": agentic,
        "comparison": _comparison_summary(rag, graphrag, agentic),
    }
