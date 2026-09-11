"""Official 100-question public benchmark (final).

Writes only to eval/results/final_public_benchmark/.
Does not read or overwrite the partial 86-record complete-run files.
Does not modify RAG, GraphRAG, TigerGraph, Ollama, or the agent.
"""

import csv
import json
import statistics
import sys
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(".")

import requests

from config import (
    CHUNKS_PATH,
    CORPUS_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PUBLIC_EVAL_PATH,
    TFIDF_INDEX_PATH,
    TIGERGRAPH_GRAPH,
    TIGERGRAPH_REST_URL,
)
from agent.orchestrator import investigate
from rag.graph_retriever import retrieve_graph
from rag.pipeline import INSUFFICIENT_EVIDENCE, _build_context, _count_tokens, ask_ollama, rag_answer
from eval.benchmark_public import (
    PIPELINES,
    _safe_print,
    gold_doc_hit,
    load_eval,
    match_answer,
)
from eval.run_public_complete import classify_agentic_failure, load_all_questions

ALL_TYPES = ("lookup", "aggregation", "temporal", "multi_hop", "superlative")
OUT_DIR = Path("eval") / "results" / "final_public_benchmark"
RESULTS_PATH = OUT_DIR / "benchmark_public_final.json"
SUMMARY_PATH = OUT_DIR / "benchmark_public_summary.json"
REPORT_PATH = OUT_DIR / "benchmark_public_report.md"
FAILURES_JSON = OUT_DIR / "benchmark_public_failures.json"
FAILURES_CSV = OUT_DIR / "benchmark_public_failures.csv"
JSONL_PATH = OUT_DIR / "benchmark_public_final.jsonl"
LOG_PATH = OUT_DIR / "benchmark_public_final.log"
OLD_PARTIAL = Path("eval") / "results" / "benchmark_public_complete.json"

OLLAMA_LAST = {}
_ORIG_POST = requests.post


def _capturing_post(*args, **kwargs):
    response = _ORIG_POST(*args, **kwargs)
    url = str(args[0] if args else kwargs.get("url") or "")
    if "/api/generate" in url:
        try:
            body = response.json()
        except Exception:
            body = {}
        OLLAMA_LAST.clear()
        OLLAMA_LAST.update({
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "prompt_eval_duration": body.get("prompt_eval_duration"),
            "eval_duration": body.get("eval_duration"),
            "total_duration": body.get("total_duration"),
        })
    return response


requests.post = _capturing_post


def _ollama_stats():
    prompt = OLLAMA_LAST.get("prompt_eval_count")
    output = OLLAMA_LAST.get("eval_count")
    total = None
    if prompt is not None or output is not None:
        total = int(prompt or 0) + int(output or 0)
    return {
        "llm_prompt_eval_count": prompt,
        "llm_eval_count": output,
        "llm_total_tokens": total,
        "llm_prompt_eval_duration_ns": OLLAMA_LAST.get("prompt_eval_duration"),
        "llm_eval_duration_ns": OLLAMA_LAST.get("eval_duration"),
        "llm_total_duration_ns": OLLAMA_LAST.get("total_duration"),
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    return float(statistics.median(values))


def _retrieval_recall(gold_ids, retrieved_ids):
    gold = set(gold_ids or [])
    if not gold:
        return 0.0
    return len(gold & set(retrieved_ids or [])) / len(gold)


def stats(items):
    n = len(items)
    empty = {
        "n": 0,
        "overall_accuracy": 0.0,
        "exact_accuracy": 0.0,
        "retrieval_success": 0.0,
        "retrieval_completeness": 0.0,
        "avg_latency_sec": 0.0,
        "median_latency_sec": 0.0,
        "avg_context_tokens": 0.0,
        "avg_llm_prompt_tokens": 0.0,
        "avg_llm_output_tokens": 0.0,
        "avg_llm_total_tokens": 0.0,
        "token_efficiency_per_1k_tokens": 0.0,
        "avg_agent_steps": 0.0,
        "avg_tool_calls": 0.0,
        "n_failures": 0,
        "n_timeouts": 0,
        "n_errors": 0,
    }
    if n == 0:
        return empty
    acc = sum(1 for r in items if r.get("normalized_match")) / n
    exact = sum(1 for r in items if r.get("exact_match")) / n
    retr = sum(1 for r in items if r.get("gold_doc_retrieved")) / n
    completeness = _mean([
        _retrieval_recall(r.get("gold_doc_ids"), r.get("retrieved_document_ids"))
        for r in items
    ])
    latencies = [float(r.get("latency_sec") or 0.0) for r in items]
    tokens = [float(r.get("context_token_count") or 0.0) for r in items]
    llm_prompt = [float(r["llm_prompt_eval_count"]) for r in items if r.get("llm_prompt_eval_count") is not None]
    llm_out = [float(r["llm_eval_count"]) for r in items if r.get("llm_eval_count") is not None]
    llm_tot = [float(r["llm_total_tokens"]) for r in items if r.get("llm_total_tokens") is not None]
    avg_tokens = _mean(tokens)
    efficiency = (acc / avg_tokens * 1000.0) if avg_tokens else 0.0
    return {
        "n": n,
        "overall_accuracy": round(acc, 4),
        "exact_accuracy": round(exact, 4),
        "retrieval_success": round(retr, 4),
        "retrieval_completeness": round(completeness, 4),
        "avg_latency_sec": round(_mean(latencies), 2),
        "median_latency_sec": round(_median(latencies), 2),
        "avg_context_tokens": round(avg_tokens, 1),
        "avg_llm_prompt_tokens": round(_mean(llm_prompt), 1) if llm_prompt else 0.0,
        "avg_llm_output_tokens": round(_mean(llm_out), 1) if llm_out else 0.0,
        "avg_llm_total_tokens": round(_mean(llm_tot), 1) if llm_tot else 0.0,
        "token_efficiency_per_1k_tokens": round(efficiency, 4),
        "avg_agent_steps": round(_mean([r.get("agent_iterations") or 0 for r in items]), 2),
        "avg_tool_calls": round(_mean([len(r.get("agent_tool_calls") or []) for r in items]), 2),
        "n_failures": sum(1 for r in items if not r.get("normalized_match")),
        "n_timeouts": sum(
            1 for r in items
            if "timeout" in (r.get("error") or "").lower()
            or "timed out" in (r.get("error") or "").lower()
        ),
        "n_errors": sum(1 for r in items if r.get("error")),
    }


def summarize(rows):
    by_pipeline = {name: [] for name in PIPELINES}
    by_qtype = {name: defaultdict(list) for name in PIPELINES}
    for row in rows:
        by_pipeline[row["pipeline"]].append(row)
        by_qtype[row["pipeline"]][row["qtype"]].append(row)
    overall = {name: stats(items) for name, items in by_pipeline.items()}
    qtype_stats = {}
    qtype_counts = Counter(row["qtype"] for row in rows if row.get("pipeline") == "rag")
    for name in PIPELINES:
        qtype_stats[name] = {
            qtype: stats(by_qtype[name][qtype]) for qtype in ALL_TYPES
        }
    agentic_rows = by_pipeline["agentic"]
    categories = Counter(classify_agentic_failure(row) for row in agentic_rows)
    failures = {}
    for name in PIPELINES:
        failures[name] = [
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "predicted_answer": row.get("predicted_answer"),
                "gold_answer": row.get("gold_answer"),
                "gold_doc_retrieved": row.get("gold_doc_retrieved"),
                "retrieval_completeness": row.get("retrieval_completeness"),
                "error": row.get("error"),
                "latency_sec": row.get("latency_sec"),
            }
            for row in by_pipeline[name]
            if not row.get("normalized_match")
        ]
    return {
        "overall": overall,
        "qtype_counts": dict(qtype_counts),
        "by_qtype": qtype_stats,
        "agentic_failure_categories": dict(categories),
        "failures": failures,
        "n_records": len(rows),
    }


def run_rag(question):
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    raw = rag_answer(question)
    latency = time.perf_counter() - t0
    chunks = raw.get("retrieved_chunks") or []
    evidence = [
        {
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
            "text": (item.get("text") or "")[:1500],
        }
        for item in chunks
    ]
    out = {
        "predicted_answer": raw.get("answer") or "",
        "retrieved_document_ids": list(raw.get("document_ids") or []),
        "context_char_count": int(raw.get("context_char_count") or 0),
        "context_token_count": int(raw.get("context_token_count") or 0),
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
        "evidence_used": evidence,
    }
    out.update(_ollama_stats())
    return out


def run_graphrag(question):
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    retrieved = retrieve_graph(question)
    evidence = retrieved.get("evidence") or []
    chunks = [
        {
            "doc_id": item.get("doc_id"),
            "title": item.get("title") or "",
            "text": item.get("text") or "",
        }
        for item in evidence
    ]
    context = _build_context(chunks) if chunks else ""
    if chunks:
        answer = ask_ollama(question, context) or INSUFFICIENT_EVIDENCE
    else:
        answer = INSUFFICIENT_EVIDENCE
    latency = time.perf_counter() - t0
    out = {
        "predicted_answer": answer,
        "retrieved_document_ids": list(retrieved.get("document_ids") or []),
        "context_char_count": len(context),
        "context_token_count": _count_tokens(context),
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
        "evidence_used": [
            {
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "text": (item.get("text") or "")[:1500],
            }
            for item in evidence
        ],
        "graph_paths": list(retrieved.get("traversals") or retrieved.get("graph_paths") or []),
    }
    out.update(_ollama_stats())
    return out


def run_agentic(question):
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    raw = investigate(question)
    latency = time.perf_counter() - t0
    state = raw.get("state") or {}
    evidence = state.get("evidence") or []
    blob = "\n\n".join(item.get("text") or "" for item in evidence)
    answer = raw.get("final_answer")
    if not answer:
        answer = INSUFFICIENT_EVIDENCE
    tools = [
        step.get("chosen_action")
        for step in (raw.get("trace") or [])
        if step.get("chosen_action") and step.get("chosen_action") != "stop"
    ]
    out = {
        "predicted_answer": answer,
        "retrieved_document_ids": list(state.get("document_ids") or []),
        "context_char_count": len(blob),
        "context_token_count": _count_tokens(blob),
        "agent_iterations": int(state.get("iteration_count") or 0),
        "agent_tool_calls": tools,
        "latency_sec": round(latency, 2),
        "agent_stop_reason": raw.get("stop_reason"),
        "agent_stopping_status": raw.get("stopping_status"),
        "investigation_kind": raw.get("investigation_kind"),
        "agent_trace": raw.get("trace") or [],
        "evidence_used": evidence,
        "graph_paths": list(state.get("graph_paths") or []),
        "identified_entities": state.get("identified_entities") or {},
        "unresolved_questions": list(state.get("unresolved_questions") or []),
        "supporting_evidence": bool(raw.get("supporting_evidence")),
    }
    out.update(_ollama_stats())
    return out


RUNNERS = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": run_agentic,
}


def _empty_raw(error, latency=0.0):
    return {
        "predicted_answer": "",
        "retrieved_document_ids": [],
        "context_char_count": 0,
        "context_token_count": 0,
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
        "evidence_used": [],
        "error": error,
        **{k: None for k in (
            "llm_prompt_eval_count", "llm_eval_count", "llm_total_tokens",
            "llm_prompt_eval_duration_ns", "llm_eval_duration_ns", "llm_total_duration_ns",
        )},
    }


def run_one(pipeline, question):
    t0 = time.perf_counter()
    try:
        return RUNNERS[pipeline](question)
    except Exception as exc:
        latency = time.perf_counter() - t0
        err = str(exc)
        if "timed out" in err.lower() or "timeout" in err.lower():
            err = f"timeout: {err}"
        return _empty_raw(f"{err}\n{traceback.format_exc(limit=4)}", latency=latency)


def _save(payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(payload.get("summary") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(record):
    with JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def preflight():
    problems = []
    print("PREFLIGHT")
    print("=" * 70)
    try:
        tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=15)
        names = [m.get("name") for m in (tags.json().get("models") or [])]
        print("Ollama:", tags.status_code, names)
        if tags.status_code != 200:
            problems.append("Ollama not healthy")
        if OLLAMA_MODEL not in names and not any(OLLAMA_MODEL in (n or "") for n in names):
            problems.append(f"model {OLLAMA_MODEL} not listed")
    except Exception as exc:
        problems.append(f"Ollama unreachable: {exc}")
        print("Ollama: FAIL", exc)
    try:
        url = (
            f"{TIGERGRAPH_REST_URL}/graph/{TIGERGRAPH_GRAPH}/vertices/Event?limit=1"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
            raw = json.loads(resp.read().decode("utf-8"))
        print("TigerGraph:", status, "error=", raw.get("error"), url)
        if raw.get("error"):
            problems.append(f"TigerGraph error {raw.get('error')}")
    except Exception as exc:
        problems.append(f"TigerGraph unreachable: {exc}")
        print("TigerGraph: FAIL", exc)
    for path in (CORPUS_PATH, CHUNKS_PATH, TFIDF_INDEX_PATH, PUBLIC_EVAL_PATH):
        exists = Path(path).exists()
        print("file:", path, "OK" if exists else "MISSING")
        if not exists:
            problems.append(f"missing {path}")
    try:
        from rag.pipeline import rag_answer as _ra
        from rag.graph_retriever import retrieve_graph as _rg
        from agent.orchestrator import investigate as _inv
        print("imports: rag_answer, retrieve_graph, investigate OK")
    except Exception as exc:
        problems.append(f"import failure: {exc}")
        print("imports: FAIL", exc)
    if OLD_PARTIAL.exists():
        print("preserving old partial:", OLD_PARTIAL)
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        n = len(existing.get("records") or [])
        print("existing final records:", n)
    else:
        print("output directory is new:", OUT_DIR)
    print("model:", OLLAMA_MODEL)
    print("pipelines:", ", ".join(PIPELINES))
    if problems:
        raise SystemExit("preflight failed: " + "; ".join(problems))
    print("PREFLIGHT OK")
    print()


def write_failures(rows):
    summary = summarize(rows)
    FAILURES_JSON.write_text(
        json.dumps(summary.get("failures") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with FAILURES_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pipeline", "qid", "qtype", "predicted_answer", "gold_answer",
                "gold_doc_retrieved", "retrieval_completeness", "error", "latency_sec",
            ],
        )
        writer.writeheader()
        for pipeline, items in (summary.get("failures") or {}).items():
            for row in items:
                gold = row.get("gold_answer")
                if isinstance(gold, list):
                    gold = " | ".join(str(x) for x in gold)
                writer.writerow({
                    "pipeline": pipeline,
                    "qid": row.get("qid"),
                    "qtype": row.get("qtype"),
                    "predicted_answer": row.get("predicted_answer"),
                    "gold_answer": gold,
                    "gold_doc_retrieved": row.get("gold_doc_retrieved"),
                    "retrieval_completeness": row.get("retrieval_completeness"),
                    "error": row.get("error"),
                    "latency_sec": row.get("latency_sec"),
                })


def _by_qid(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["qid"]][row["pipeline"]] = row
    return grouped


def agentic_effectiveness(rows):
    grouped = _by_qid(rows)
    agent_beats_rag = []
    agent_beats_graph = []
    agent_unnecessary = []
    agent_failed = []
    multi_action = 0
    minimal = 0
    agentic_rows = [r for r in rows if r["pipeline"] == "agentic"]
    n_agent = len(agentic_rows) or 1
    for row in agentic_rows:
        n_tools = len(row.get("agent_tool_calls") or [])
        if n_tools >= 2:
            multi_action += 1
        if n_tools <= 2:
            minimal += 1
    for qid, pipes in grouped.items():
        rag = pipes.get("rag") or {}
        graph = pipes.get("graphrag") or {}
        agent = pipes.get("agentic") or {}
        if not agent:
            continue
        a_ok = bool(agent.get("normalized_match"))
        r_ok = bool(rag.get("normalized_match"))
        g_ok = bool(graph.get("normalized_match"))
        rec = {
            "qid": qid,
            "qtype": agent.get("qtype"),
            "gold": agent.get("gold_answer"),
            "agentic": agent.get("predicted_answer"),
            "rag": rag.get("predicted_answer"),
            "graphrag": graph.get("predicted_answer"),
        }
        if a_ok and not r_ok:
            agent_beats_rag.append(rec)
        if a_ok and not g_ok:
            agent_beats_graph.append(rec)
        if a_ok and r_ok and g_ok:
            agent_unnecessary.append({"qid": qid, "qtype": agent.get("qtype")})
        if not a_ok:
            agent_failed.append({
                **rec,
                "stop_reason": agent.get("agent_stop_reason"),
                "kind": agent.get("investigation_kind"),
                "tools": agent.get("agent_tool_calls"),
            })
    rag_s = stats([r for r in rows if r["pipeline"] == "rag"])
    graph_s = stats([r for r in rows if r["pipeline"] == "graphrag"])
    agent_s = stats(agentic_rows)
    return {
        "accuracy_delta_vs_rag": round(
            agent_s["overall_accuracy"] - rag_s["overall_accuracy"], 4
        ),
        "accuracy_delta_vs_graphrag": round(
            agent_s["overall_accuracy"] - graph_s["overall_accuracy"], 4
        ),
        "avg_context_token_delta_vs_rag": round(
            agent_s["avg_context_tokens"] - rag_s["avg_context_tokens"], 1
        ),
        "avg_context_token_delta_vs_graphrag": round(
            agent_s["avg_context_tokens"] - graph_s["avg_context_tokens"], 1
        ),
        "avg_llm_total_token_delta_vs_rag": round(
            agent_s["avg_llm_total_tokens"] - rag_s["avg_llm_total_tokens"], 1
        ),
        "avg_llm_total_token_delta_vs_graphrag": round(
            agent_s["avg_llm_total_tokens"] - graph_s["avg_llm_total_tokens"], 1
        ),
        "avg_latency_delta_vs_rag": round(
            agent_s["avg_latency_sec"] - rag_s["avg_latency_sec"], 2
        ),
        "avg_latency_delta_vs_graphrag": round(
            agent_s["avg_latency_sec"] - graph_s["avg_latency_sec"], 2
        ),
        "avg_investigation_steps": agent_s["avg_agent_steps"],
        "pct_multiple_investigative_actions": round(100.0 * multi_action / n_agent, 2),
        "pct_minimal_investigation": round(100.0 * minimal / n_agent, 2),
        "n_agent_beats_rag": len(agent_beats_rag),
        "n_agent_beats_graphrag": len(agent_beats_graph),
        "n_agent_unnecessary_all_correct": len(agent_unnecessary),
        "n_agent_failed": len(agent_failed),
        "agent_beats_rag": agent_beats_rag,
        "agent_beats_graphrag": agent_beats_graph,
        "agent_unnecessary": agent_unnecessary,
        "agent_failed": agent_failed,
    }


def when_agent_needed(rows):
    grouped = _by_qid(rows)
    by_qtype = defaultdict(lambda: {
        "n": 0,
        "all_correct": 0,
        "agent_only": 0,
        "agent_needed_vs_rag": 0,
        "agent_needed_vs_graph": 0,
        "graph_enough": 0,
        "rag_enough": 0,
        "all_failed": 0,
        "agent_failed_graph_ok": 0,
    })
    examples = defaultdict(list)
    for qid, pipes in grouped.items():
        if len(pipes) < 3:
            continue
        qtype = (pipes.get("agentic") or pipes.get("rag") or {}).get("qtype")
        a = bool((pipes.get("agentic") or {}).get("normalized_match"))
        r = bool((pipes.get("rag") or {}).get("normalized_match"))
        g = bool((pipes.get("graphrag") or {}).get("normalized_match"))
        bucket = by_qtype[qtype]
        bucket["n"] += 1
        if a and r and g:
            bucket["all_correct"] += 1
            examples[qtype].append({"qid": qid, "pattern": "all_correct"})
        elif a and not r and not g:
            bucket["agent_only"] += 1
            examples[qtype].append({"qid": qid, "pattern": "agent_only"})
        if a and not r:
            bucket["agent_needed_vs_rag"] += 1
        if a and not g:
            bucket["agent_needed_vs_graph"] += 1
        if g and not a:
            bucket["agent_failed_graph_ok"] += 1
            examples[qtype].append({"qid": qid, "pattern": "graph_enough_agent_failed"})
        if g and r:
            bucket["graph_enough"] += 1
        if r and not a:
            bucket["rag_enough"] += 1
        if not a and not r and not g:
            bucket["all_failed"] += 1
            examples[qtype].append({"qid": qid, "pattern": "all_failed"})
    return {"by_qtype": dict(by_qtype), "examples": dict(examples)}


def write_report(rows):
    summary = summarize(rows)
    effect = agentic_effectiveness(rows)
    need = when_agent_needed(rows)
    overall = summary["overall"]
    lines = []
    add = lines.append
    add("# Final public benchmark report")
    add("")
    add("Official 100-question run of RAG, GraphRAG, and Agentic GraphRAG.")
    add("Old partial 86-record results were not mixed into this file.")
    add("")
    add("## 1. Overall comparison")
    add("")
    add("| Pipeline | n | Accuracy | Retrieval success | Completeness | Avg latency (s) | Median latency (s) | Avg context tokens | Avg LLM total tokens | Token efficiency /1k | Avg steps | Avg tools | Failures | Timeouts | Errors |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in PIPELINES:
        row = overall[name]
        add(
            f"| {name} | {row['n']} | {row['overall_accuracy']:.4f} | "
            f"{row['retrieval_success']:.4f} | {row['retrieval_completeness']:.4f} | "
            f"{row['avg_latency_sec']:.2f} | {row['median_latency_sec']:.2f} | "
            f"{row['avg_context_tokens']:.1f} | {row['avg_llm_total_tokens']:.1f} | "
            f"{row['token_efficiency_per_1k_tokens']:.4f} | {row['avg_agent_steps']:.2f} | "
            f"{row['avg_tool_calls']:.2f} | {row['n_failures']} | {row['n_timeouts']} | "
            f"{row['n_errors']} |"
        )
    add("")
    add("## 2. Qtype breakdown")
    add("")
    counts = summary.get("qtype_counts") or {}
    for qtype in ALL_TYPES:
        add(f"### {qtype} (n={counts.get(qtype, 0)})")
        add("")
        add("| Pipeline | Accuracy | Retrieval success | Completeness | Avg latency | Avg context tokens | Avg steps |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for name in PIPELINES:
            row = summary["by_qtype"][name][qtype]
            add(
                f"| {name} | {row['overall_accuracy']:.4f} | {row['retrieval_success']:.4f} | "
                f"{row['retrieval_completeness']:.4f} | {row['avg_latency_sec']:.2f} | "
                f"{row['avg_context_tokens']:.1f} | {row['avg_agent_steps']:.2f} |"
            )
        add("")
    add("## 3. Failure analysis")
    add("")
    for name in PIPELINES:
        fails = summary["failures"][name]
        add(f"### {name} ({len(fails)} failures)")
        add("")
        if not fails:
            add("None.")
            add("")
            continue
        add("| qid | qtype | gold | predicted | gold doc retrieved | error |")
        add("|---|---|---|---|---|---|")
        for row in fails:
            gold = row.get("gold_answer")
            if isinstance(gold, list):
                gold = " / ".join(str(x) for x in gold)
            pred = str(row.get("predicted_answer") or "").replace("|", "/")
            gold = str(gold or "").replace("|", "/")
            err_lines = str(row.get("error") or "").splitlines()
            err = (err_lines[0] if err_lines else "").replace("|", "/")[:80]
            add(
                f"| {row.get('qid')} | {row.get('qtype')} | {gold} | {pred[:80]} | "
                f"{row.get('gold_doc_retrieved')} | {err} |"
            )
        add("")
    add("Agentic failure categories:")
    add("")
    for name, count in sorted(
        (summary.get("agentic_failure_categories") or {}).items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        add(f"- {name}: {count}")
    add("")
    add("## 4. Agentic effectiveness")
    add("")
    add(f"- Accuracy vs RAG: {effect['accuracy_delta_vs_rag']:+.4f}")
    add(f"- Accuracy vs GraphRAG: {effect['accuracy_delta_vs_graphrag']:+.4f}")
    add(f"- Avg context-token delta vs RAG: {effect['avg_context_token_delta_vs_rag']:+.1f}")
    add(f"- Avg context-token delta vs GraphRAG: {effect['avg_context_token_delta_vs_graphrag']:+.1f}")
    add(f"- Avg LLM-total-token delta vs RAG: {effect['avg_llm_total_token_delta_vs_rag']:+.1f}")
    add(f"- Avg LLM-total-token delta vs GraphRAG: {effect['avg_llm_total_token_delta_vs_graphrag']:+.1f}")
    add(f"- Avg latency delta vs RAG: {effect['avg_latency_delta_vs_rag']:+.2f}s")
    add(f"- Avg latency delta vs GraphRAG: {effect['avg_latency_delta_vs_graphrag']:+.2f}s")
    add(f"- Average investigation steps: {effect['avg_investigation_steps']}")
    add(f"- Multiple investigative actions: {effect['pct_multiple_investigative_actions']}%")
    add(f"- Minimal investigation (≤2 tools): {effect['pct_minimal_investigation']}%")
    add(f"- Agentic correct and RAG wrong: {effect['n_agent_beats_rag']}")
    add(f"- Agentic correct and GraphRAG wrong: {effect['n_agent_beats_graphrag']}")
    add(f"- All three correct (agent unnecessary for correctness): {effect['n_agent_unnecessary_all_correct']}")
    add(f"- Agentic still failed: {effect['n_agent_failed']}")
    add("")
    add("## 5. Which questions need an agent?")
    add("")
    add("Measured from this 100-question public set, not a generic claim.")
    add("")
    for qtype in ALL_TYPES:
        bucket = need["by_qtype"].get(qtype) or {}
        n = bucket.get("n") or 0
        if not n:
            continue
        add(f"### {qtype} (n={n})")
        add("")
        add(f"- All three correct: {bucket.get('all_correct', 0)} ({100.0 * bucket.get('all_correct', 0) / n:.1f}%)")
        add(f"- Agentic only correct: {bucket.get('agent_only', 0)} ({100.0 * bucket.get('agent_only', 0) / n:.1f}%)")
        add(f"- Agentic needed vs RAG: {bucket.get('agent_needed_vs_rag', 0)}")
        add(f"- Agentic needed vs GraphRAG: {bucket.get('agent_needed_vs_graph', 0)}")
        add(f"- GraphRAG correct and Agentic wrong: {bucket.get('agent_failed_graph_ok', 0)}")
        add(f"- All three failed: {bucket.get('all_failed', 0)}")
        add("")
    add("Interpretation:")
    add("")
    conclusions = []
    for qtype in ALL_TYPES:
        bucket = need["by_qtype"].get(qtype) or {}
        n = bucket.get("n") or 0
        if not n:
            continue
        agent_only_pct = 100.0 * bucket.get("agent_only", 0) / n
        all_ok_pct = 100.0 * bucket.get("all_correct", 0) / n
        vs_graph = bucket.get("agent_needed_vs_graph", 0)
        if agent_only_pct >= 30 or vs_graph >= max(2, n * 0.25):
            conclusions.append(
                f"- **{qtype}: agent is worth it.** Agentic-only correctness is {agent_only_pct:.1f}% "
                f"and Agentic beats GraphRAG on {vs_graph}/{n} questions."
            )
        elif all_ok_pct >= 50 and vs_graph <= 1:
            conclusions.append(
                f"- **{qtype}: agent is usually unnecessary for correctness.** "
                f"All three already match on {all_ok_pct:.1f}% of questions."
            )
        else:
            conclusions.append(
                f"- **{qtype}: mixed.** All-correct {all_ok_pct:.1f}%; "
                f"Agentic-only {agent_only_pct:.1f}%; Agentic beats GraphRAG on {vs_graph}/{n}."
            )
    if conclusions:
        lines.extend(conclusions)
    else:
        add("- Not enough complete triples to judge.")
    add("")
    add("## 6. Artifacts")
    add("")
    add(f"- `{RESULTS_PATH.as_posix()}`")
    add(f"- `{SUMMARY_PATH.as_posix()}`")
    add(f"- `{REPORT_PATH.as_posix()}`")
    add(f"- `{FAILURES_JSON.as_posix()}`")
    add(f"- `{FAILURES_CSV.as_posix()}`")
    add("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    extra = {
        "overall": overall,
        "by_qtype": summary["by_qtype"],
        "qtype_counts": summary.get("qtype_counts"),
        "agentic_effectiveness": {
            k: v for k, v in effect.items()
            if k not in {
                "agent_beats_rag", "agent_beats_graphrag",
                "agent_unnecessary", "agent_failed",
            }
        },
        "agentic_effectiveness_examples": {
            "beats_rag": effect["agent_beats_rag"],
            "beats_graphrag": effect["agent_beats_graphrag"],
            "unnecessary": effect["agent_unnecessary"],
            "failed": effect["agent_failed"],
        },
        "when_agent_needed": need,
        "agentic_failure_categories": summary.get("agentic_failure_categories"),
    }
    SUMMARY_PATH.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    write_failures(rows)


def print_table(summary):
    print()
    print("FINAL PUBLIC BENCHMARK COMPARISON")
    print("=" * 130)
    header = (
        f"{'Pipeline':<16} {'Accuracy':>10} {'Retrieval':>12} {'Completeness':>14} "
        f"{'AvgLat':>10} {'MedLat':>10} {'Tokens':>10} {'LLMTot':>10} {'Steps':>8}"
    )
    print(header)
    print("-" * 130)
    for name in PIPELINES:
        row = summary["overall"][name]
        print(
            f"{name:<16} {row['overall_accuracy']:>10.4f} "
            f"{row['retrieval_success']:>12.4f} {row['retrieval_completeness']:>14.4f} "
            f"{row['avg_latency_sec']:>10.2f} {row['median_latency_sec']:>10.2f} "
            f"{row['avg_context_tokens']:>10.1f} {row['avg_llm_total_tokens']:>10.1f} "
            f"{row['avg_agent_steps']:>8.2f}"
        )


def main():
    questions = load_all_questions()
    counts = Counter(row.get("qtype") for row in questions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight()
    print("FINAL PUBLIC BENCHMARK")
    print("=" * 70)
    print("source:", PUBLIC_EVAL_PATH)
    print("n_questions:", len(questions))
    print("qtype_counts:", dict(counts))
    print("output:", OUT_DIR.resolve())
    print("old partial preserved:", OLD_PARTIAL)
    print("Ollama model:", OLLAMA_MODEL)
    print("pipelines:", ", ".join(PIPELINES))
    print("save: after every pipeline-question pair")
    print()

    records = []
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            existing_questions = existing.get("questions") or []
            if len(existing_questions) == 100:
                records = list(existing.get("records") or [])
        except Exception:
            records = []
    done = {(row.get("pipeline"), row.get("qid")) for row in records}
    payload = {
        "run": "official_final_public",
        "n_questions": 100,
        "slice": None,
        "model": OLLAMA_MODEL,
        "questions": [
            {"qid": q["qid"], "qtype": q["qtype"], "question": q["question"]}
            for q in questions
        ],
        "records": records,
        "summary": summarize(records) if records else {},
    }
    _save(payload)
    print(f"already recorded: {len(done)} / 300")

    total = 300
    finished = len(done)
    for idx, item in enumerate(questions, start=1):
        gold_docs = list(item.get("gold_doc_ids") or [])
        gold_answer = item.get("answer")
        for pipeline in PIPELINES:
            if (pipeline, item["qid"]) in done:
                print(
                    f"[{idx}/100] skipping {pipeline} {item['qid']} (already recorded)",
                    flush=True,
                )
                continue
            print(f"[{idx}/100] running {pipeline} {item['qid']} ...", flush=True)
            raw = run_one(pipeline, item["question"])
            exact, normalized = match_answer(raw.get("predicted_answer"), gold_answer)
            hit, found = gold_doc_hit(gold_docs, raw.get("retrieved_document_ids"))
            record = {
                "pipeline": pipeline,
                "qid": item["qid"],
                "qtype": item["qtype"],
                "question": item["question"],
                "gold_answer": gold_answer,
                "predicted_answer": raw.get("predicted_answer") or "",
                "exact_match": exact,
                "normalized_match": normalized,
                "gold_doc_ids": gold_docs,
                "retrieved_document_ids": raw.get("retrieved_document_ids") or [],
                "gold_doc_retrieved": hit,
                "gold_docs_found": found,
                "retrieval_completeness": round(
                    _retrieval_recall(gold_docs, raw.get("retrieved_document_ids")), 4
                ),
                "context_char_count": raw.get("context_char_count") or 0,
                "context_token_count": raw.get("context_token_count") or 0,
                "llm_prompt_eval_count": raw.get("llm_prompt_eval_count"),
                "llm_eval_count": raw.get("llm_eval_count"),
                "llm_total_tokens": raw.get("llm_total_tokens"),
                "agent_iterations": raw.get("agent_iterations") or 0,
                "agent_tool_calls": raw.get("agent_tool_calls") or [],
                "latency_sec": raw.get("latency_sec") or 0.0,
                "evidence_used": raw.get("evidence_used") or [],
            }
            if pipeline == "agentic":
                record["investigation_kind"] = raw.get("investigation_kind")
                record["agent_stop_reason"] = raw.get("agent_stop_reason")
                record["agent_stopping_status"] = raw.get("agent_stopping_status")
                record["agent_trace"] = raw.get("agent_trace") or []
                record["graph_paths"] = raw.get("graph_paths") or []
                record["identified_entities"] = raw.get("identified_entities") or {}
                record["unresolved_questions"] = raw.get("unresolved_questions") or []
                record["supporting_evidence"] = raw.get("supporting_evidence")
                record["failure_category"] = classify_agentic_failure(record)
            if raw.get("graph_paths") and pipeline != "agentic":
                record["graph_paths"] = raw.get("graph_paths")
            if raw.get("error"):
                record["error"] = raw["error"]
                if pipeline == "agentic":
                    record["failure_category"] = classify_agentic_failure(record)
            records.append(record)
            done.add((pipeline, item["qid"]))
            finished += 1
            payload["records"] = records
            payload["summary"] = summarize(records)
            _save(payload)
            _append_jsonl(record)
            _safe_print(
                f"  pred={record['predicted_answer']!r} "
                f"norm_match={normalized} gold_doc={hit} "
                f"latency={record['latency_sec']}s "
                f"progress={finished}/{total}"
            )

    payload["records"] = records
    payload["summary"] = summarize(records)
    _save(payload)
    write_report(records)
    print_table(payload["summary"])
    print()
    print("wrote", RESULTS_PATH.resolve())
    print("report", REPORT_PATH.resolve())
    print("FINAL PUBLIC BENCHMARK COMPLETE")
    if len(records) != 300:
        raise SystemExit(f"incomplete run: {len(records)} records, expected 300")
    qids = {row["qid"] for row in records}
    if len(qids) != 100:
        raise SystemExit(f"expected 100 qids, found {len(qids)}")
    for name in PIPELINES:
        n = sum(1 for row in records if row["pipeline"] == name)
        if n != 100:
            raise SystemExit(f"{name} has {n} records, expected 100")


if __name__ == "__main__":
    main()
