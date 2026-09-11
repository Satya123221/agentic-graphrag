"""Hidden-set evaluation runner for RAG, GraphRAG, and Agentic GraphRAG.

Answerless evaluation only: records raw pipeline outputs and execution
statistics. Does not score accuracy, match gold labels, or modify pipelines.
Writes only to eval/results/hidden_benchmark/.
"""

import json
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.append(".")

import requests

from config import (
    CORPUS_PATH,
    HIDDEN_EVAL_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TIGERGRAPH_GRAPH,
    TIGERGRAPH_REST_URL,
)
from agent.orchestrator import investigate
from rag.graph_retriever import retrieve_graph
from rag.pipeline import INSUFFICIENT_EVIDENCE, _build_context, _count_tokens, ask_ollama, rag_answer

PIPELINES = ("rag", "graphrag", "agentic")
EXPECTED_QUESTIONS = 50
EXPECTED_RECORDS = 150
OUT_DIR = Path("eval") / "results" / "hidden_benchmark"
JSONL_PATH = OUT_DIR / "hidden_results.jsonl"
RESULTS_PATH = OUT_DIR / "hidden_results.json"
SUMMARY_PATH = OUT_DIR / "hidden_summary.json"
REPORT_PATH = OUT_DIR / "hidden_report.md"

LLM_KEYS = (
    "llm_prompt_eval_count",
    "llm_eval_count",
    "llm_total_tokens",
    "llm_prompt_eval_duration_ns",
    "llm_eval_duration_ns",
    "llm_total_duration_ns",
)

OLLAMA_LAST = {}
GOLD_FIELDS = ("gold_answer", "gold_answers", "gold_doc_ids", "gold")


def _safe_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    kwargs.setdefault("flush", True)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))
        sys.stdout.buffer.flush()


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    return float(statistics.median(values))


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


@contextmanager
def capture_ollama_stats(enabled=True):
    """Capture Ollama generate token metadata without changing pipeline code."""
    OLLAMA_LAST.clear()
    if not enabled:
        yield OLLAMA_LAST
        return
    original = requests.post

    def wrapping(*args, **kwargs):
        response = original(*args, **kwargs)
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

    requests.post = wrapping
    try:
        yield OLLAMA_LAST
    finally:
        requests.post = original


def load_hidden_questions(path=None):
    """Load answerless hidden questions. Gold fields are ignored if present."""
    path = Path(path) if path is not None else HIDDEN_EVAL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"hidden eval file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = raw.get("qid")
            question = raw.get("question")
            if not qid or not question:
                raise ValueError(f"hidden question on line {line_no} is missing qid or question")
            rows.append({
                "qid": qid,
                "question": question,
                "qtype": raw.get("qtype"),
            })
    qids = [row["qid"] for row in rows]
    if len(qids) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} hidden questions, found {len(qids)}"
        )
    if len(set(qids)) != EXPECTED_QUESTIONS:
        raise ValueError("hidden questions do not contain 50 unique qids")
    return rows


def iter_jobs(questions, pipelines=PIPELINES):
    for item in questions:
        for pipeline in pipelines:
            yield item, pipeline


def job_key(qid, pipeline):
    return (qid, pipeline)


def load_completed(jsonl_path, results_path=None):
    """Load unique completed qid/pipeline records. First occurrence wins."""
    records = []
    seen = set()
    jsonl_path = Path(jsonl_path)
    results_path = Path(results_path) if results_path is not None else None
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = job_key(rec.get("qid"), rec.get("pipeline"))
                if not rec.get("qid") or not rec.get("pipeline") or key in seen:
                    continue
                seen.add(key)
                records.append(rec)
        return records, seen
    if results_path and results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        for rec in payload.get("records") or []:
            key = job_key(rec.get("qid"), rec.get("pipeline"))
            if not rec.get("qid") or not rec.get("pipeline") or key in seen:
                continue
            seen.add(key)
            records.append(rec)
    return records, seen


def _append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def run_rag_hidden(question, rag_fn=None):
    runner = rag_fn or rag_answer
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    raw = runner(question) or {}
    latency = round(time.perf_counter() - t0, 2)
    chunks = raw.get("retrieved_chunks") or raw.get("evidence") or []
    out = {
        "answer": raw.get("answer") if raw.get("answer") is not None else "",
        "retrieved_document_ids": list(raw.get("document_ids") or raw.get("retrieved_document_ids") or []),
        "evidence": list(chunks),
        "similarity_scores": list(raw.get("similarity_scores") or []),
        "context_char_count": int(raw.get("context_char_count") or 0),
        "context_token_count": int(raw.get("context_token_count") or 0),
        "latency_seconds": latency,
        "error": None,
    }
    out.update(_ollama_stats())
    return out


def run_graphrag_hidden(question, retrieve_fn=None, generate_fn=None, build_context_fn=None):
    retrieve = retrieve_fn or retrieve_graph
    generate = generate_fn or ask_ollama
    build_context = build_context_fn or _build_context
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    retrieved = retrieve(question) or {}
    evidence = retrieved.get("evidence") or []
    chunks = [
        {
            "doc_id": item.get("doc_id"),
            "title": item.get("title") or "",
            "text": item.get("text") or "",
        }
        for item in evidence
    ]
    context = build_context(chunks) if chunks else ""
    if chunks:
        answer = generate(question, context) or INSUFFICIENT_EVIDENCE
    else:
        answer = INSUFFICIENT_EVIDENCE
    latency = round(time.perf_counter() - t0, 2)
    out = {
        "answer": answer,
        "matched_entities": retrieved.get("matched_entities") or {},
        "graph_paths": list(retrieved.get("graph_paths") or retrieved.get("traversals") or []),
        "retrieved_document_ids": list(retrieved.get("document_ids") or []),
        "evidence": list(evidence),
        "scores": list(retrieved.get("scores") or []),
        "stats": dict(retrieved.get("stats") or {}),
        "context_char_count": len(context),
        "context_token_count": _count_tokens(context) if context else 0,
        "latency_seconds": latency,
        "error": None,
    }
    out.update(_ollama_stats())
    return out


def run_agentic_hidden(question, investigate_fn=None):
    runner = investigate_fn or investigate
    OLLAMA_LAST.clear()
    t0 = time.perf_counter()
    raw = runner(question) or {}
    latency = round(time.perf_counter() - t0, 2)
    state = raw.get("state") or {}
    evidence = list(state.get("evidence") or raw.get("evidence") or [])
    blob = "\n\n".join(item.get("text") or "" for item in evidence if isinstance(item, dict))
    trace = list(raw.get("trace") or raw.get("agent_trace") or [])
    tools = [
        step.get("chosen_action")
        for step in trace
        if isinstance(step, dict) and step.get("chosen_action") and step.get("chosen_action") != "stop"
    ]
    document_ids = list(state.get("document_ids") or raw.get("retrieved_document_ids") or [])
    if not document_ids:
        document_ids = [
            item.get("doc_id") for item in evidence
            if isinstance(item, dict) and item.get("doc_id")
        ]
    answer = raw.get("final_answer")
    if answer is None:
        answer = raw.get("answer")
    if answer is None:
        answer = ""
    out = {
        "answer": answer,
        "agent_trace": trace,
        "identified_entities": state.get("identified_entities") or raw.get("identified_entities") or {},
        "graph_paths": list(state.get("graph_paths") or raw.get("graph_paths") or []),
        "evidence": evidence,
        "retrieved_document_ids": document_ids,
        "agent_tool_calls": tools,
        "tools_used": list(raw.get("tools_used") or tools),
        "agent_iterations": int(state.get("iteration_count") or raw.get("agent_iterations") or 0),
        "unresolved_questions": list(
            state.get("unresolved_questions") or raw.get("unresolved_questions") or []
        ),
        "agent_stop_reason": raw.get("stop_reason") or raw.get("agent_stop_reason"),
        "agent_stopping_status": raw.get("stopping_status") or raw.get("agent_stopping_status"),
        "investigation_kind": raw.get("investigation_kind"),
        "supporting_evidence": raw.get("supporting_evidence"),
        "context_char_count": len(blob),
        "context_token_count": _count_tokens(blob) if blob else 0,
        "latency_seconds": latency,
        "error": None,
    }
    out.update(_ollama_stats())
    return out


def default_runners(rag_fn=None, retrieve_fn=None, generate_fn=None, investigate_fn=None):
    return {
        "rag": lambda question: run_rag_hidden(question, rag_fn=rag_fn),
        "graphrag": lambda question: run_graphrag_hidden(
            question, retrieve_fn=retrieve_fn, generate_fn=generate_fn
        ),
        "agentic": lambda question: run_agentic_hidden(
            question, investigate_fn=investigate_fn
        ),
    }


def _empty_raw(error, latency=0.0):
    payload = {
        "answer": "",
        "retrieved_document_ids": [],
        "evidence": [],
        "latency_seconds": round(latency, 2),
        "error": error,
        "agent_trace": [],
        "identified_entities": {},
        "graph_paths": [],
        "agent_tool_calls": [],
        "tools_used": [],
        "agent_iterations": 0,
        "unresolved_questions": [],
        "matched_entities": {},
        "similarity_scores": [],
        "scores": [],
        "stats": {},
        "context_char_count": 0,
        "context_token_count": 0,
    }
    payload.update({key: None for key in LLM_KEYS})
    return payload


def run_one(pipeline, question, runners):
    t0 = time.perf_counter()
    try:
        raw = runners[pipeline](question) or {}
        if "latency_seconds" not in raw:
            raw["latency_seconds"] = round(time.perf_counter() - t0, 2)
        raw.setdefault("error", None)
        return raw
    except Exception as exc:
        latency = time.perf_counter() - t0
        err = str(exc)
        if "timed out" in err.lower() or "timeout" in err.lower():
            err = f"timeout: {err}"
        else:
            err = f"{err}\n{traceback.format_exc(limit=4)}"
        return _empty_raw(err, latency=latency)


def build_record(item, pipeline, raw):
    raw = raw or {}
    record = {
        "qid": item["qid"],
        "question": item["question"],
        "qtype": item.get("qtype"),
        "pipeline": pipeline,
        "answer": raw.get("answer") if raw.get("answer") is not None else "",
        "latency_seconds": float(raw.get("latency_seconds") or 0.0),
        "error": raw.get("error"),
        "retrieved_document_ids": list(raw.get("retrieved_document_ids") or []),
        "evidence": list(raw.get("evidence") or []),
        "context_char_count": int(raw.get("context_char_count") or 0),
        "context_token_count": int(raw.get("context_token_count") or 0),
    }
    for key in LLM_KEYS:
        record[key] = raw.get(key)
    if pipeline == "rag":
        record["similarity_scores"] = list(raw.get("similarity_scores") or [])
    if pipeline == "graphrag":
        record["matched_entities"] = raw.get("matched_entities") or {}
        record["graph_paths"] = list(raw.get("graph_paths") or [])
        record["scores"] = list(raw.get("scores") or [])
        record["stats"] = dict(raw.get("stats") or {})
    if pipeline == "agentic":
        record["agent_trace"] = list(raw.get("agent_trace") or [])
        record["identified_entities"] = raw.get("identified_entities") or {}
        record["graph_paths"] = list(raw.get("graph_paths") or [])
        record["agent_tool_calls"] = list(raw.get("agent_tool_calls") or [])
        record["tools_used"] = list(raw.get("tools_used") or record["agent_tool_calls"])
        record["agent_iterations"] = int(raw.get("agent_iterations") or 0)
        record["unresolved_questions"] = list(raw.get("unresolved_questions") or [])
        record["agent_stop_reason"] = raw.get("agent_stop_reason")
        record["agent_stopping_status"] = raw.get("agent_stopping_status")
        record["investigation_kind"] = raw.get("investigation_kind")
        record["supporting_evidence"] = raw.get("supporting_evidence")
    for field in GOLD_FIELDS:
        record.pop(field, None)
    record.pop("exact_match", None)
    record.pop("normalized_match", None)
    return record


def _is_timeout(error):
    text = (error or "").lower()
    return "timeout" in text or "timed out" in text


def _pipeline_stats(items):
    n = len(items)
    latencies = [float(r.get("latency_seconds") or 0.0) for r in items]
    errors = [r for r in items if r.get("error")]
    timeouts = [r for r in items if _is_timeout(r.get("error"))]
    success = n - len(errors)
    llm_prompt = [float(r["llm_prompt_eval_count"]) for r in items if r.get("llm_prompt_eval_count") is not None]
    llm_out = [float(r["llm_eval_count"]) for r in items if r.get("llm_eval_count") is not None]
    llm_tot = [float(r["llm_total_tokens"]) for r in items if r.get("llm_total_tokens") is not None]
    with_llm = sum(
        1 for r in items
        if r.get("llm_prompt_eval_count") is not None or r.get("llm_eval_count") is not None
    )
    stats = {
        "n": n,
        "n_success": success,
        "n_failed": len(errors),
        "n_timeouts": len(timeouts),
        "avg_latency_seconds": round(_mean(latencies), 2),
        "median_latency_seconds": round(_median(latencies), 2),
        "min_latency_seconds": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_seconds": round(max(latencies), 2) if latencies else 0.0,
        "n_with_ollama_token_metadata": with_llm,
        "avg_llm_prompt_tokens": round(_mean(llm_prompt), 1) if llm_prompt else 0.0,
        "avg_llm_output_tokens": round(_mean(llm_out), 1) if llm_out else 0.0,
        "avg_llm_total_tokens": round(_mean(llm_tot), 1) if llm_tot else 0.0,
    }
    if items and items[0].get("pipeline") == "agentic":
        steps = [int(r.get("agent_iterations") or 0) for r in items]
        tool_counts = [len(r.get("agent_tool_calls") or r.get("tools_used") or []) for r in items]
        stats.update({
            "avg_agent_steps": round(_mean(steps), 2),
            "median_agent_steps": round(_median(steps), 2),
            "min_agent_steps": min(steps) if steps else 0,
            "max_agent_steps": max(steps) if steps else 0,
            "avg_tool_calls": round(_mean(tool_counts), 2),
            "n_with_trace": sum(1 for r in items if r.get("agent_trace")),
        })
    return stats


def summarize_hidden(records, questions=None):
    """Descriptive execution statistics only. No accuracy metrics."""
    by_pipeline = {name: [] for name in PIPELINES}
    for row in records:
        name = row.get("pipeline")
        if name in by_pipeline:
            by_pipeline[name].append(row)
    questions = questions or []
    qtype_counts = {}
    for row in questions:
        qtype = row.get("qtype") or "unknown"
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1
    agentic = by_pipeline["agentic"]
    return {
        "n_questions": len(questions) or len({r.get("qid") for r in records}),
        "n_records": len(records),
        "expected_records": EXPECTED_RECORDS,
        "records_per_pipeline": {name: len(items) for name, items in by_pipeline.items()},
        "n_success": sum(1 for r in records if not r.get("error")),
        "n_failed": sum(1 for r in records if r.get("error")),
        "n_timeouts": sum(1 for r in records if _is_timeout(r.get("error"))),
        "qtype_counts": qtype_counts,
        "by_pipeline": {name: _pipeline_stats(items) for name, items in by_pipeline.items()},
        "latency_seconds": {
            "avg": round(_mean([float(r.get("latency_seconds") or 0.0) for r in records]), 2),
            "median": round(_median([float(r.get("latency_seconds") or 0.0) for r in records]), 2),
        },
        "agentic_steps": {
            "avg": round(_mean([int(r.get("agent_iterations") or 0) for r in agentic]), 2),
            "median": round(_median([int(r.get("agent_iterations") or 0) for r in agentic]), 2),
            "n_with_trace": sum(1 for r in agentic if r.get("agent_trace")),
        },
        "ollama_usage": {
            "n_with_token_metadata": sum(
                1 for r in records
                if r.get("llm_prompt_eval_count") is not None or r.get("llm_eval_count") is not None
            ),
            "avg_llm_total_tokens": round(
                _mean([
                    float(r["llm_total_tokens"])
                    for r in records
                    if r.get("llm_total_tokens") is not None
                ]),
                1,
            ),
        },
        "complete": len(records) == EXPECTED_RECORDS and all(
            len(items) == EXPECTED_QUESTIONS for items in by_pipeline.values()
        ),
    }


def write_report(records, summary, report_path=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    report_path = Path(report_path) if report_path is not None else out_dir / "hidden_report.md"
    jsonl_path = out_dir / "hidden_results.jsonl"
    results_path = out_dir / "hidden_results.json"
    summary_path = out_dir / "hidden_summary.json"
    lines = [
        "# Hidden evaluation report",
        "",
        "Answerless execution report. No accuracy is computed because hidden questions have no gold answers.",
        "",
        "## Counts",
        "",
        f"- Questions: {summary.get('n_questions')}",
        f"- Records: {summary.get('n_records')} / {summary.get('expected_records')}",
        f"- Successful executions: {summary.get('n_success')}",
        f"- Failed executions: {summary.get('n_failed')}",
        f"- Timeouts: {summary.get('n_timeouts')}",
        f"- Complete: {summary.get('complete')}",
        "",
        "Records per pipeline:",
        "",
    ]
    for name in PIPELINES:
        n = (summary.get("records_per_pipeline") or {}).get(name, 0)
        lines.append(f"- {name}: {n}")
    lines.extend(["", "## Latency (seconds)", "", "| Pipeline | n | success | failed | timeouts | avg | median | min | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name in PIPELINES:
        row = (summary.get("by_pipeline") or {}).get(name) or {}
        lines.append(
            f"| {name} | {row.get('n', 0)} | {row.get('n_success', 0)} | "
            f"{row.get('n_failed', 0)} | {row.get('n_timeouts', 0)} | "
            f"{row.get('avg_latency_seconds', 0)} | {row.get('median_latency_seconds', 0)} | "
            f"{row.get('min_latency_seconds', 0)} | {row.get('max_latency_seconds', 0)} |"
        )
    agent = (summary.get("by_pipeline") or {}).get("agentic") or {}
    lines.extend([
        "",
        "## Agentic steps",
        "",
        f"- Average steps: {agent.get('avg_agent_steps', 0)}",
        f"- Median steps: {agent.get('median_agent_steps', 0)}",
        f"- Min / max steps: {agent.get('min_agent_steps', 0)} / {agent.get('max_agent_steps', 0)}",
        f"- Average tool calls: {agent.get('avg_tool_calls', 0)}",
        f"- Records with investigation trace: {agent.get('n_with_trace', 0)}",
        "",
        "## Ollama usage",
        "",
        f"- Records with token metadata: {(summary.get('ollama_usage') or {}).get('n_with_token_metadata', 0)}",
        f"- Average total LLM tokens (where available): {(summary.get('ollama_usage') or {}).get('avg_llm_total_tokens', 0)}",
        "",
        "## Question types in the hidden file",
        "",
    ])
    for qtype, count in (summary.get("qtype_counts") or {}).items():
        lines.append(f"- {qtype}: {count}")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- `{jsonl_path.as_posix()}`",
        f"- `{results_path.as_posix()}`",
        f"- `{summary_path.as_posix()}`",
        f"- `{report_path.as_posix()}`",
        "",
        "Resume by rerunning `python -m eval.run_hidden`. Completed qid/pipeline pairs are skipped.",
        "",
    ])
    _atomic_write_text(report_path, "\n".join(lines))


def write_artifacts(questions, records, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    summary = summarize_hidden(records, questions=questions)
    payload = {
        "run": "hidden_benchmark",
        "n_questions": len(questions),
        "n_records": len(records),
        "model": OLLAMA_MODEL,
        "questions": [
            {"qid": q["qid"], "qtype": q.get("qtype"), "question": q["question"]}
            for q in questions
        ],
        "records": records,
        "summary": summary,
    }
    _atomic_write_json(out_dir / "hidden_results.json", payload)
    _atomic_write_json(out_dir / "hidden_summary.json", summary)
    write_report(records, summary, out_dir=out_dir)
    return payload


def preflight(eval_path=HIDDEN_EVAL_PATH):
    problems = []
    print("HIDDEN PREFLIGHT")
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
        import urllib.request
        url = f"{TIGERGRAPH_REST_URL}/graph/{TIGERGRAPH_GRAPH}/vertices/Event?limit=1"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
            raw = json.loads(resp.read().decode("utf-8"))
        print("TigerGraph:", status, "error=", raw.get("error"))
        if raw.get("error"):
            problems.append(f"TigerGraph error {raw.get('error')}")
    except Exception as exc:
        problems.append(f"TigerGraph unreachable: {exc}")
        print("TigerGraph: FAIL", exc)
    exists = Path(eval_path).exists()
    print("file:", eval_path, "OK" if exists else "MISSING")
    if not exists:
        problems.append(f"missing {eval_path}")
    corpus_ok = Path(CORPUS_PATH).exists()
    print("file:", CORPUS_PATH, "OK" if corpus_ok else "MISSING")
    if not corpus_ok:
        problems.append(f"missing {CORPUS_PATH}")
    print("model:", OLLAMA_MODEL)
    print("pipelines:", ", ".join(PIPELINES))
    if problems:
        raise SystemExit("preflight failed: " + "; ".join(problems))
    print("PREFLIGHT OK")
    print()


def run_hidden(
    eval_path=None,
    out_dir=None,
    runners=None,
    do_preflight=False,
    capture_llm=None,
):
    """Run or resume the hidden evaluation. Never scores accuracy."""
    eval_path = Path(eval_path) if eval_path is not None else HIDDEN_EVAL_PATH
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    jsonl_path = out_dir / "hidden_results.jsonl"
    results_path = out_dir / "hidden_results.json"
    questions = load_hidden_questions(eval_path)
    jobs = list(iter_jobs(questions))
    if len(jobs) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} jobs, found {len(jobs)}")
    if do_preflight:
        preflight(eval_path=eval_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, done = load_completed(jsonl_path, results_path=results_path)
    if runners is None:
        runners = default_runners()
        if capture_llm is None:
            capture_llm = True
    elif capture_llm is None:
        capture_llm = False

    _safe_print("HIDDEN BENCHMARK")
    _safe_print("=" * 70)
    _safe_print("source:", eval_path)
    _safe_print("n_questions:", len(questions))
    _safe_print("expected_records:", EXPECTED_RECORDS)
    _safe_print("already recorded:", f"{len(done)} / {EXPECTED_RECORDS}")
    _safe_print("output:", out_dir.resolve())
    _safe_print()

    with capture_ollama_stats(enabled=capture_llm):
        for index, (item, pipeline) in enumerate(jobs, start=1):
            key = job_key(item["qid"], pipeline)
            _safe_print(f"{index}/{EXPECTED_RECORDS}")
            if key in done:
                _safe_print(f"  skip {pipeline} {item['qid']} (already recorded)")
                continue
            _safe_print(f"  running {pipeline} {item['qid']}")
            raw = run_one(pipeline, item["question"], runners)
            record = build_record(item, pipeline, raw)
            records.append(record)
            done.add(key)
            _append_jsonl(jsonl_path, record)
            write_artifacts(questions, records, out_dir=out_dir)
            _safe_print(
                f"  answer={record.get('answer')!r} "
                f"error={record.get('error')!r} "
                f"latency={record.get('latency_seconds')}s"
            )

    payload = write_artifacts(questions, records, out_dir=out_dir)
    _safe_print()
    _safe_print("wrote", (out_dir / "hidden_results.json").resolve())
    if payload["summary"]["complete"]:
        _safe_print("HIDDEN BENCHMARK COMPLETE")
    else:
        _safe_print(
            f"HIDDEN BENCHMARK INCOMPLETE: {len(records)}/{EXPECTED_RECORDS}. "
            "Rerun python -m eval.run_hidden to resume."
        )
    return payload


def main():
    return run_hidden(do_preflight=True, capture_llm=True)


if __name__ == "__main__":
    main()
