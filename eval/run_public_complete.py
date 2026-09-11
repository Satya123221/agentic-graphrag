"""Complete 100-question public benchmark.

Uses the existing scoring and pipeline runners from benchmark_public.py.
Does not modify RAG, GraphRAG, TigerGraph, the agent, or eval_public.jsonl.
Does not use the 8-question slice.
"""

import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(".")

from rag.pipeline import INSUFFICIENT_EVIDENCE
from eval.benchmark_public import (
    PIPELINES,
    RUNNERS,
    RESULTS_DIR,
    _safe_print,
    gold_doc_hit,
    load_eval,
    match_answer,
)

REPORT_TYPES = ("lookup", "aggregation", "temporal", "multi_hop")
ALL_TYPES = ("lookup", "aggregation", "temporal", "multi_hop", "superlative")
RESULTS_PATH = RESULTS_DIR / "benchmark_public_complete.json"
LOG_PATH = RESULTS_DIR / "benchmark_public_complete.log"
QUESTION_TIMEOUT_NOTE = (
    "Pipeline exceptions including Ollama/HTTP timeouts are recorded as failures."
)


def load_all_questions():
    rows = load_eval()
    if len(rows) != 100:
        raise SystemExit(
            f"Expected 100 questions in data/eval_public.jsonl, found {len(rows)}"
        )
    qids = [row["qid"] for row in rows]
    if len(set(qids)) != 100:
        raise SystemExit("eval_public.jsonl does not contain 100 unique qids")
    return rows


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _retrieval_recall(gold_ids, retrieved_ids):
    gold = set(gold_ids or [])
    if not gold:
        return 0.0
    return len(gold & set(retrieved_ids or [])) / len(gold)


def classify_agentic_failure(record):
    error = (record.get("error") or "").lower()
    if error:
        if "timed out" in error or "timeout" in error:
            return "timeout"
        return "exception"
    if record.get("normalized_match"):
        return "success"
    predicted = (record.get("predicted_answer") or "").strip()
    stop = record.get("agent_stop_reason") or ""
    if predicted == INSUFFICIENT_EVIDENCE or not predicted:
        if stop == "maximum iteration limit":
            return "iteration_limit_no_answer"
        return "insufficient_evidence"
    gold = set(record.get("gold_doc_ids") or [])
    got = set(record.get("retrieved_document_ids") or [])
    if not record.get("gold_doc_retrieved"):
        return "wrong_answer_retrieval_miss"
    if gold and not gold <= got:
        if stop == "maximum iteration limit":
            return "iteration_limit_incomplete_retrieval"
        return "wrong_answer_incomplete_retrieval"
    if stop == "maximum iteration limit":
        return "iteration_limit_wrong_answer"
    return "wrong_answer_with_gold_docs"


def stats(items):
    n = len(items)
    if n == 0:
        return {
            "n": 0,
            "overall_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "retrieval_success": 0.0,
            "retrieval_completeness": 0.0,
            "avg_latency_sec": 0.0,
            "avg_context_tokens": 0.0,
            "token_efficiency_per_1k_tokens": 0.0,
            "avg_agent_steps": 0.0,
            "avg_tool_calls": 0.0,
            "n_failures": 0,
            "n_timeouts": 0,
            "n_errors": 0,
        }
    acc = sum(1 for r in items if r["normalized_match"]) / n
    exact = sum(1 for r in items if r["exact_match"]) / n
    retr = sum(1 for r in items if r["gold_doc_retrieved"]) / n
    completeness = _mean([
        _retrieval_recall(r.get("gold_doc_ids"), r.get("retrieved_document_ids"))
        for r in items
    ])
    latency = _mean([r["latency_sec"] for r in items])
    tokens = _mean([r["context_token_count"] for r in items])
    steps = _mean([r.get("agent_iterations") or 0 for r in items])
    tools = _mean([len(r.get("agent_tool_calls") or []) for r in items])
    n_fail = sum(1 for r in items if not r["normalized_match"])
    n_timeout = sum(
        1 for r in items
        if "timeout" in (r.get("error") or "").lower()
        or "timed out" in (r.get("error") or "").lower()
    )
    n_err = sum(1 for r in items if r.get("error"))
    efficiency = (acc / tokens * 1000.0) if tokens else 0.0
    return {
        "n": n,
        "overall_accuracy": round(acc, 4),
        "exact_accuracy": round(exact, 4),
        "retrieval_success": round(retr, 4),
        "retrieval_completeness": round(completeness, 4),
        "avg_latency_sec": round(latency, 2),
        "avg_context_tokens": round(tokens, 1),
        "token_efficiency_per_1k_tokens": round(efficiency, 4),
        "avg_agent_steps": round(steps, 2),
        "avg_tool_calls": round(tools, 2),
        "n_failures": n_fail,
        "n_timeouts": n_timeout,
        "n_errors": n_err,
    }


def summarize(rows):
    by_pipeline = {name: [] for name in PIPELINES}
    by_qtype = {name: defaultdict(list) for name in PIPELINES}
    for row in rows:
        by_pipeline[row["pipeline"]].append(row)
        by_qtype[row["pipeline"]][row["qtype"]].append(row)
    overall = {name: stats(items) for name, items in by_pipeline.items()}
    qtype_stats = {}
    for name in PIPELINES:
        qtype_stats[name] = {
            qtype: stats(by_qtype[name][qtype]) for qtype in ALL_TYPES
        }
    agentic_rows = by_pipeline["agentic"]
    categories = Counter(
        classify_agentic_failure(row) for row in agentic_rows
    )
    failures = [row for row in agentic_rows if not row.get("normalized_match")]
    traces = []
    for row in failures:
        traces.append({
            "qid": row["qid"],
            "qtype": row["qtype"],
            "question": row["question"],
            "gold_answer": row.get("gold_answer"),
            "predicted_answer": row.get("predicted_answer"),
            "failure_category": classify_agentic_failure(row),
            "gold_doc_retrieved": row.get("gold_doc_retrieved"),
            "n_gold_docs": len(row.get("gold_doc_ids") or []),
            "n_gold_docs_found": len(row.get("gold_docs_found") or []),
            "n_retrieved": len(row.get("retrieved_document_ids") or []),
            "agent_iterations": row.get("agent_iterations"),
            "agent_tool_calls": row.get("agent_tool_calls"),
            "agent_stop_reason": row.get("agent_stop_reason"),
            "agent_stopping_status": row.get("agent_stopping_status"),
            "error": row.get("error"),
            "agent_trace": row.get("agent_trace") or [],
        })
    pipeline_failures = {}
    for name in PIPELINES:
        pipeline_failures[name] = [
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "predicted_answer": row.get("predicted_answer"),
                "gold_answer": row.get("gold_answer"),
                "gold_doc_retrieved": row.get("gold_doc_retrieved"),
                "error": row.get("error"),
            }
            for row in by_pipeline[name]
            if not row.get("normalized_match")
        ]
    return {
        "overall": overall,
        "by_qtype": qtype_stats,
        "agentic_failure_categories": dict(categories),
        "agentic_failure_traces": traces,
        "failures": pipeline_failures,
    }


def print_table(summary):
    print()
    print("COMPLETE PUBLIC BENCHMARK COMPARISON")
    print("=" * 130)
    header = (
        f"{'Pipeline':<16} {'Accuracy':>10} {'Retrieval':>12} {'Completeness':>14} "
        f"{'Latency':>10} {'Tokens':>10} {'TokEff/1k':>10} {'Steps':>8} {'Tools':>8} "
        f"{'Fail':>6} {'TO':>4}"
    )
    print(header)
    print("-" * 130)
    for name in PIPELINES:
        row = summary["overall"][name]
        print(
            f"{name:<16} {row['overall_accuracy']:>10.4f} "
            f"{row['retrieval_success']:>12.4f} {row['retrieval_completeness']:>14.4f} "
            f"{row['avg_latency_sec']:>10.2f} {row['avg_context_tokens']:>10.1f} "
            f"{row['token_efficiency_per_1k_tokens']:>10.4f} "
            f"{row['avg_agent_steps']:>8.2f} {row['avg_tool_calls']:>8.2f} "
            f"{row['n_failures']:>6} {row['n_timeouts']:>4}"
        )
    print()
    print("ACCURACY BY QTYPE")
    print("=" * 110)
    for qtype in ALL_TYPES:
        print(f"\n{qtype}")
        print(
            f"{'Pipeline':<16} {'Accuracy':>10} {'Retrieval':>12} "
            f"{'Completeness':>14} {'Latency':>10} {'Tokens':>10} {'Steps':>8}"
        )
        for name in PIPELINES:
            row = summary["by_qtype"][name][qtype]
            print(
                f"{name:<16} {row['overall_accuracy']:>10.4f} "
                f"{row['retrieval_success']:>12.4f} {row['retrieval_completeness']:>14.4f} "
                f"{row['avg_latency_sec']:>10.2f} {row['avg_context_tokens']:>10.1f} "
                f"{row['avg_agent_steps']:>8.2f}"
            )
    print()
    print("AGENTIC FAILURE CATEGORIES")
    print("=" * 70)
    cats = summary.get("agentic_failure_categories") or {}
    for name, count in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name}: {count}")


def _save(payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _empty_raw(error, latency=0.0):
    return {
        "predicted_answer": "",
        "retrieved_document_ids": [],
        "context_char_count": 0,
        "context_token_count": 0,
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
        "error": error,
    }


def run_one(pipeline, question):
    t0 = time.perf_counter()
    try:
        raw = RUNNERS[pipeline](question)
        return raw
    except Exception as exc:
        latency = time.perf_counter() - t0
        err = str(exc)
        if "timed out" in err.lower() or "timeout" in err.lower():
            err = f"timeout: {err}"
        tb = traceback.format_exc(limit=4)
        return _empty_raw(f"{err}\n{tb}", latency=latency)


def main():
    questions = load_all_questions()
    counts = Counter(row.get("qtype") for row in questions)
    print("COMPLETE PUBLIC BENCHMARK")
    print("=" * 70)
    print("source: data/eval_public.jsonl")
    print("n_questions:", len(questions))
    print("unique_qids:", len({row['qid'] for row in questions}))
    print("qtype_counts:", dict(counts))
    print("slice: DISABLED (running all 100 questions)")
    print("pipelines:", ", ".join(PIPELINES))
    print(QUESTION_TIMEOUT_NOTE)
    if len(questions) != 100:
        raise SystemExit("refusing to start: not 100 questions")
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
        "n_questions": 100,
        "slice": None,
        "questions": [
            {"qid": q["qid"], "qtype": q["qtype"], "question": q["question"]}
            for q in questions
        ],
        "records": records,
        "summary": {},
    }
    _save(payload)
    print(f"already recorded: {len(done)} / {100 * len(PIPELINES)} pipeline-question pairs")

    total = len(questions) * len(PIPELINES)
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
                "agent_iterations": raw.get("agent_iterations") or 0,
                "agent_tool_calls": raw.get("agent_tool_calls") or [],
                "latency_sec": raw.get("latency_sec") or 0.0,
            }
            if pipeline == "agentic":
                record["agent_stop_reason"] = raw.get("agent_stop_reason")
                record["agent_stopping_status"] = raw.get("agent_stopping_status")
                record["failure_category"] = classify_agentic_failure(record)
                if not normalized:
                    record["agent_trace"] = raw.get("agent_trace") or []
            if raw.get("error"):
                record["error"] = raw["error"]
                if pipeline == "agentic":
                    record["failure_category"] = classify_agentic_failure(record)
            records.append(record)
            finished += 1
            payload["records"] = records
            payload["summary"] = summarize(records)
            _save(payload)
            _safe_print(
                f"  pred={record['predicted_answer']!r} "
                f"norm_match={normalized} gold_doc={hit} "
                f"latency={record['latency_sec']}s "
                f"progress={finished}/{total}"
            )

    payload["records"] = records
    payload["summary"] = summarize(records)
    _save(payload)
    print_table(payload["summary"])
    print()
    print("wrote", RESULTS_PATH.resolve())
    print("COMPLETE PUBLIC BENCHMARK")
    expected = 100 * len(PIPELINES)
    if len(records) != expected:
        raise SystemExit(
            f"incomplete run: {len(records)} records, expected {expected}"
        )


if __name__ == "__main__":
    main()
