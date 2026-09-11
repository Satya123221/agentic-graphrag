"""Benchmark RAG, GraphRAG, and Agentic GraphRAG on a small public slice.

Does not modify TigerGraph, RAG, GraphRAG, or the agent orchestrator.
GraphRAG has no built-in generator; answers are produced in this eval
script from its retrieved evidence using the existing corpus-grounded
Ollama helper (same model/settings as RAG, no retrieval changes).
"""

import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.append(".")

from config import PUBLIC_EVAL_PATH
from agent.orchestrator import investigate
from rag.graph_retriever import retrieve_graph
from rag.pipeline import INSUFFICIENT_EVIDENCE, _build_context, _count_tokens, ask_ollama, rag_answer

WANTED_TYPES = ("lookup", "aggregation", "temporal", "multi_hop")
PER_TYPE = 2
PIPELINES = ("rag", "graphrag", "agentic")
RESULTS_DIR = Path("eval") / "results"
RESULTS_PATH = RESULTS_DIR / "benchmark_public.json"


def _safe_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    kwargs.setdefault("flush", True)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(
            (text + "\n").encode(encoding, errors="replace")
        )
        sys.stdout.buffer.flush()


def load_eval():
    rows = []
    with PUBLIC_EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_questions(rows):
    selected = []
    counts = defaultdict(int)
    for row in rows:
        qtype = row.get("qtype")
        if qtype not in WANTED_TYPES:
            continue
        if counts[qtype] >= PER_TYPE:
            continue
        selected.append(row)
        counts[qtype] += 1
        if all(counts[t] >= PER_TYPE for t in WANTED_TYPES):
            break
    return selected


def normalize_answer(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("‘", "'").lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def match_answer(predicted, gold_answers):
    pred = (predicted or "").strip()
    pred_norm = normalize_answer(pred)
    exact = False
    normalized = False
    golds = gold_answers if isinstance(gold_answers, list) else [gold_answers]
    for gold in golds:
        gold_text = (gold if isinstance(gold, str) else str(gold)).strip()
        if pred == gold_text:
            exact = True
            normalized = True
            break
        if pred_norm and pred_norm == normalize_answer(gold_text):
            normalized = True
    return exact, normalized


def gold_doc_hit(gold_doc_ids, retrieved_ids):
    gold = set(gold_doc_ids or [])
    retrieved = set(retrieved_ids or [])
    found = sorted(gold & retrieved)
    return bool(found), found


def _save(payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_rag(question):
    t0 = time.perf_counter()
    raw = rag_answer(question)
    latency = time.perf_counter() - t0
    return {
        "predicted_answer": raw.get("answer") or "",
        "retrieved_document_ids": list(raw.get("document_ids") or []),
        "context_char_count": int(raw.get("context_char_count") or 0),
        "context_token_count": int(raw.get("context_token_count") or 0),
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
    }


def run_graphrag(question):
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
    return {
        "predicted_answer": answer,
        "retrieved_document_ids": list(retrieved.get("document_ids") or []),
        "context_char_count": len(context),
        "context_token_count": _count_tokens(context),
        "agent_iterations": 0,
        "agent_tool_calls": [],
        "latency_sec": round(latency, 2),
    }


def run_agentic(question):
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
    return {
        "predicted_answer": answer,
        "retrieved_document_ids": list(state.get("document_ids") or []),
        "context_char_count": len(blob),
        "context_token_count": _count_tokens(blob),
        "agent_iterations": int(state.get("iteration_count") or 0),
        "agent_tool_calls": tools,
        "latency_sec": round(latency, 2),
        "agent_stop_reason": raw.get("stop_reason"),
        "agent_stopping_status": raw.get("stopping_status"),
        "agent_trace": raw.get("trace") or [],
    }


RUNNERS = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": run_agentic,
}


def summarize(rows):
    by_pipeline = {name: [] for name in PIPELINES}
    by_qtype = {name: defaultdict(list) for name in PIPELINES}
    for row in rows:
        by_pipeline[row["pipeline"]].append(row)
        by_qtype[row["pipeline"]][row["qtype"]].append(row)

    def stats(items):
        n = len(items) or 1
        acc = sum(1 for r in items if r["normalized_match"]) / n
        exact = sum(1 for r in items if r["exact_match"]) / n
        retr = sum(1 for r in items if r["gold_doc_retrieved"]) / n
        latency = sum(r["latency_sec"] for r in items) / n
        tokens = sum(r["context_token_count"] for r in items) / n
        steps = sum(r["agent_iterations"] for r in items) / n
        return {
            "n": len(items),
            "overall_accuracy": round(acc, 4),
            "exact_accuracy": round(exact, 4),
            "retrieval_success": round(retr, 4),
            "avg_latency_sec": round(latency, 2),
            "avg_context_tokens": round(tokens, 1),
            "avg_agent_steps": round(steps, 2),
        }

    overall = {name: stats(items) for name, items in by_pipeline.items()}
    qtype_stats = {}
    for name in PIPELINES:
        qtype_stats[name] = {
            qtype: stats(by_qtype[name][qtype]) for qtype in WANTED_TYPES
        }
    return {"overall": overall, "by_qtype": qtype_stats}


def print_table(summary):
    print()
    print("COMPARISON")
    print("=" * 110)
    header = (
        f"{'Pipeline':<16} {'Overall Accuracy':>18} {'Retrieval Success':>19} "
        f"{'Avg Latency':>12} {'Avg Context Tokens':>20} {'Avg Agent Steps':>16}"
    )
    print(header)
    print("-" * 110)
    for name in PIPELINES:
        row = summary["overall"][name]
        print(
            f"{name:<16} {row['overall_accuracy']:>18.2f} "
            f"{row['retrieval_success']:>19.2f} {row['avg_latency_sec']:>12.2f} "
            f"{row['avg_context_tokens']:>20.1f} {row['avg_agent_steps']:>16.2f}"
        )
    print()
    print("BY QTYPE")
    print("=" * 110)
    for qtype in WANTED_TYPES:
        print(f"\n{qtype}")
        print(
            f"{'Pipeline':<16} {'Accuracy':>10} {'Retrieval':>12} "
            f"{'Latency':>10} {'Tokens':>10} {'Steps':>8}"
        )
        for name in PIPELINES:
            row = summary["by_qtype"][name][qtype]
            print(
                f"{name:<16} {row['overall_accuracy']:>10.2f} "
                f"{row['retrieval_success']:>12.2f} {row['avg_latency_sec']:>10.2f} "
                f"{row['avg_context_tokens']:>10.1f} {row['avg_agent_steps']:>8.2f}"
            )


def main():
    questions = select_questions(load_eval())
    print("PUBLIC SLICE BENCHMARK")
    print("=" * 70)
    print("selected:")
    for row in questions:
        print(f"  {row['qid']}  {row['qtype']}  {row['question'][:90]}")
    print()

    records = []
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            records = list(existing.get("records") or [])
        except Exception:
            records = []
    done = {(row.get("pipeline"), row.get("qid")) for row in records}
    payload = {
        "questions": [
            {"qid": q["qid"], "qtype": q["qtype"], "question": q["question"]}
            for q in questions
        ],
        "records": records,
        "summary": {},
    }
    _save(payload)

    for item in questions:
        gold_docs = list(item.get("gold_doc_ids") or [])
        gold_answer = item.get("answer")
        for pipeline in PIPELINES:
            if (pipeline, item["qid"]) in done:
                print(f"skipping {pipeline} {item['qid']} (already recorded)", flush=True)
                continue
            print(f"running {pipeline} {item['qid']} ...", flush=True)
            try:
                raw = RUNNERS[pipeline](item["question"])
            except Exception as exc:
                raw = {
                    "predicted_answer": "",
                    "retrieved_document_ids": [],
                    "context_char_count": 0,
                    "context_token_count": 0,
                    "agent_iterations": 0,
                    "agent_tool_calls": [],
                    "latency_sec": 0.0,
                    "error": str(exc),
                }
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
                "context_char_count": raw.get("context_char_count") or 0,
                "context_token_count": raw.get("context_token_count") or 0,
                "agent_iterations": raw.get("agent_iterations") or 0,
                "agent_tool_calls": raw.get("agent_tool_calls") or [],
                "latency_sec": raw.get("latency_sec") or 0.0,
            }
            if pipeline == "agentic":
                record["agent_stop_reason"] = raw.get("agent_stop_reason")
                record["agent_stopping_status"] = raw.get("agent_stopping_status")
            if raw.get("error"):
                record["error"] = raw["error"]
            records.append(record)
            payload["summary"] = summarize(records)
            _save(payload)
            _safe_print(
                f"  pred={record['predicted_answer']!r} "
                f"norm_match={normalized} gold_doc={hit} "
                f"latency={record['latency_sec']}s"
            )

    payload["summary"] = summarize(records)
    _save(payload)
    print_table(payload["summary"])
    print()
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
