"""GraphRAG retrieval validation on representative public eval questions.

Does not modify TigerGraph or the TF-IDF RAG baseline.
Success = at least one gold_doc_id appears in retrieved document IDs.
"""

import json
import sys
import time
from collections import defaultdict

sys.path.append(".")

from config import PUBLIC_EVAL_PATH
from rag.graph_retriever import retrieve_graph

WANTED_TYPES = ("lookup", "aggregation", "temporal", "multi_hop")
PER_TYPE = 2
EVIDENCE_SNIPPET = 220


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


def snippet(text):
    text = (text or "").replace("\n", " ")
    return text[:EVIDENCE_SNIPPET]


def main():
    questions = select_questions(load_eval())
    print("GRAPHRAG PUBLIC RETRIEVAL VALIDATION")
    print("=" * 70)
    print("selected:")
    for row in questions:
        print(f"  {row['qid']}  {row['qtype']}  {row['question'][:90]}")
    print()

    results = []
    for row in questions:
        gold_docs = list(row.get("gold_doc_ids") or [])
        t0 = time.perf_counter()
        retrieved = retrieve_graph(row["question"])
        latency = time.perf_counter() - t0
        found = sorted(set(gold_docs) & set(retrieved["document_ids"]))
        success = len(found) > 0
        paths = []
        seen = set()
        for item in retrieved.get("graph_paths") or []:
            name = item.get("path")
            if name and name not in seen:
                seen.add(name)
                paths.append(name)
        record = {
            "qid": row["qid"],
            "qtype": row["qtype"],
            "question": row["question"],
            "gold_answer": row.get("answer"),
            "gold_doc_ids": gold_docs,
            "retrieved_document_ids": retrieved["document_ids"],
            "gold_docs_found": found,
            "gold_hit": success,
            "latency_sec": round(latency, 2),
            "graph_paths": paths,
            "matched_events": [
                f"{e['id']} | {e['name']}"
                for e in retrieved["matched_entities"].get("events") or []
            ],
            "matched_venues": retrieved["matched_entities"].get("venues") or [],
            "evidence": [
                {
                    "doc_id": ev["doc_id"],
                    "title": ev["title"],
                    "snippet": snippet(ev["text"]),
                }
                for ev in (retrieved.get("evidence") or [])[:4]
            ],
            "stats": retrieved.get("stats") or {},
        }
        results.append(record)
        print("-" * 70)
        print("qid:", record["qid"])
        print("qtype:", record["qtype"])
        print("gold_answer:", record["gold_answer"])
        print("retrieved_document_ids:", record["retrieved_document_ids"])
        print("gold_doc_retrieved:", record["gold_hit"], found)
        print("retrieval:", "SUCCESS" if success else "FAILURE")
        print("graph_paths:", paths)
        print("latency_sec:", record["latency_sec"])
        print("matched_venues:", record["matched_venues"])
        print("matched_events:")
        for line in record["matched_events"][:6]:
            print("  ", line)
        print("retrieved_evidence:")
        for ev in record["evidence"]:
            print(f"  {ev['doc_id']} | {ev['title']}")
            print(f"    {ev['snippet']}")
        print(flush=True)

    by_type = defaultdict(list)
    for rec in results:
        by_type[rec["qtype"]].append(rec["gold_hit"])

    overall = sum(1 for r in results if r["gold_hit"])
    print("=" * 70)
    print("SUMMARY")
    print(f"overall_success_rate: {overall}/{len(results)} = {overall / len(results):.2f}")
    for qtype in WANTED_TYPES:
        hits = by_type[qtype]
        n = len(hits)
        s = sum(hits)
        print(f"  {qtype}: {s}/{n} = {(s / n if n else 0):.2f}")
    failures = [r for r in results if not r["gold_hit"]]
    print("failures:", [r["qid"] for r in failures] or "none")


if __name__ == "__main__":
    main()
