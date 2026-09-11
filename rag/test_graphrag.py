"""Run GraphRAG retrieval on the same public question as the RAG baseline."""

import json
import sys

sys.path.append(".")

from config import PUBLIC_EVAL_PATH
from rag.graph_retriever import retrieve_graph

TEST_QID = "pub-002"


def load_public_question(qid):
    with PUBLIC_EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("qid") == qid:
                return row
    raise SystemExit(f"Question {qid} not found in {PUBLIC_EVAL_PATH}")


def main():
    item = load_public_question(TEST_QID)
    print(f"Running GraphRAG retrieval for {item['qid']} ...", flush=True)
    result = retrieve_graph(item["question"])

    print("GRAPHRAG RETRIEVAL TEST")
    print("=" * 70)
    print("qid:", item["qid"])
    print("question:", item["question"])
    print("gold:", item.get("answer"))
    print("gold_doc_ids:", item.get("gold_doc_ids"))
    print()
    print("matched_entities.games:", result["matched_entities"]["games"])
    print("matched_entities.sports:", result["matched_entities"]["sports"])
    print("matched_entities.temporal:", result["matched_entities"]["temporal"])
    print("matched_entities.gender:", result["matched_entities"]["gender"])
    print("matched_events:")
    for event in result["matched_entities"]["events"]:
        print(
            f"  {event['id']} | {event['name']} | "
            f"games={event['games_id']} sport={event['sport_id']} "
            f"score={event['score']}"
        )
    print("matched_athletes:")
    for athlete in result["matched_entities"]["athletes"]:
        print(f"  {athlete['id']} | {athlete['name']} | event={athlete['event_id']}")
    print()
    print("document_ids:", result["document_ids"])
    gold = set(item.get("gold_doc_ids") or [])
    print("gold_docs_found:", sorted(gold & set(result["document_ids"])))
    print("stats:", result["stats"])
    print()
    print("graph_paths (first 20):")
    for row in result["graph_paths"][:20]:
        print(f"  {row['path']} | {row.get('from_id')} -> {row.get('to_id')}")
    print()
    print("evidence:")
    for item_ev in result["evidence"]:
        snippet = item_ev["text"][:240].replace("\n", " ")
        print(f"  {item_ev['doc_id']} | {item_ev['title']} | score={item_ev['score']}")
        print(f"    {snippet}")


if __name__ == "__main__":
    main()
