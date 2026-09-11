"""Run one public evaluation question through the baseline RAG pipeline."""

import json
import sys

sys.path.append(".")

from config import PUBLIC_EVAL_PATH
from rag.pipeline import rag_answer

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
    print(f"Running {item['qid']} ...", flush=True)
    print("Retrieving chunks ...", flush=True)
    result = rag_answer(item["question"])

    print("BASELINE RAG TEST")
    print("=" * 70)
    print("qid:", item["qid"])
    print("qtype:", item.get("qtype"))
    print("question:", item["question"])
    print("gold:", item.get("answer"))
    print("gold_doc_ids:", item.get("gold_doc_ids"))
    print()
    print("answer:", result["answer"])
    print("document_ids:", result["document_ids"])
    print("similarity_scores:", [round(s, 4) for s in result["similarity_scores"]])
    print("context_char_count:", result["context_char_count"])
    print("context_token_count:", result["context_token_count"])
    print()
    print("retrieved_chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(
            f"  {i}. {chunk['chunk_id']} | {chunk['doc_id']} | "
            f"{chunk['title']} | score={chunk['score']:.4f}"
        )


if __name__ == "__main__":
    main()
