"""Baseline RAG: TF-IDF retrieval plus a single Ollama generation call."""

import re
import sys

sys.path.append(".")

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, RAG_TOP_K
from rag.retriever import retrieve

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the retrieved context."
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CONTEXT_CHARS_PER_CHUNK = 1200

SYSTEM_PROMPT = """You are a corpus-grounded question answering system.
Answer using ONLY the retrieved context below.
Do not use outside knowledge.
Do not guess or invent names, numbers, dates, or events.
If the context does not contain enough information to answer, reply exactly:
Insufficient evidence in the retrieved context.
Give a concise factual answer with no preamble.
/no_think"""


def _build_context(results):
    parts = []
    for result in results:
        text = result["text"]
        if len(text) > CONTEXT_CHARS_PER_CHUNK:
            text = text[:CONTEXT_CHARS_PER_CHUNK].rsplit(" ", 1)[0]
        parts.append(
            f"[Document: {result['doc_id']} | {result['title']}]\n{text}"
        )
    return "\n\n".join(parts)


def _count_tokens(text):
    return len(text.split())


def _strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def ask_ollama(question, context):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": 256,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600,
    )
    if response.status_code == 400:
        payload.pop("think", None)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=600,
        )
    response.raise_for_status()
    body = response.json()
    return _strip_think(body.get("response", ""))


def rag_answer(question, top_k=None):
    if top_k is None:
        top_k = RAG_TOP_K
    results = retrieve(question, top_k=top_k)
    context = _build_context(results)
    document_ids = list(dict.fromkeys(r["doc_id"] for r in results))
    similarity_scores = [r["score"] for r in results]
    answer = (
        ask_ollama(question, context) if results else INSUFFICIENT_EVIDENCE
    )
    if not answer:
        answer = INSUFFICIENT_EVIDENCE
    return {
        "answer": answer,
        "retrieved_chunks": results,
        "document_ids": document_ids,
        "similarity_scores": similarity_scores,
        "context_char_count": len(context),
        "context_token_count": _count_tokens(context),
    }


if __name__ == "__main__":
    question = input("Enter your question: ")
    result = rag_answer(question)
    print("\nRAG ANSWER")
    print("=" * 70)
    print(result["answer"])
    print("\nSOURCES")
    print("=" * 70)
    for chunk, score in zip(result["retrieved_chunks"], result["similarity_scores"]):
        print(
            f"- {chunk['doc_id']} | {chunk['title']} | "
            f"score={score:.4f} | {chunk['chunk_id']}"
        )
    print(
        f"\nContext chars={result['context_char_count']} "
        f"tokens={result['context_token_count']}"
    )
