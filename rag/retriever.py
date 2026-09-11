"""TF-IDF chunk retriever over the existing corpus index."""

import sys

sys.path.append(".")

import pickle

from sklearn.metrics.pairwise import cosine_similarity

from config import TFIDF_INDEX_PATH

_index = None


def load_index():
    global _index
    if _index is None:
        with TFIDF_INDEX_PATH.open("rb") as f:
            _index = pickle.load(f)
    return _index


def retrieve(query, top_k=5):
    """Return the top-k chunks for query, highest cosine score first.

    Ties are broken by chunk_id so ranking is deterministic.
    """
    index = load_index()
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]),
    )[:top_k]

    results = []
    for i in ranked:
        chunk = chunks[i]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "text": chunk["text"],
            "score": float(scores[i]),
        })
    return results


if __name__ == "__main__":
    question = input("Enter your question: ")
    results = retrieve(question)
    print("\nRETRIEVAL RESULTS")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Doc ID: {result['doc_id']}")
        print(result["text"][:300])
