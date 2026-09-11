import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import sys; sys.path.append("."); from config import DATA_DIR

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "tfidf_index.pkl"

chunks = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

matrix = vectorizer.fit_transform(texts)

with open(INDEX_PATH, "wb") as f:
    pickle.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": chunks
        },
        f
    )

print("TF-IDF INDEX COMPLETE")
print("Chunks indexed:", len(chunks))
print("Features:", len(vectorizer.get_feature_names_out()))
print("Output:", INDEX_PATH)
