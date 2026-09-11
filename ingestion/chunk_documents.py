import json
import sys; sys.path.append("."); from config import CORPUS_PATH, DATA_DIR

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

with open(CORPUS_PATH, "r", encoding="utf-8") as infile, \
     open(CHUNKS_PATH, "w", encoding="utf-8") as outfile:

    chunk_count = 0

    for line in infile:
        if not line.strip():
            continue

        doc = json.loads(line)
        words = doc["text"].split()

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunk = {
                "chunk_id": f"{doc['doc_id']}_{chunk_count}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": doc["url"],
                "text": " ".join(words[start:end])
            }

            outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP

print("CHUNKING COMPLETE")
print("Chunks:", chunk_count)
print("Output:", CHUNKS_PATH)
