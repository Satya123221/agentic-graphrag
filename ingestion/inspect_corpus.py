import json
import sys; sys.path.append("."); from config import CORPUS_PATH

count = 0
total_tokens = 0

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            doc = json.loads(line)
            count += 1
            total_tokens += doc.get("approx_tokens", 0)

print("CORPUS INSPECTION COMPLETE")
print("Documents:", count)
print("Approx tokens:", total_tokens)
