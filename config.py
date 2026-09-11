from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
TFIDF_INDEX_PATH = DATA_DIR / "tfidf_index.pkl"
PUBLIC_EVAL_PATH = DATA_DIR / "eval_public.jsonl"
HIDDEN_EVAL_PATH = DATA_DIR / "eval_hidden.jsonl"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
TIGERGRAPH_REST_URL = os.getenv("TIGERGRAPH_REST_URL", "http://127.0.0.1:14240/restpp")
TIGERGRAPH_GRAPH = os.getenv("TIGERGRAPH_GRAPH", "AgenticGraphRAG")
