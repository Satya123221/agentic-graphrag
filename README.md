# Agentic GraphRAG for Olympic Intelligence

An interactive local agent for Olympic questions. It investigates with **TigerGraph**, corpus evidence, and **dynamic tool selection** — and it does not send every question through Agentic GraphRAG.

**Not every question needs Agentic GraphRAG.** A deterministic router chooses the capability: Olympic investigation, general LLM, arithmetic, or an explicit unsupported-current response.

Official frozen public benchmark (100 questions): **RAG 10% · GraphRAG 60% · Agentic 92%**. Those numbers are from `eval/results/final_public_benchmark/` and are not updated by later planner or router work.

---

## 1. Project Overview

This project is a local chatbot that can **investigate** Olympic questions instead of answering from a single retrieval pass.

For an Olympic question it:

1. Builds an investigation state (entities, documents, graph paths, evidence).
2. Chooses **one read-only tool per step** from current gaps.
3. Walks TigerGraph relationships and loads matching corpus text.
4. Returns a final answer only when the evidence supports it, plus a visible investigation trace.

If the graph and corpus cannot support the answer, it returns **Insufficient evidence** rather than inventing a name or count.

The same repository also includes baseline **RAG**, **GraphRAG**, a live three-pipeline compare UI, a frozen public/hidden evaluation, and a read-only metrics dashboard.

---

## 2. Why Agentic GraphRAG?

| Pipeline | How it answers | Typical failure |
|---|---|---|
| **RAG** | TF-IDF over document chunks, then one local Ollama generation call | Similarity can retrieve the wrong page; weak on counts, max, and multi-constraint questions |
| **GraphRAG** | Match entities in TigerGraph, follow relationships, then generate from those documents | Fixed retrieve-then-generate; still one shot after ranking |
| **Agentic GraphRAG** | Inspect gaps, pick a tool, update state, repeat until evidence is sufficient or no useful action remains | Stops with Insufficient evidence when the graph/corpus cannot support the claim |

The agent does **not** follow a fixed search → graph → documents sequence. The next tool depends on the question kind and on what is already in investigation state (for example: resolve a venue, then reverse-walk `HELD_AT`; or collect a full event set before counting).

Official frozen results show where that matters most: aggregation (Agentic **95.2%** vs GraphRAG **4.8%**) and superlative (Agentic **100%** vs GraphRAG **0%**).

---

## 3. Architecture

Full diagrams and explanation:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/architecture.mmd`](docs/architecture.mmd)

**Chatbot (`/api/ask`)**

```
User → Frontend → FastAPI → Question Router → selected capability → answer
```

**Olympic questions**

```
Router
  → Agentic Orchestrator (investigate)
  → Investigation State
  → dynamic tools (one per step)
  → TigerGraph + corpus documents
  → evidence evaluation
  → final answer + investigation trace
```

**Compare (`/api/compare`)** runs RAG, GraphRAG, and Agentic GraphRAG side by side. It does **not** use the chatbot router.

**Benchmark runners** call the three pipelines directly and write frozen artifacts. They do not use the chatbot router. The dashboard only reads those files.

---

## 4. Agent Tools

Implemented in `agent/tools.py`. All are read-only.

| Tool | What it does |
|---|---|
| `search_entities` | Match named vertices mentioned in the question (OlympicGames, Sport, Venue, Country, Athlete, and related attributes). |
| `search_events` | Find Event vertices by name overlap. Can return a truncated ranking or the **complete** matching set (used for aggregation and superlatives). |
| `traverse_graph` | Walk a typed edge from a known vertex (for example Event → WON_BY → Athlete, or reverse Venue ← HELD_AT ← Event). |
| `retrieve_documents` | Load Olympic corpus text by document id, or via `Document --DESCRIBES--> Event`. |
| `lookup_evidence` | Fetch corpus text for one document or event page into evidence. |

`retrieve_documents` / `lookup_evidence` look up `data/corpus.jsonl` by id. They do **not** use the TF-IDF index. TF-IDF is the RAG baseline only (`rag/retriever.py`).

---

## 5. TigerGraph Knowledge Graph

Graph name: `AgenticGraphRAG` (default). REST: `TIGERGRAPH_REST_URL` (default `http://127.0.0.1:14240/restpp`). Schema: `graph/schema.gsql`.

**Vertices**

- Document
- Event
- OlympicGames
- Athlete
- Sport
- Venue
- Country

**Directed edges**

```
Document  --DESCRIBES-->  Event
Event     --PART_OF---->  OlympicGames
Event     --IN_SPORT--->  Sport
Event     --HELD_AT---->  Venue
Event     --WON_BY----->  Athlete
Athlete   --REPRESENTS->  Country
```

The agent also reverse-walks `Venue ← HELD_AT ← Event` for venue + date questions. No other vertex or edge types are defined in this schema.

---

## 6. Question Routing

`backend/router.py` classifies with **deterministic rules**. It does not call Ollama to choose a route.

| Route | When | What runs |
|---|---|---|
| **Agentic GraphRAG** | Olympic / corpus question | Existing `investigate()` planner |
| **General LLM** | Clearly outside the Olympic corpus (for example “What is photosynthesis?”) | Local Ollama `qwen3:8b`, **no** corpus or graph retrieval |
| **Calculation** | Pure arithmetic (for example “What is 25 * 16?”) | Restricted Python AST; result `400` |
| **Current/Web** | Time-sensitive / live questions (for example “What is the weather today?”) | Fixed unsupported message |

**Web search is not implemented.** The Current/Web route exists so the bot does not invent live facts or scrape Olympic pages as if they were current news.

Dated Olympic questions stay on Agentic GraphRAG even when they mention months, years, or “immediately before”. Example: a 2004 / 2012 / 2016 event question is Olympic, not “current web”.

If an Olympic question has no supporting graph/corpus evidence, the route remains Agentic GraphRAG and the answer is **Insufficient evidence** — not a general-LLM guess.

---

## 7. Supported Reasoning Patterns

Verified live Agentic behavior on this graph and corpus (not a claim that later code changed the frozen 92% public score):

| Pattern | Example | Result |
|---|---|---|
| Aggregation | How many shooting events at the 2004 Summer Olympics had more than 37 competitors? | **8** |
| Country-constrained athlete lookup | Who participated from India in shooting? | **Abhinav Bindra** |
| Temporal | Who won the men's 20km walk immediately before the 2016 Summer Olympics? | **Chen Ding** |
| Superlative / max | Which event had the highest number of competitors at the 2004 Summer Olympics? | **Cycling at the 2004 Summer Olympics – Men's individual road race** |
| Venue + date + winner | Which athlete won gold at the event held at Royal Artillery Barracks on 28 July 2012? | **Yi Siling** |
| Insufficient evidence | Venue + date with no matching Event (for example ExCeL on 14 August 2016) | **Insufficient evidence** (no hallucinated winner) |

The agent also handles multi-hop graph walks such as Event → Athlete → Country (`WON_BY` then `REPRESENTS`).

---

## 8. Benchmark

These are the **official frozen public benchmark** results: 100 questions, three pipelines, artifacts under `eval/results/final_public_benchmark/`.

They are **not** recomputed from later planner, router, or demo changes. Do not treat live chatbot traces as a replacement for this table.

### Overall (n = 100)

| | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|
| Accuracy | 10.0% | 60.0% | **92.0%** |
| Retrieval success | 44.0% | 95.0% | 97.0% |
| Completeness | 25.6% | 85.2% | 86.5% |
| Average latency | 250.0s | 389.8s | 18.4s |
| Median latency | 241.7s | 426.7s | 0.42s |

Source fields: `overall_accuracy`, `retrieval_success`, `retrieval_completeness`, `avg_latency_sec`, `median_latency_sec` in `eval/results/final_public_benchmark/benchmark_public_summary.json`.

### Question-type accuracy

| Type | n | RAG | GraphRAG | Agentic |
|---|---:|---:|---:|---:|
| Lookup | 19 | 26.3% | 94.7% | 94.7% |
| Aggregation | 21 | 0% | 4.8% | 95.2% |
| Temporal | 22 | 9.1% | 95.5% | 95.5% |
| Multi-hop | 28 | 10.7% | 71.4% | 82.1% |
| Superlative | 10 | 0% | 0% | 100% |

Dashboard copy of the same headline: RAG 10%, GraphRAG 60%, Agentic 92%.

---

## 9. Hidden Evaluation

Frozen artifacts: `eval/results/hidden_benchmark/`.

- **50** hidden questions
- **150** pipeline records (50 RAG + 50 GraphRAG + 50 Agentic)
- Raw outputs and execution stats captured
- **No hidden accuracy is claimed.** Hidden questions have no gold answers in this repository, and the hidden runner does not score them.

Do not invent or publish hidden gold labels.

---

## 10. Interactive Demo

Requires a running TigerGraph graph `AgenticGraphRAG`, local Ollama with `qwen3:8b`, and the corpus/index files under `data/`.

```text
python -m backend
```

Open [http://127.0.0.1:8010](http://127.0.0.1:8010)

- **Ask** — `/api/ask` goes through the question router. Olympic questions use Agentic GraphRAG.
- **Investigation trace** — each tool step, arguments, and planner reason.
- **Evidence** — corpus passages, matched entities, and graph paths.
- **Compare RAG vs GraphRAG vs Agentic** — `/api/compare` runs all three live pipelines. RAG and GraphRAG generate with local Ollama, so compare can take a long time per question.

The UI shows the selected route (Agentic GraphRAG, General LLM, Calculation, or Current/Web).

---

## 11. Metrics Dashboard

```text
python -m eval.dashboard
```

Open [http://127.0.0.1:8008](http://127.0.0.1:8008)

Read-only FastAPI UI over the official public and hidden artifacts. It does **not** rerun pipelines and does **not** rewrite `eval/results/`.

---

## 12. Local Setup

Inspected from this repository: `requirements.txt`, `.env.example`, and `config.py`. There is **no** Dockerfile or `docker-compose.yml` in the repo.

### Python

```text
python -m venv .venv
```

Activate the venv for your OS, then:

```text
pip install -r requirements.txt
```

`requirements.txt`: `requests`, `python-dotenv`, `numpy`, `pandas`, `scikit-learn`, `pytest`, `fastapi`, `uvicorn`.

### Environment

Copy `.env.example` to `.env`. The example sets:

```text
OLLAMA_MODEL=qwen3:8b
```

`config.py` defaults (override with environment variables as needed):

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` |
| `RAG_TOP_K` | `5` |
| `TIGERGRAPH_REST_URL` | `http://127.0.0.1:14240/restpp` |
| `TIGERGRAPH_GRAPH` | `AgenticGraphRAG` |

### Ollama

Install [Ollama](https://ollama.com) locally and make model `qwen3:8b` available at `OLLAMA_BASE_URL`. Generation uses `temperature=0` and `seed=0`.

### TigerGraph Community Edition

The live agent and GraphRAG talk to TigerGraph 4.2 REST (`rag/graph_client.py`). Community Edition must already be running with graph `AgenticGraphRAG`.

This repository does not vendor a Docker Compose stack. Community Edition is commonly run via TigerGraph’s own Docker distribution; use that product’s install process, then apply:

- `graph/schema.gsql`
- load scripts under `graph/` (`load_documents.gsql`, `load_won_by.gsql`, `load_venues.gsql`, `load_held_at.gsql`, `load_countries.gsql`, `load_represents.gsql`, …)

Graph JSONL/TSV is produced by `ingestion/` and `graph/generate_data.py` from `data/corpus.jsonl`. Those are **offline** prep steps, not per-question runtime.

### Data expected at runtime

- `data/corpus.jsonl`
- `data/chunks.jsonl`
- `data/tfidf_index.pkl` (RAG)
- `data/eval_public.jsonl` / `data/eval_hidden.jsonl` (evaluation runners)

Do not rerun the official 100-question public benchmark to “refresh” scores. The submission scores are the frozen files in `eval/results/`.

---

## 13. Safety / Evidence Behavior

- Olympic answers must come from TigerGraph + corpus evidence.
- If that evidence is missing or does not match the question (wrong date, no `HELD_AT` event, no country match), the system returns **Insufficient evidence** instead of inventing an athlete, count, or event name.
- General LLM is used only for non-Olympic questions and does not retrieve Olympic documents.
- **Current/web** questions receive an explicit unsupported-current response because **web search is not implemented**.
- Compare-pipeline crashes (timeouts, connection errors) are shown as **Pipeline error**, not as Insufficient evidence.

---

## 14. Limitations and Future Work

**Limitations (honest)**

- Web search / live news is **not** implemented.
- The corpus and graph are Olympic-focused.
- Local Ollama `qwen3:8b` is slow, especially RAG/GraphRAG compare.
- Some venue + date wordings have **no matching Event** in the graph (legitimate Insufficient evidence, not a license to guess).
- Official public scores are a frozen 100-question run; they are not live-recomputed after later routing or planner work.

**Possible future work**

- Web search integration for the Current/Web route
- Richer uncertainty and conflict handling
- Source authority and supersession
- Broader corpus beyond Olympic event pages

---

## 15. Project Structure

```text
backend/           FastAPI app, router, ask/compare adapters
frontend/          Chat UI (Ask + Compare)
agent/             Orchestrator, investigation state, tools
rag/               RAG (TF-IDF) and GraphRAG retriever
graph/             schema.gsql and TigerGraph load scripts
ingestion/         Offline corpus → graph/index extraction
docs/              Architecture markdown + Mermaid
eval/              Public/hidden runners and read-only dashboard
eval/results/      Official frozen public and hidden artifacts
config.py          Paths and environment defaults
requirements.txt   Python dependencies
.env.example       Example OLLAMA_MODEL
```

Important entry points:

| Path | Role |
|---|---|
| `backend/app.py` | `/api/ask`, `/api/compare`, serves `frontend/index.html` |
| `backend/router.py` | Deterministic question router |
| `agent/orchestrator.py` | Agentic planner (`investigate`) |
| `agent/state.py` | Investigation memory |
| `agent/tools.py` | Five read-only tools |
| `rag/pipeline.py` | RAG generate |
| `rag/graph_retriever.py` | GraphRAG retrieve |
| `eval/run_public_final.py` | Official public runner (frozen outputs) |
| `eval/run_hidden.py` | Hidden runner (raw outputs, no accuracy) |
| `eval/dashboard/app.py` | Metrics dashboard |
| `graph/schema.gsql` | Vertex and edge types |

---

## 16. Hackathon Deliverables

- Working interactive agent (`python -m backend` → http://127.0.0.1:8010)
- RAG / GraphRAG / Agentic implementations and live compare
- Public evaluation (frozen 100-question official results)
- Hidden evaluation (50 questions, 150 records, no hidden accuracy)
- Metrics dashboard (`python -m eval.dashboard` → http://127.0.0.1:8008)
- Architecture documentation (`docs/architecture.md`, `docs/architecture.mmd`)
- This GitHub repository

---

TigerGraph Agentic GraphRAG Hackathon. Local only: TigerGraph CE, Ollama `qwen3:8b`, no external web search.
