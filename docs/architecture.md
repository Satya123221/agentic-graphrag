# TigerGraph Agentic GraphRAG — System Architecture

This document describes the **implemented** architecture of the local hackathon system. It was written from the repository as it exists today: chatbot frontend, FastAPI backend, deterministic question router, Agentic GraphRAG orchestrator, TigerGraph graph, corpus retrieval, local Ollama, and a separate frozen-benchmark dashboard.

It does **not** invent services that are not in the code.

## 1. Two product surfaces

| Surface | Entry | Port | Purpose |
|---|---|---|---|
| Live chatbot | `frontend/index.html` via `backend.app` | `127.0.0.1:8010` | Ask a question (`/api/ask`) or compare RAG vs GraphRAG vs Agentic (`/api/compare`) |
| Evaluation dashboard | `eval/dashboard` | `127.0.0.1:8008` | Read-only view of **frozen** public and hidden benchmark artifacts. Does not rerun pipelines. |

The chatbot and the dashboard do not share a request path. Benchmark scores shown on the dashboard come from `eval/results/`, not from a live `/api/ask` call.

## 2. Live chatbot data flow

```
User
  → Chatbot / Frontend
  → FastAPI Backend
  → Question Router
       ├── Calculation
       ├── General LLM  (local Ollama qwen3:8b, no corpus)
       ├── Current/Web  (implemented unsupported message; no web client)
       └── Agentic GraphRAG
              → Agent Orchestrator
              → Investigation State
              → Dynamic tool selection (one tool per step)
              → TigerGraph + Olympic corpus
              → Evidence evaluation
              → Final answer + investigation trace
```

`/api/ask` is `backend.app.api_ask` → `backend.service.route_and_answer`.  
The router (`backend/router.py`) classifies with **deterministic rules**. It does not call Ollama to choose a route.

`/api/compare` does **not** use the router. It runs the three Olympic pipelines side by side: RAG, GraphRAG, and Agentic GraphRAG (`backend.service.compare_pipelines`).

## 3. Question router

Implemented routes and labels:

| `route` | `route_label` | What runs | Retrieval |
|---|---|---|---|
| `calculation` | Calculation | Restricted Python AST arithmetic | None |
| `general_llm` | General LLM | `ask_general_llm()` → local Ollama `qwen3:8b` | None (must not pull Olympic documents) |
| `current_web` | Current/Web | Fixed unsupported message | None. **Web search is not implemented.** |
| `agentic_graphrag` | Agentic GraphRAG | Existing `investigate()` planner | TigerGraph + corpus by document/event id |

Priority used by `classify()`:

1. Pure arithmetic expression → Calculation  
2. Time-sensitive wording, unless it is a dated Olympic question → Current/Web  
3. Olympic / corpus signals → Agentic GraphRAG  
4. Otherwise → General LLM  

If Agentic GraphRAG cannot support the question from graph + corpus evidence, the answer is **Insufficient evidence**. The system does not fall back to general knowledge to invent a winner, count, or event name.

## 4. Agentic GraphRAG investigation

This is the core Olympic pipeline. The planner was **not** rewritten for the router; the router only decides whether to call it.

### 4.1 Orchestrator loop (`agent/orchestrator.py` → `investigate()`)

Each question starts an `InvestigationState` (`agent/state.py`) with empty entities, evidence, documents, and graph paths.

Each iteration:

1. Stop if evidence is already sufficient, or if the iteration cap is reached.  
2. `inspect_gaps()` — what is still missing for this question kind.  
3. `propose_actions()` — candidate tools from **current gaps only**, not a fixed search → graph → documents sequence.  
4. `choose_action()` — pick **one** read-only tool.  
5. `call_tool()` — execute that tool.  
6. Merge entities, events, documents, evidence, and graph paths into state.  
7. Repeat.

Question kinds already classified inside the orchestrator (separate from the chatbot router): `who_won`, `lookup`, `aggregation`, `superlative`, `temporal`, `venue_date`, `multi_hop`.

### 4.2 Dynamic tool selection

The five implemented tools (`agent/tools.py`) are:

| Tool | Role |
|---|---|
| `search_entities` | Match named vertices (games, sports, venues, countries, athletes) from the question |
| `search_events` | Find Event vertices (including complete sport/year sets for aggregation and superlatives) |
| `traverse_graph` | Walk a typed edge, including reverse `Venue ← HELD_AT ← Event` |
| `retrieve_documents` | Load Olympic **corpus** text by document id or `Document → DESCRIBES → Event` |
| `lookup_evidence` | Fetch a specific document’s corpus text into evidence |

Which tool runs next depends on state. Examples that exist in the planner today:

- Venue + date: identify Venue → reverse `HELD_AT` → date-filter Events → retrieve evidence → `WON_BY` / gold infobox.  
- Country-constrained participant: Event `WON_BY` Athlete, then Athlete `REPRESENTS` Country.  
- Aggregation / superlative: discover the **full** scoped Event set, retrieve leftover events, then count or take max/min.  
- Temporal: resolve previous/next OlympicGames before selecting the Event.

This is **not** a fixed RAG pipeline. A who-won lookup may stop after two tools; a shooting aggregation may retrieve many events; a venue question may never call `search_events`.

### 4.3 Planner policy

`choose_action()` is deterministic when the top candidate is unique. Local Ollama is used only as a **planner fallback** when several tools look equally useful. Agentic **answers** are extracted from collected evidence (`extract_answer`), not generated from unconstrained model memory.

If gaps remain and no useful tool is left, the orchestrator stops. An empty extraction becomes **Insufficient evidence** in the API (`backend.service.format_investigation`).

## 5. Graph and corpus

### 5.1 TigerGraph

- REST: `TIGERGRAPH_REST_URL` (default `http://127.0.0.1:14240/restpp`)  
- Graph name: `AgenticGraphRAG`  
- Access: read-only helpers in `rag/graph_client.py`

Vertex types from `graph/schema.gsql`:

- `Document`, `Event`, `OlympicGames`, `Athlete`, `Sport`, `Venue`, `Country`

Directed edges:

```
Document  --DESCRIBES--> Event
Event     --PART_OF----> OlympicGames
Event     --IN_SPORT---> Sport
Event     --HELD_AT----> Venue
Event     --WON_BY-----> Athlete
Athlete   --REPRESENTS-> Country
```

Reverse walk used by the agent: `Venue ← HELD_AT ← Event`.

### 5.2 Corpus retrieval (Agentic / GraphRAG)

Agentic `retrieve_documents` / `lookup_evidence` and GraphRAG `retrieve_graph` load `data/corpus.jsonl` **by document id** after graph matching. They do **not** rank chunks with TF-IDF.

### 5.3 TF-IDF (RAG baseline only)

`rag/retriever.py` loads `data/tfidf_index.pkl` and returns top-k chunks by cosine similarity. That index is built offline from `data/chunks.jsonl`. TF-IDF is the **RAG** retriever. It is not an Agentic tool.

## 6. RAG vs GraphRAG vs Agentic

Used by `/api/compare` and by the public/hidden benchmark runners. Chatbot `/api/ask` Olympic questions use **only Agentic GraphRAG**.

| Pipeline | Retrieval | Generation | If evidence is missing |
|---|---|---|---|
| RAG | TF-IDF chunks | Ollama `qwen3:8b`, corpus-grounded prompt | `Insufficient evidence in the retrieved context.` (displayed as Insufficient evidence) |
| GraphRAG | TigerGraph entity/path match + corpus text | Same Ollama helper, only if evidence exists | Insufficient evidence; Ollama is skipped |
| Agentic GraphRAG | Dynamic tools over TigerGraph + corpus | Answer extracted from evidence; Ollama is not required to generate the answer | Insufficient evidence |

All three are local. None call an external web search API.

## 7. Local Ollama

- Default model: `qwen3:8b` (`config.py`)  
- Default URL: `http://localhost:11434`  
- Safe generation settings used in-repo: `temperature=0`, `seed=0`, `think=False`

Used when needed:

- RAG answer generation  
- GraphRAG answer generation (only with retrieved evidence)  
- Agentic **planner fallback** (not every step)  
- Router **General LLM** path (no Olympic context)

## 8. Insufficient evidence vs hallucination

The Olympic pipelines are corpus- and graph-grounded:

- RAG / GraphRAG prompts forbid outside knowledge and require the exact insufficient-evidence reply when context is missing.  
- Agentic returns an answer only after `evidence_is_sufficient()` and `extract_answer()`. No evidence → no guessed athlete or count.  
- Unsupported Olympic questions (for example a venue+date with no matching Event) stay on the Agentic route and return Insufficient evidence rather than a general-LLM guess.  
- Current/time-sensitive questions are not answered from stale Olympic pages or from an invented web tool.

Compare-pipeline **errors** (timeouts, exceptions) are shown as **Pipeline error**, not as Insufficient evidence.

## 9. Benchmark and dashboard path (separate)

```
Public eval  (data/eval_public.jsonl, 100 questions)
Hidden eval  (data/eval_hidden.jsonl, 50 questions)
  → eval/run_public_final.py  /  eval/run_hidden.py
  → RAG, GraphRAG, Agentic GraphRAG  (direct pipeline calls, not the chatbot router)
  → Metrics / frozen files under eval/results/
  → eval/dashboard  (read-only)
```

Official **frozen** public headline recorded in the dashboard copy: RAG **10%**, GraphRAG **60%**, Agentic **92%**. Those files are evaluation artifacts; this architecture document does not change them.

Hidden evaluation stores raw pipeline outputs only. It has no gold labels, so hidden accuracy is not computed.

Question types in the public set: lookup, aggregation, temporal, multi_hop, superlative.

## 10. Offline preparation (not on the live request path)

These scripts build artifacts that the live system reads. They do not run per chatbot question:

- `ingestion/chunk_documents.py`, `ingestion/create_tfidf_index.py`  
- `ingestion/extract_*.py` and `graph/generate_data.py` for graph JSONL/TSV  
- `graph/*.gsql` schema and load files into TigerGraph  

## 11. Implemented vs not implemented

**Implemented**

- Chatbot UI, FastAPI `/api/ask` and `/api/compare`  
- Deterministic question router (four routes above)  
- Agentic orchestrator, investigation state, five read-only tools  
- TigerGraph schema and REST access  
- Corpus + TF-IDF RAG baseline  
- GraphRAG retriever  
- Local Ollama `qwen3:8b`  
- Insufficient-evidence stop condition  
- Read-only evaluation dashboard over frozen results  

**Implemented as an explicit unsupported path (not a live capability)**

- Current/Web: the router recognizes time-sensitive questions and returns a fixed “web search is not available” message  

**Not implemented (do not draw as live boxes)**

- External web search / browsing — **NOT IMPLEMENTED**  
- Cloud LLMs  
- Vector database  
- Writes to TigerGraph at query time  
- Using the chatbot router inside the official benchmark runners  

## 12. Key source files

| Concern | File |
|---|---|
| Chat UI | `frontend/index.html` |
| HTTP API | `backend/app.py` |
| Ask / compare adapters | `backend/service.py` |
| Question router | `backend/router.py` |
| Orchestrator / planner | `agent/orchestrator.py` |
| Investigation memory | `agent/state.py` |
| Tools | `agent/tools.py` |
| RAG | `rag/pipeline.py`, `rag/retriever.py` |
| GraphRAG | `rag/graph_retriever.py` |
| TigerGraph REST | `rag/graph_client.py` |
| Graph schema | `graph/schema.gsql` |
| Config | `config.py` |
| Public / hidden runners | `eval/run_public_final.py`, `eval/run_hidden.py` |
| Dashboard | `eval/dashboard/app.py` |
