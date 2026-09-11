"""Local Agentic GraphRAG application API.

Thin FastAPI layer over existing RAG, GraphRAG, and Agentic interfaces.
Does not reimplement retrieval or planning.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.service import compare_pipelines, route_and_answer

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Agentic GraphRAG")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "agentic-graphrag-app"}


@app.post("/api/ask")
def api_ask(payload: AskRequest):
    try:
        return route_and_answer(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/compare")
def api_compare(payload: AskRequest):
    try:
        return compare_pipelines(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main():
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8010,
        reload=False,
    )


if __name__ == "__main__":
    main()
