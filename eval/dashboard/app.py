"""Local FastAPI dashboard over frozen official benchmark artifacts.

Read-only. Does not rerun pipelines or rewrite result files.

Start:
    python -m eval.dashboard
Then open http://127.0.0.1:8008
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from eval.dashboard.artifacts import available_runs, load_run
from eval.dashboard.views import dashboard_view

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agentic GraphRAG evaluation dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/runs")
def api_runs():
    return {"runs": available_runs()}


@app.get("/api/dashboard")
def api_dashboard(run: str = "public"):
    try:
        return dashboard_view(run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/health")
def api_health():
    bundle = load_run("public")
    return {
        "ok": True,
        "run": bundle["final"].get("run"),
        "n_questions": len(bundle["questions"]),
        "n_records": len(bundle["records"]),
    }


def main():
    import uvicorn
    uvicorn.run(
        "eval.dashboard.app:app",
        host="127.0.0.1",
        port=8008,
        reload=False,
    )


if __name__ == "__main__":
    main()
