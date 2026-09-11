"""Load frozen benchmark artifacts. Never writes to result files."""

from functools import lru_cache
from pathlib import Path

# Public filenames stay stable for existing tests and the public dashboard.
FINAL_NAME = "benchmark_public_final.json"
SUMMARY_NAME = "benchmark_public_summary.json"
REPORT_NAME = "benchmark_public_report.md"
FAILURES_JSON_NAME = "benchmark_public_failures.json"
FAILURES_CSV_NAME = "benchmark_public_failures.csv"

HIDDEN_FINAL_NAME = "hidden_results.json"
HIDDEN_SUMMARY_NAME = "hidden_summary.json"
HIDDEN_REPORT_NAME = "hidden_report.md"

RUN_SPECS = {
    "public": {
        "directory": Path("eval") / "results" / "final_public_benchmark",
        "final": FINAL_NAME,
        "summary": SUMMARY_NAME,
        "report": REPORT_NAME,
        "failures_json": FAILURES_JSON_NAME,
        "failures_csv": FAILURES_CSV_NAME,
    },
    "hidden": {
        "directory": Path("eval") / "results" / "hidden_benchmark",
        "final": HIDDEN_FINAL_NAME,
        "summary": HIDDEN_SUMMARY_NAME,
        "report": HIDDEN_REPORT_NAME,
        "failures_json": None,
        "failures_csv": None,
    },
}

# Backward-compatible mapping of run id -> directory.
RUNS = {run_id: spec["directory"] for run_id, spec in RUN_SPECS.items()}

PIPELINES = ("rag", "graphrag", "agentic")
QTYPES = ("lookup", "aggregation", "temporal", "multi_hop", "superlative")


def run_dir(run_id="public"):
    spec = RUN_SPECS.get(run_id)
    if spec is None:
        raise KeyError("unknown benchmark run: " + run_id)
    return spec["directory"]


def available_runs():
    found = []
    for run_id, spec in RUN_SPECS.items():
        directory = spec["directory"]
        ready = (directory / spec["final"]).is_file() and (directory / spec["summary"]).is_file()
        found.append({
            "id": run_id,
            "path": str(directory).replace("\\", "/"),
            "ready": ready,
        })
    return found


def _read_json(path):
    import json
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path):
    with path.open(encoding="utf-8") as handle:
        return handle.read()


@lru_cache(maxsize=4)
def load_run(run_id="public"):
    """Load one frozen run. Artifacts are opened read-only."""
    spec = RUN_SPECS.get(run_id)
    if spec is None:
        raise KeyError("unknown benchmark run: " + run_id)
    directory = spec["directory"]
    final_path = directory / spec["final"]
    summary_path = directory / spec["summary"]
    if not final_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "frozen artifacts not found for run " + run_id + " in " + str(directory)
        )
    final = _read_json(final_path)
    summary = _read_json(summary_path)
    failures = {}
    failures_name = spec.get("failures_json")
    failures_path = directory / failures_name if failures_name else directory / FAILURES_JSON_NAME
    if failures_name and failures_path.is_file():
        failures = _read_json(failures_path)
    report = ""
    report_name = spec.get("report")
    report_path = directory / report_name if report_name else directory / REPORT_NAME
    if report_name and report_path.is_file():
        report = _read_text(report_path)
    records = list(final.get("records") or [])
    questions = list(final.get("questions") or [])
    csv_name = spec.get("failures_csv")
    csv_path = directory / csv_name if csv_name else directory / FAILURES_CSV_NAME
    return {
        "run_id": run_id,
        "directory": str(directory).replace("\\", "/"),
        "final": final,
        "summary": summary,
        "failures": failures,
        "report": report,
        "records": records,
        "questions": questions,
        "paths": {
            "final": str(final_path).replace("\\", "/"),
            "summary": str(summary_path).replace("\\", "/"),
            "failures_json": str(failures_path).replace("\\", "/"),
            "failures_csv": str(csv_path).replace("\\", "/"),
            "report": str(report_path).replace("\\", "/"),
        },
    }
