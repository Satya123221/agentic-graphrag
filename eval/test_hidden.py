"""Tests for the hidden evaluation runner.

Does not run the real 150-record hidden benchmark.
Does not modify official public benchmark artifacts.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config import HIDDEN_EVAL_PATH
from eval.run_hidden import (
    EXPECTED_QUESTIONS,
    EXPECTED_RECORDS,
    PIPELINES,
    default_runners,
    iter_jobs,
    load_completed,
    load_hidden_questions,
    run_hidden,
    summarize_hidden,
)

PUBLIC_DIR = Path("eval") / "results" / "final_public_benchmark"
ARTIFACTS = [
    PUBLIC_DIR / "benchmark_public_final.json",
    PUBLIC_DIR / "benchmark_public_summary.json",
    PUBLIC_DIR / "benchmark_public_failures.json",
    PUBLIC_DIR / "benchmark_public_failures.csv",
    PUBLIC_DIR / "benchmark_public_report.md",
]

ACCURACY_KEY_MARKERS = (
    "accuracy",
    "exact_match",
    "normalized_match",
    "gold_answer",
    "gold_doc",
    "correctness",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_keys(payload):
    keys = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key))
            keys.update(_collect_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_collect_keys(item))
    return keys


def _fake_trace():
    return [{
        "iteration": 1,
        "chosen_action": "search_events",
        "tool_arguments": {"query": "hidden", "top_k": 5},
        "tool_result": {
            "ok": True,
            "n_entities": 1,
            "n_events": 1,
            "n_documents": 0,
            "n_evidence": 0,
        },
        "reason": "search events",
        "planner": "deterministic",
    }, {
        "iteration": 2,
        "chosen_action": "retrieve_documents",
        "tool_arguments": {"event_id": "Q1", "limit": 2},
        "tool_result": {
            "ok": True,
            "n_entities": 0,
            "n_events": 0,
            "n_documents": 1,
            "n_evidence": 1,
        },
        "reason": "retrieve evidence",
        "planner": "deterministic",
    }]


def _mock_fns():
    rag = Mock(return_value={
        "answer": "rag-answer",
        "retrieved_chunks": [{
            "doc_id": "D1",
            "title": "t",
            "text": "x",
            "score": 0.1,
            "chunk_id": "D1#0",
        }],
        "document_ids": ["D1"],
        "similarity_scores": [0.1],
        "context_char_count": 1,
        "context_token_count": 1,
    })
    retrieve = Mock(return_value={
        "matched_entities": {"events": [{"id": "E1", "name": "Event"}]},
        "graph_paths": [{
            "path": "Document-(DESCRIBES)->Event",
            "from_id": "D1",
            "to_id": "E1",
        }],
        "document_ids": ["D1"],
        "evidence": [{"doc_id": "D1", "title": "t", "text": "graph evidence"}],
        "scores": [{"event_id": "E1", "name": "Event", "score": 1.0}],
        "stats": {"n_documents": 1},
    })
    generate = Mock(return_value="graphrag-answer")
    investigate = Mock(return_value={
        "question": "q",
        "final_answer": "agentic-answer",
        "investigation_kind": "lookup",
        "stop_reason": "sufficient evidence for an answer",
        "stopping_status": "answered",
        "supporting_evidence": True,
        "tools_used": ["search_events", "retrieve_documents"],
        "trace": _fake_trace(),
        "state": {
            "iteration_count": 2,
            "document_ids": ["D1"],
            "evidence": [{"doc_id": "D1", "title": "t", "text": "agent evidence"}],
            "identified_entities": {
                "Event": [{"id": "E1", "type": "Event", "name": "Event"}],
            },
            "graph_paths": [{
                "path": "Document-(DESCRIBES)->Event",
                "from_id": "D1",
                "to_id": "E1",
            }],
            "unresolved_questions": [],
        },
    })
    return rag, retrieve, generate, investigate


class HiddenEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {path: _sha256(path) for path in ARTIFACTS if path.is_file()}
        cls.questions = load_hidden_questions(HIDDEN_EVAL_PATH)

    def _run(self, out_dir, rag, retrieve, generate, investigate):
        with patch("eval.run_hidden.rag_answer") as real_rag, \
             patch("eval.run_hidden.retrieve_graph") as real_graph, \
             patch("eval.run_hidden.ask_ollama") as real_ollama, \
             patch("eval.run_hidden.investigate") as real_agent:
            payload = run_hidden(
                eval_path=HIDDEN_EVAL_PATH,
                out_dir=out_dir,
                runners=default_runners(
                    rag_fn=rag,
                    retrieve_fn=retrieve,
                    generate_fn=generate,
                    investigate_fn=investigate,
                ),
                do_preflight=False,
                capture_llm=False,
            )
            real_rag.assert_not_called()
            real_graph.assert_not_called()
            real_ollama.assert_not_called()
            real_agent.assert_not_called()
        return payload

    def test_hidden_jsonl_is_parsed_and_has_50_questions(self):
        self.assertTrue(HIDDEN_EVAL_PATH.is_file())
        self.assertEqual(len(self.questions), EXPECTED_QUESTIONS)
        self.assertEqual(len(self.questions), 50)
        qids = [row["qid"] for row in self.questions]
        self.assertEqual(len(set(qids)), 50)
        first = self.questions[0]
        self.assertEqual(first["qid"], "eval-001")
        self.assertTrue(first["question"])
        self.assertNotIn("gold_answer", first)
        self.assertNotIn("answer", first)
        self.assertNotIn("gold_doc_ids", first)
        raw = json.loads(HIDDEN_EVAL_PATH.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(raw), {"qid", "question", "qtype"})

    def test_jobs_are_150_qid_pipeline_combinations(self):
        jobs = list(iter_jobs(self.questions))
        self.assertEqual(len(jobs), EXPECTED_RECORDS)
        self.assertEqual(len(jobs), 50 * 3)
        keys = {(item["qid"], pipeline) for item, pipeline in jobs}
        self.assertEqual(len(keys), 150)
        self.assertEqual({pipeline for _, pipeline in jobs}, set(PIPELINES))

    def test_mocked_run_creates_150_records_without_real_backends(self):
        rag, retrieve, generate, investigate = _mock_fns()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run(Path(tmp), rag, retrieve, generate, investigate)
            records = payload["records"]
            self.assertEqual(len(records), 150)
            self.assertEqual(payload["n_questions"], 50)
            self.assertEqual(rag.call_count, 50)
            self.assertEqual(retrieve.call_count, 50)
            self.assertEqual(generate.call_count, 50)
            self.assertEqual(investigate.call_count, 50)
            keys = {(row["qid"], row["pipeline"]) for row in records}
            self.assertEqual(len(keys), 150)
            by_pipe = {name: [r for r in records if r["pipeline"] == name] for name in PIPELINES}
            self.assertEqual(len(by_pipe["rag"]), 50)
            self.assertEqual(len(by_pipe["graphrag"]), 50)
            self.assertEqual(len(by_pipe["agentic"]), 50)
            rag_row = by_pipe["rag"][0]
            self.assertEqual(rag_row["answer"], "rag-answer")
            self.assertEqual(rag_row["retrieved_document_ids"], ["D1"])
            self.assertTrue(rag_row["evidence"])
            graph_row = by_pipe["graphrag"][0]
            self.assertEqual(graph_row["answer"], "graphrag-answer")
            self.assertTrue(graph_row["matched_entities"])
            self.assertTrue(graph_row["graph_paths"])
            agent_row = by_pipe["agentic"][0]
            self.assertEqual(agent_row["answer"], "agentic-answer")
            self.assertEqual(len(agent_row["agent_trace"]), 2)
            self.assertEqual(agent_row["agent_trace"][0]["chosen_action"], "search_events")
            self.assertEqual(agent_row["tools_used"], ["search_events", "retrieve_documents"])
            self.assertEqual(agent_row["agent_iterations"], 2)
            self.assertIsNone(agent_row["error"])
            for required in ("qid", "question", "pipeline", "answer", "latency_seconds", "error"):
                self.assertIn(required, rag_row)
                self.assertIn(required, graph_row)
                self.assertIn(required, agent_row)
            jsonl = Path(tmp) / "hidden_results.jsonl"
            self.assertTrue(jsonl.is_file())
            self.assertEqual(len(jsonl.read_text(encoding="utf-8").splitlines()), 150)

    def test_resume_skips_completed_qid_pipeline_and_prevents_duplicates(self):
        rag, retrieve, generate, investigate = _mock_fns()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = self._run(out_dir, rag, retrieve, generate, investigate)
            self.assertEqual(len(first["records"]), 150)
            jsonl = out_dir / "hidden_results.jsonl"
            original_lines = jsonl.read_text(encoding="utf-8").splitlines()
            jsonl.write_text(
                original_lines[0] + "\n" + original_lines[0] + "\n" + "\n".join(original_lines[1:]) + "\n",
                encoding="utf-8",
            )
            loaded, seen = load_completed(jsonl)
            self.assertEqual(len(loaded), 150)
            self.assertEqual(len(seen), 150)

            rag.reset_mock()
            retrieve.reset_mock()
            generate.reset_mock()
            investigate.reset_mock()
            second = self._run(out_dir, rag, retrieve, generate, investigate)
            self.assertEqual(len(second["records"]), 150)
            rag.assert_not_called()
            retrieve.assert_not_called()
            generate.assert_not_called()
            investigate.assert_not_called()
            keys = [(row["qid"], row["pipeline"]) for row in second["records"]]
            self.assertEqual(len(keys), len(set(keys)))

    def test_pipeline_error_is_recorded_and_run_continues(self):
        rag, retrieve, generate, investigate = _mock_fns()
        rag.side_effect = TimeoutError("timed out")
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run(Path(tmp), rag, retrieve, generate, investigate)
            self.assertEqual(len(payload["records"]), 150)
            rag_rows = [r for r in payload["records"] if r["pipeline"] == "rag"]
            agent_rows = [r for r in payload["records"] if r["pipeline"] == "agentic"]
            graph_rows = [r for r in payload["records"] if r["pipeline"] == "graphrag"]
            self.assertTrue(all(r.get("error") for r in rag_rows))
            self.assertTrue(all("timeout" in (r.get("error") or "").lower() for r in rag_rows))
            self.assertTrue(all(not r.get("error") for r in agent_rows))
            self.assertTrue(all(not r.get("error") for r in graph_rows))
            self.assertTrue(all(r.get("agent_trace") for r in agent_rows))
            self.assertEqual(payload["summary"]["n_failed"], 50)
            self.assertEqual(payload["summary"]["n_success"], 100)
            self.assertEqual(payload["summary"]["n_timeouts"], 50)

    def test_agentic_trace_is_preserved(self):
        rag, retrieve, generate, investigate = _mock_fns()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run(Path(tmp), rag, retrieve, generate, investigate)
            agent_rows = [r for r in payload["records"] if r["pipeline"] == "agentic"]
            self.assertEqual(len(agent_rows), 50)
            for row in agent_rows:
                self.assertEqual(len(row["agent_trace"]), 2)
                self.assertEqual(row["agent_trace"][1]["chosen_action"], "retrieve_documents")
                self.assertIn("tool_result", row["agent_trace"][0])
                self.assertEqual(row["answer"], "agentic-answer")

    def test_hidden_summary_has_no_accuracy_metrics(self):
        rag, retrieve, generate, investigate = _mock_fns()
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run(Path(tmp), rag, retrieve, generate, investigate)
            summary = payload["summary"]
            summary_from_fn = summarize_hidden(payload["records"], questions=self.questions)
            for blob in (summary, summary_from_fn):
                keys = {key.lower() for key in _collect_keys(blob)}
                for marker in ACCURACY_KEY_MARKERS:
                    self.assertTrue(
                        all(marker not in key for key in keys),
                        f"summary contained accuracy-related key matching {marker}: {keys}",
                    )
            dumped = json.dumps(summary)
            self.assertNotIn("overall_accuracy", dumped)
            self.assertNotIn("exact_match", dumped)
            self.assertNotIn("normalized_match", dumped)
            self.assertIn("n_questions", summary)
            self.assertIn("n_records", summary)
            self.assertIn("records_per_pipeline", summary)
            self.assertIn("n_success", summary)
            self.assertIn("n_failed", summary)
            self.assertIn("latency_seconds", summary)
            self.assertIn("agentic_steps", summary)
            self.assertIn("ollama_usage", summary)
            report = (Path(tmp) / "hidden_report.md").read_text(encoding="utf-8")
            self.assertNotIn("overall_accuracy", report)
            self.assertIn("No accuracy is computed", report)

    def test_public_benchmark_artifacts_untouched(self):
        self.assertTrue(self.hashes)
        for path, expected in self.hashes.items():
            self.assertEqual(_sha256(path), expected, str(path))


if __name__ == "__main__":
    unittest.main()
