"""Application-layer tests for the Agentic GraphRAG demo.

Does not modify official benchmark artifacts. Does not rerun the benchmark.
"""

import hashlib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.app import AskRequest, api_ask, api_compare, app, index
from backend.service import (
    INSUFFICIENT_DISPLAY,
    PIPELINE_ERROR_DISPLAY,
    _failed_pipeline,
    ask_question,
    compare_pipelines,
    format_investigation,
    route_and_answer,
)

PUBLIC_DIR = Path("eval") / "results" / "final_public_benchmark"
HIDDEN_DIR = Path("eval") / "results" / "hidden_benchmark"
SMOKE_QUESTION = (
    "Who won gold in the women's 57 kg judo event at the 2016 Summer Olympics?"
)
ARTIFACTS = [
    PUBLIC_DIR / "benchmark_public_final.json",
    PUBLIC_DIR / "benchmark_public_summary.json",
    PUBLIC_DIR / "benchmark_public_failures.json",
    PUBLIC_DIR / "benchmark_public_failures.csv",
    PUBLIC_DIR / "benchmark_public_report.md",
    HIDDEN_DIR / "hidden_results.json",
    HIDDEN_DIR / "hidden_summary.json",
    HIDDEN_DIR / "hidden_report.md",
    HIDDEN_DIR / "hidden_results.jsonl",
]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fake_result(answer="Rafaela Silva"):
    return {
        "question": SMOKE_QUESTION,
        "investigation_kind": "who_won",
        "state": {
            "iteration_count": 2,
            "evidence": [{
                "doc_id": "Q26217865",
                "title": "Judo at the 2016 Summer Olympics – Women's 57 kg",
                "text": "gold: Rafaela Silva",
            }],
            "identified_entities": {
                "Event": [{"id": "Q26217865", "type": "Event", "name": "Judo 57 kg"}],
                "Sport": [{"id": "judo", "type": "Sport", "name": "Judo"}],
            },
            "graph_paths": [{
                "path": "Document-(DESCRIBES)->Event",
                "from_id": "Q26217865",
                "to_id": "Q26217865",
            }],
            "unresolved_questions": [],
        },
        "trace": [{
            "iteration": 1,
            "chosen_action": "search_events",
            "tool_arguments": {"query": SMOKE_QUESTION, "top_k": 5},
            "tool_result": {
                "ok": True,
                "n_entities": 1,
                "n_events": 1,
                "n_documents": 0,
                "n_evidence": 0,
                "error": None,
            },
            "reason": "Question already names an event; search Event vertices.",
            "planner": "deterministic",
            "newly_discovered_entities": [],
            "newly_discovered_documents": [],
            "missing_before": ["events"],
            "missing_after": ["corpus_evidence"],
        }, {
            "iteration": 2,
            "chosen_action": "retrieve_documents",
            "tool_arguments": {"event_id": "Q26217865", "limit": 2},
            "tool_result": {
                "ok": True,
                "n_entities": 0,
                "n_events": 0,
                "n_documents": 1,
                "n_evidence": 1,
                "error": None,
            },
            "reason": "Corpus evidence is still missing for a known Event.",
            "planner": "deterministic",
            "newly_discovered_documents": ["Q26217865"],
            "missing_after": [],
        }],
        "stopping_status": "answered",
        "stop_reason": "sufficient evidence for an answer",
        "final_answer": answer,
        "supporting_evidence": True,
        "tools_used": ["search_events", "retrieve_documents"],
    }


def _fake_rag(answer="Wrong athlete"):
    return {
        "answer": answer,
        "retrieved_chunks": [{
            "doc_id": "Q1",
            "title": "Some page",
            "text": "unrelated text",
            "score": 0.41,
            "chunk_id": "Q1#0",
        }],
        "document_ids": ["Q1"],
        "similarity_scores": [0.41],
        "context_char_count": 14,
        "context_token_count": 2,
    }


def _fake_graph():
    return {
        "matched_entities": {
            "games": [{"id": "Q8613", "name": "2016 Summer Olympics", "year": 2016, "season": "Summer"}],
            "sports": [{"id": "judo", "name": "Judo"}],
            "venues": [],
            "countries": [],
            "gender": "women",
            "temporal": {},
            "events": [{"id": "Q26217865", "name": "Judo 57 kg", "score": 1.0}],
            "athletes": [{"id": "a1", "name": "Rafaela Silva", "event_id": "Q26217865"}],
        },
        "graph_paths": [{
            "path": "Document-(DESCRIBES)->Event",
            "from_id": "Q26217865",
            "to_id": "Q26217865",
        }],
        "document_ids": ["Q26217865"],
        "evidence": [{
            "doc_id": "Q26217865",
            "title": "Judo at the 2016 Summer Olympics – Women's 57 kg",
            "text": "gold: Rafaela Silva",
            "score": 1.0,
        }],
        "scores": [{"event_id": "Q26217865", "name": "Judo 57 kg", "score": 1.0}],
        "stats": {"n_matched_events": 1, "n_documents": 1},
    }


class AppContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {path: _sha256(path) for path in ARTIFACTS if path.is_file()}

    def test_ui_route_exists(self):
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/", paths)
        self.assertIn("/api/ask", paths)
        self.assertIn("/api/compare", paths)
        response = index()
        self.assertTrue(str(response.path).endswith("index.html"))
        html = Path(response.path).read_text(encoding="utf-8")
        self.assertIn("Agentic GraphRAG", html)
        self.assertIn("Ask complex questions. The agent investigates using graph, documents, and evidence.", html)
        self.assertIn("/api/ask", html)
        self.assertIn("Investigation Trace", html)
        self.assertIn("route_label", html)
        self.assertIn("Compare RAG vs GraphRAG vs Agentic", html)
        self.assertIn("/api/compare", html)
        self.assertIn("Local", html)

    def test_ask_invokes_existing_orchestrator(self):
        fake = _fake_result()
        with patch("backend.service.investigate", return_value=fake) as mocked:
            payload = ask_question(SMOKE_QUESTION)
            mocked.assert_called_once_with(SMOKE_QUESTION)
            self.assertEqual(payload["answer"], "Rafaela Silva")
            self.assertEqual(payload["steps"], 2)
            self.assertEqual(payload["stopping_reason"], "sufficient evidence for an answer")
            self.assertTrue(payload["trace"])
            self.assertTrue(payload["evidence"])
            self.assertFalse(payload["llm_used"])
            self.assertEqual(payload["tools_used"], ["search_events", "retrieve_documents"])
            endpoint = api_ask(AskRequest(question=SMOKE_QUESTION))
            self.assertEqual(endpoint["answer"], "Rafaela Silva")
            self.assertEqual(endpoint["route_label"], "Agentic GraphRAG")
            self.assertEqual(mocked.call_count, 2)

    def test_compare_invokes_existing_pipeline_interfaces(self):
        fake_agentic = _fake_result()
        fake_rag = _fake_rag()
        fake_graph = _fake_graph()
        with patch("backend.service.rag_answer", return_value=fake_rag) as rag_mock, \
             patch("backend.service.retrieve_graph", return_value=fake_graph) as graph_mock, \
             patch("backend.service.ask_ollama", return_value="Rafaela Silva") as ollama_mock, \
             patch("backend.service.investigate", return_value=fake_agentic) as inv_mock, \
             patch("rag.pipeline.ask_ollama") as unused_rag_ollama, \
             patch("rag.graph_retriever.retrieve_graph") as unused_tg:
            payload = api_compare(AskRequest(question=SMOKE_QUESTION))

        rag_mock.assert_called_once_with(SMOKE_QUESTION)
        graph_mock.assert_called_once_with(SMOKE_QUESTION)
        ollama_mock.assert_called_once()
        inv_mock.assert_called_once_with(SMOKE_QUESTION)
        unused_rag_ollama.assert_not_called()
        unused_tg.assert_not_called()

        for result in (payload,):
            self.assertEqual(result["question"], SMOKE_QUESTION)
            self.assertIn("rag", result)
            self.assertIn("graphrag", result)
            self.assertIn("agentic", result)
            self.assertIn("comparison", result)
            self.assertEqual(result["rag"]["answer"], "Wrong athlete")
            self.assertTrue(result["rag"]["evidence"])
            self.assertEqual(result["graphrag"]["answer"], "Rafaela Silva")
            self.assertTrue(result["graphrag"]["entities"])
            self.assertTrue(result["graphrag"]["evidence"])
            self.assertEqual(result["agentic"]["answer"], "Rafaela Silva")
            self.assertTrue(result["agentic"]["trace"])
            self.assertEqual(len(result["agentic"]["trace"]), 2)
            self.assertEqual(result["agentic"]["tools_used"], ["search_events", "retrieve_documents"])
            cmp = result["comparison"]
            self.assertEqual(cmp["rag_answer"], "Wrong athlete")
            self.assertEqual(cmp["graphrag_answer"], "Rafaela Silva")
            self.assertEqual(cmp["agentic_answer"], "Rafaela Silva")
            self.assertEqual(cmp["agreeing_pipelines"], ["graphrag", "agentic"])
            self.assertTrue(cmp["agentic_used_multiple_tools"])
            self.assertIn("rag", cmp["latencies_sec"])
            self.assertIn("graphrag", cmp["latencies_sec"])
            self.assertIn("agentic", cmp["latencies_sec"])

        self.assertEqual(rag_mock.call_count, 1)
        self.assertEqual(graph_mock.call_count, 1)
        self.assertEqual(ollama_mock.call_count, 1)
        self.assertEqual(inv_mock.call_count, 1)

    def test_compare_partial_failure_still_returns_other_pipelines(self):
        def boom(_question):
            raise RuntimeError("ollama unavailable")

        payload = compare_pipelines(
            SMOKE_QUESTION,
            rag_fn=boom,
            retrieve_fn=lambda _q: _fake_graph(),
            generate_fn=lambda _q, _c: "Rafaela Silva",
            investigate_fn=lambda _q: _fake_result(),
        )
        self.assertFalse(payload["rag"]["ok"])
        self.assertEqual(payload["rag"]["answer"], PIPELINE_ERROR_DISPLAY)
        self.assertNotEqual(payload["rag"]["answer"], INSUFFICIENT_DISPLAY)
        self.assertIn("ollama unavailable", payload["rag"]["error"])
        self.assertTrue(payload["graphrag"]["ok"])
        self.assertEqual(payload["graphrag"]["answer"], "Rafaela Silva")
        self.assertTrue(payload["agentic"]["ok"])
        self.assertTrue(payload["agentic"]["trace"])
        self.assertEqual(payload["comparison"]["agreeing_pipelines"], ["graphrag", "agentic"])

    def test_compare_skips_graphrag_ollama_without_evidence(self):
        empty_graph = {
            "matched_entities": {},
            "graph_paths": [],
            "document_ids": [],
            "evidence": [],
            "scores": [],
            "stats": {},
        }
        generate = Mock(side_effect=AssertionError("Ollama should not be called"))
        payload = compare_pipelines(
            SMOKE_QUESTION,
            rag_fn=lambda _q: {
                "answer": "Insufficient evidence in the retrieved context.",
                "retrieved_chunks": [],
                "document_ids": [],
                "similarity_scores": [],
                "context_char_count": 0,
                "context_token_count": 0,
            },
            retrieve_fn=lambda _q: empty_graph,
            generate_fn=generate,
            investigate_fn=lambda _q: _fake_result(),
        )
        generate.assert_not_called()
        self.assertTrue(payload["rag"]["ok"])
        self.assertIsNone(payload["rag"]["error"])
        self.assertEqual(payload["rag"]["answer"], INSUFFICIENT_DISPLAY)
        self.assertTrue(payload["graphrag"]["ok"])
        self.assertEqual(payload["graphrag"]["answer"], INSUFFICIENT_DISPLAY)
        self.assertFalse(payload["graphrag"]["llm_used"])
        self.assertEqual(payload["agentic"]["answer"], "Rafaela Silva")
        self.assertTrue(payload["agentic"]["trace"])

    def test_empty_answer_is_insufficient_evidence(self):
        formatted = format_investigation(_fake_result(answer=""))
        self.assertEqual(formatted["answer"], INSUFFICIENT_DISPLAY)

    def test_compare_preserves_numeric_agentic_answer_and_original_question(self):
        shooting_q = (
            "How many shooting events at the 2004 Summer Olympics had more than "
            "37 competitors?"
        )
        seen = {"rag": [], "graph": [], "agentic": []}

        def rag_fn(question):
            seen["rag"].append(question)
            return {
                "answer": "Insufficient evidence in the retrieved context.",
                "retrieved_chunks": [],
                "document_ids": [],
                "similarity_scores": [],
                "context_char_count": 0,
                "context_token_count": 0,
            }

        def retrieve_fn(question):
            seen["graph"].append(question)
            return {
                "matched_entities": {},
                "graph_paths": [],
                "document_ids": [],
                "evidence": [],
                "scores": [],
                "stats": {},
            }

        def investigate_fn(question):
            seen["agentic"].append(question)
            return _fake_result(answer="8")

        payload = compare_pipelines(
            shooting_q,
            rag_fn=rag_fn,
            retrieve_fn=retrieve_fn,
            generate_fn=Mock(side_effect=AssertionError("Ollama should not run")),
            investigate_fn=investigate_fn,
        )
        self.assertEqual(seen["rag"], [shooting_q])
        self.assertEqual(seen["graph"], [shooting_q])
        self.assertEqual(seen["agentic"], [shooting_q])
        self.assertEqual(payload["question"], shooting_q)
        self.assertEqual(payload["agentic"]["answer"], "8")
        self.assertTrue(payload["agentic"]["ok"])
        self.assertIsNone(payload["agentic"]["error"])
        self.assertTrue(payload["rag"]["ok"])
        self.assertEqual(payload["rag"]["answer"], INSUFFICIENT_DISPLAY)
        self.assertTrue(payload["graphrag"]["ok"])
        self.assertEqual(payload["graphrag"]["answer"], INSUFFICIENT_DISPLAY)
        self.assertEqual(payload["comparison"]["agentic_answer"], "8")

    def test_failed_pipeline_is_not_insufficient_evidence(self):
        failed = _failed_pipeline("agentic", SMOKE_QUESTION, 1.25, "timed out")
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["answer"], PIPELINE_ERROR_DISPLAY)
        self.assertNotEqual(failed["answer"], INSUFFICIENT_DISPLAY)
        self.assertEqual(failed["error"], "timed out")
        self.assertEqual(failed["question"], SMOKE_QUESTION)
        self.assertEqual(failed["pipeline"], "agentic")
        self.assertEqual(failed["latency_sec"], 1.25)

    def test_compare_pipeline_exceptions_preserve_error_not_insufficient(self):
        shooting_q = (
            "How many shooting events at the 2004 Summer Olympics had more than "
            "37 competitors?"
        )

        def boom(_question):
            raise TimeoutError("timed out")

        payload = compare_pipelines(
            shooting_q,
            rag_fn=boom,
            retrieve_fn=boom,
            generate_fn=Mock(side_effect=AssertionError("should not generate")),
            investigate_fn=boom,
        )
        for name in ("rag", "graphrag", "agentic"):
            row = payload[name]
            self.assertFalse(row["ok"], name)
            self.assertEqual(row["answer"], PIPELINE_ERROR_DISPLAY, name)
            self.assertNotEqual(row["answer"], INSUFFICIENT_DISPLAY, name)
            self.assertIn("timed out", row["error"] or "")
            self.assertEqual(row["question"], shooting_q)
        html = Path("frontend/index.html").read_text(encoding="utf-8")
        self.assertIn("row.ok === false", html)
        self.assertIn("Pipeline error", html)
        self.assertIn("row.error", html)

    def test_artifacts_untouched(self):
        for path, expected in self.hashes.items():
            self.assertEqual(_sha256(path), expected, str(path))


class AppLiveSmokeTests(unittest.TestCase):
    def test_live_public_judo_question(self):
        result = route_and_answer(SMOKE_QUESTION)
        print("LIVE SMOKE question:", SMOKE_QUESTION)
        print("LIVE SMOKE answer:", result.get("answer"))
        print("LIVE SMOKE route:", result.get("route_label"))
        print("LIVE SMOKE steps:", result.get("steps"))
        print("LIVE SMOKE tools:", result.get("tools_used"))
        print("LIVE SMOKE stop:", result.get("stopping_reason"))
        self.assertEqual(result["route_label"], "Agentic GraphRAG")
        self.assertTrue(result["answer"])
        self.assertIn("Silva", result["answer"])
        self.assertGreaterEqual(result["steps"], 1)
        self.assertTrue(result["trace"])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["stopping_reason"])
        self.assertIn("search_events", result["tools_used"])


if __name__ == "__main__":
    unittest.main()
