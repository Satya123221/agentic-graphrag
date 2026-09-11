"""Tests for the deterministic question router.

Does not modify official benchmark artifacts. Does not rerun the benchmark.
Does not rewrite the Agentic planner.
"""

import hashlib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.app import AskRequest, api_ask
from backend.router import (
    CURRENT_WEB_ANSWER,
    ROUTE_AGENTIC,
    ROUTE_CALCULATION,
    ROUTE_CURRENT,
    ROUTE_GENERAL,
    classify,
    parse_math_expression,
)
from backend.service import (
    INSUFFICIENT_DISPLAY,
    ask_question,
    route_and_answer,
)

PUBLIC_DIR = Path("eval") / "results" / "final_public_benchmark"
HIDDEN_DIR = Path("eval") / "results" / "hidden_benchmark"
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

OLYMPIC_Q = (
    "Who won gold in the women's 57 kg judo event at the 2016 Summer Olympics?"
)
PHOTOSYNTHESIS_Q = "What is photosynthesis?"
CALC_Q = "What is 25 * 16?"
UNSUPPORTED_OLYMPIC_Q = (
    "Which athlete won gold at the event held at ExCeL on 14 August 2016?"
)
INDIA_SHOOTING_Q = "who participated from india in shooting"
SUPERLATIVE_Q = (
    "Which event had the highest number of competitors at the 2004 Summer Olympics?"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fake_agentic(question, answer="Rafaela Silva"):
    return {
        "question": question,
        "investigation_kind": "who_won",
        "state": {
            "iteration_count": 2,
            "evidence": [{
                "doc_id": "Q26217865",
                "title": "Judo at the 2016 Summer Olympics – Women's 57 kg",
                "text": "gold: Rafaela Silva",
            }],
            "identified_entities": {},
            "graph_paths": [],
            "unresolved_questions": [],
        },
        "trace": [{
            "iteration": 1,
            "chosen_action": "search_events",
            "tool_arguments": {},
            "tool_result": {
                "ok": True,
                "n_entities": 1,
                "n_events": 1,
                "n_documents": 0,
                "n_evidence": 0,
                "error": None,
            },
            "reason": "search events",
            "planner": "deterministic",
        }],
        "stopping_status": "answered" if answer else "stopped",
        "stop_reason": (
            "sufficient evidence for an answer" if answer else "No useful next action remains."
        ),
        "final_answer": answer,
        "supporting_evidence": bool(answer),
        "tools_used": ["search_events"],
    }


class RouterClassifyTests(unittest.TestCase):
    def test_olympic_question_is_agentic(self):
        decision = classify(OLYMPIC_Q)
        self.assertEqual(decision["route"], ROUTE_AGENTIC)
        self.assertEqual(decision["route_label"], "Agentic GraphRAG")

    def test_photosynthesis_is_general_llm(self):
        decision = classify(PHOTOSYNTHESIS_Q)
        self.assertEqual(decision["route"], ROUTE_GENERAL)
        self.assertEqual(decision["route_label"], "General LLM")

    def test_simple_calculation_is_calculation(self):
        decision = classify(CALC_Q)
        self.assertEqual(decision["route"], ROUTE_CALCULATION)
        self.assertEqual(decision["route_label"], "Calculation")
        self.assertEqual(parse_math_expression(CALC_Q), "25 * 16")

    def test_unsupported_excel_question_is_agentic(self):
        decision = classify(UNSUPPORTED_OLYMPIC_Q)
        self.assertEqual(decision["route"], ROUTE_AGENTIC)

    def test_india_shooting_is_agentic(self):
        decision = classify(INDIA_SHOOTING_Q)
        self.assertEqual(decision["route"], ROUTE_AGENTIC)

    def test_superlative_is_agentic(self):
        decision = classify(SUPERLATIVE_Q)
        self.assertEqual(decision["route"], ROUTE_AGENTIC)

    def test_aggregation_is_not_calculation(self):
        question = (
            "How many shooting events at the 2004 Summer Olympics had more than "
            "37 competitors?"
        )
        decision = classify(question)
        self.assertEqual(decision["route"], ROUTE_AGENTIC)

    def test_current_question_is_current_web(self):
        decision = classify("What is the weather today?")
        self.assertEqual(decision["route"], ROUTE_CURRENT)
        self.assertEqual(decision["route_label"], "Current/Web")


class RouterDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes = {path: _sha256(path) for path in ARTIFACTS if path.is_file()}

    def test_olympic_question_uses_agentic_graphrag(self):
        investigate_fn = Mock(return_value=_fake_agentic(OLYMPIC_Q))
        general_llm_fn = Mock(side_effect=AssertionError("general LLM must not run"))
        with patch("rag.pipeline.rag_answer") as rag, \
             patch("rag.graph_retriever.retrieve_graph") as graph:
            payload = route_and_answer(
                OLYMPIC_Q,
                investigate_fn=investigate_fn,
                general_llm_fn=general_llm_fn,
            )
        investigate_fn.assert_called_once_with(OLYMPIC_Q)
        general_llm_fn.assert_not_called()
        rag.assert_not_called()
        graph.assert_not_called()
        self.assertEqual(payload["route"], ROUTE_AGENTIC)
        self.assertEqual(payload["route_label"], "Agentic GraphRAG")
        self.assertEqual(payload["answer"], "Rafaela Silva")
        self.assertTrue(payload["trace"])

    def test_photosynthesis_uses_general_llm_not_corpus(self):
        investigate_fn = Mock(side_effect=AssertionError("investigate must not run"))
        general_llm_fn = Mock(return_value="Photosynthesis converts light into chemical energy.")
        with patch("rag.pipeline.rag_answer") as rag, \
             patch("rag.graph_retriever.retrieve_graph") as graph, \
             patch("backend.router.ask_general_llm") as unused_ollama:
            payload = route_and_answer(
                PHOTOSYNTHESIS_Q,
                investigate_fn=investigate_fn,
                general_llm_fn=general_llm_fn,
            )
        investigate_fn.assert_not_called()
        unused_ollama.assert_not_called()
        rag.assert_not_called()
        graph.assert_not_called()
        general_llm_fn.assert_called_once_with(PHOTOSYNTHESIS_Q)
        self.assertEqual(payload["route"], ROUTE_GENERAL)
        self.assertEqual(payload["route_label"], "General LLM")
        self.assertIn("Photosynthesis", payload["answer"])
        self.assertTrue(payload["llm_used"])
        self.assertEqual(payload["evidence"], [])
        self.assertEqual(payload["tools_used"], [])

    def test_simple_calculation_is_deterministic(self):
        investigate_fn = Mock(side_effect=AssertionError("investigate must not run"))
        general_llm_fn = Mock(side_effect=AssertionError("general LLM must not run"))
        payload = route_and_answer(
            CALC_Q,
            investigate_fn=investigate_fn,
            general_llm_fn=general_llm_fn,
        )
        investigate_fn.assert_not_called()
        general_llm_fn.assert_not_called()
        self.assertEqual(payload["route"], ROUTE_CALCULATION)
        self.assertEqual(payload["route_label"], "Calculation")
        self.assertEqual(payload["answer"], "400")
        self.assertFalse(payload["llm_used"])

    def test_unsupported_olympic_question_is_insufficient_evidence(self):
        investigate_fn = Mock(return_value=_fake_agentic(UNSUPPORTED_OLYMPIC_Q, answer=""))
        general_llm_fn = Mock(side_effect=AssertionError("must not guess from general knowledge"))
        payload = route_and_answer(
            UNSUPPORTED_OLYMPIC_Q,
            investigate_fn=investigate_fn,
            general_llm_fn=general_llm_fn,
        )
        investigate_fn.assert_called_once_with(UNSUPPORTED_OLYMPIC_Q)
        general_llm_fn.assert_not_called()
        self.assertEqual(payload["route"], ROUTE_AGENTIC)
        self.assertEqual(payload["answer"], INSUFFICIENT_DISPLAY)

    def test_india_shooting_uses_agentic_graphrag(self):
        investigate_fn = Mock(
            return_value=_fake_agentic(INDIA_SHOOTING_Q, answer="Abhinav Bindra")
        )
        payload = route_and_answer(INDIA_SHOOTING_Q, investigate_fn=investigate_fn)
        investigate_fn.assert_called_once_with(INDIA_SHOOTING_Q)
        self.assertEqual(payload["route_label"], "Agentic GraphRAG")
        self.assertEqual(payload["answer"], "Abhinav Bindra")

    def test_superlative_uses_agentic_graphrag(self):
        investigate_fn = Mock(
            return_value=_fake_agentic(
                SUPERLATIVE_Q,
                answer="Cycling at the 2004 Summer Olympics – Men's individual road race",
            )
        )
        payload = route_and_answer(SUPERLATIVE_Q, investigate_fn=investigate_fn)
        investigate_fn.assert_called_once_with(SUPERLATIVE_Q)
        self.assertEqual(payload["route"], ROUTE_AGENTIC)
        self.assertIn("road race", payload["answer"])

    def test_current_question_does_not_use_web_or_corpus(self):
        investigate_fn = Mock(side_effect=AssertionError("investigate must not run"))
        general_llm_fn = Mock(side_effect=AssertionError("general LLM must not invent live facts"))
        payload = route_and_answer(
            "What is the weather today?",
            investigate_fn=investigate_fn,
            general_llm_fn=general_llm_fn,
        )
        investigate_fn.assert_not_called()
        general_llm_fn.assert_not_called()
        self.assertEqual(payload["route"], ROUTE_CURRENT)
        self.assertEqual(payload["route_label"], "Current/Web")
        self.assertEqual(payload["answer"], CURRENT_WEB_ANSWER)

    def test_api_ask_adds_route_metadata(self):
        fake = _fake_agentic(OLYMPIC_Q)
        with patch("backend.service.investigate", return_value=fake):
            payload = api_ask(AskRequest(question=OLYMPIC_Q))
        self.assertEqual(payload["route"], ROUTE_AGENTIC)
        self.assertEqual(payload["route_label"], "Agentic GraphRAG")
        self.assertEqual(payload["answer"], "Rafaela Silva")

    def test_ask_question_remains_agentic_only(self):
        fake = _fake_agentic(PHOTOSYNTHESIS_Q, answer="should not be used")
        with patch("backend.service.investigate", return_value=fake) as mocked:
            payload = ask_question(PHOTOSYNTHESIS_Q)
        mocked.assert_called_once_with(PHOTOSYNTHESIS_Q)
        self.assertNotIn("route", payload)
        self.assertEqual(payload["answer"], "should not be used")

    def test_artifacts_untouched(self):
        self.assertTrue(self.hashes)
        for path, expected in self.hashes.items():
            self.assertEqual(_sha256(path), expected, str(path))


if __name__ == "__main__":
    unittest.main()
