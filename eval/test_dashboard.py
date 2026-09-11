"""Tests for the read-only evaluation dashboard.

Does not modify official benchmark artifacts. Does not rerun pipelines.
"""

import hashlib
import unittest
from pathlib import Path

from eval.dashboard.artifacts import FINAL_NAME, SUMMARY_NAME, available_runs, load_run
from eval.dashboard.views import dashboard_view, hidden_dashboard_view

PUBLIC_DIR = Path("eval") / "results" / "final_public_benchmark"
HIDDEN_DIR = Path("eval") / "results" / "hidden_benchmark"
INDEX_PATH = Path("eval") / "dashboard" / "static" / "index.html"

ACCURACY_KEY_MARKERS = (
    "accuracy",
    "exact_match",
    "normalized_match",
    "gold_answer",
    "gold_doc",
    "correctness",
    "retrieval_success",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DashboardArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.final_path = PUBLIC_DIR / FINAL_NAME
        cls.summary_path = PUBLIC_DIR / SUMMARY_NAME
        cls.failure_json = PUBLIC_DIR / "benchmark_public_failures.json"
        cls.failure_csv = PUBLIC_DIR / "benchmark_public_failures.csv"
        cls.report_path = PUBLIC_DIR / "benchmark_public_report.md"
        cls.hidden_final = HIDDEN_DIR / "hidden_results.json"
        cls.hidden_summary = HIDDEN_DIR / "hidden_summary.json"
        cls.hidden_report = HIDDEN_DIR / "hidden_report.md"
        cls.hidden_jsonl = HIDDEN_DIR / "hidden_results.jsonl"
        cls.hashes = {
            cls.final_path: _sha256(cls.final_path),
            cls.summary_path: _sha256(cls.summary_path),
            cls.failure_json: _sha256(cls.failure_json),
            cls.failure_csv: _sha256(cls.failure_csv),
            cls.report_path: _sha256(cls.report_path),
            cls.hidden_final: _sha256(cls.hidden_final),
            cls.hidden_summary: _sha256(cls.hidden_summary),
            cls.hidden_report: _sha256(cls.hidden_report),
            cls.hidden_jsonl: _sha256(cls.hidden_jsonl),
        }
        cls.bundle = load_run("public")
        cls.view = dashboard_view("public")
        cls.hidden_bundle = load_run("hidden")
        cls.hidden_view = dashboard_view("hidden")

    def test_artifacts_are_loadable(self):
        self.assertTrue(self.final_path.is_file())
        self.assertTrue(self.summary_path.is_file())
        self.assertTrue(self.failure_json.is_file())
        self.assertTrue(self.failure_csv.is_file())
        self.assertTrue(self.report_path.is_file())
        self.assertEqual(self.bundle["final"].get("run"), "official_final_public")
        self.assertEqual(self.bundle["final"].get("n_questions"), 100)
        self.assertTrue(any(row["id"] == "public" and row["ready"] for row in available_runs()))

    def test_official_counts(self):
        counts = self.view["counts"]
        self.assertEqual(counts["n_questions"], 100)
        self.assertEqual(counts["n_records"], 300)
        self.assertEqual(counts["n_rag"], 100)
        self.assertEqual(counts["n_graphrag"], 100)
        self.assertEqual(counts["n_agentic"], 100)

    def test_official_accuracy(self):
        by_name = {row["pipeline"]: row for row in self.view["overall"]}
        self.assertEqual(by_name["rag"]["accuracy"], 0.1)
        self.assertEqual(by_name["graphrag"]["accuracy"], 0.6)
        self.assertEqual(by_name["agentic"]["accuracy"], 0.92)
        self.assertEqual(by_name["rag"]["accuracy_pct"], 10.0)
        self.assertEqual(by_name["graphrag"]["accuracy_pct"], 60.0)
        self.assertEqual(by_name["agentic"]["accuracy_pct"], 92.0)
        self.assertEqual(by_name["rag"]["retrieval_success"], 0.44)
        self.assertEqual(by_name["graphrag"]["retrieval_success"], 0.95)
        self.assertEqual(by_name["agentic"]["retrieval_success"], 0.97)

    def test_qtypes_and_advantage_match_official_summary(self):
        qtypes = [row["qtype"] for row in self.view["by_qtype"]]
        self.assertEqual(
            qtypes,
            ["lookup", "aggregation", "temporal", "multi_hop", "superlative"],
        )
        effect = self.bundle["summary"]["agentic_effectiveness"]
        adv = self.view["advantage"]
        self.assertEqual(adv["agentic_correct_rag_incorrect"]["n"], effect["n_agent_beats_rag"])
        self.assertEqual(
            adv["agentic_correct_graphrag_incorrect"]["n"],
            effect["n_agent_beats_graphrag"],
        )
        self.assertEqual(
            adv["all_three_correct"]["n"],
            effect["n_agent_unnecessary_all_correct"],
        )
        self.assertEqual(adv["all_three_correct"]["n"], 10)
        self.assertEqual(adv["agentic_correct_rag_incorrect"]["n"], 82)
        self.assertEqual(adv["agentic_correct_graphrag_incorrect"]["n"], 35)

    def test_agentic_ollama_generate_split(self):
        behavior = self.view["agentic_behavior"]
        self.assertEqual(behavior["n_with_ollama_generate"], 10)
        self.assertEqual(behavior["n_without_ollama_generate"], 90)
        self.assertEqual(behavior["n"], 100)
        self.assertEqual(behavior["avg_agent_steps"], 7.93)
        self.assertEqual(self.view["failures"]["n_agentic_failures"], 8)
        self.assertEqual(self.view["failures"]["categories"]["success"], 92)

    def test_hidden_files_load_successfully(self):
        self.assertTrue(self.hidden_final.is_file())
        self.assertTrue(self.hidden_summary.is_file())
        self.assertTrue(self.hidden_report.is_file())
        self.assertTrue(self.hidden_jsonl.is_file())
        self.assertEqual(self.hidden_bundle["run_id"], "hidden")
        self.assertEqual(self.hidden_bundle["final"].get("run"), "hidden_benchmark")
        self.assertTrue(any(row["id"] == "hidden" and row["ready"] for row in available_runs()))
        self.assertEqual(hidden_dashboard_view()["run_id"], "hidden")

    def test_hidden_record_counts_and_uniqueness(self):
        records = self.hidden_bundle["records"]
        self.assertEqual(len(records), 150)
        self.assertEqual(self.hidden_view["counts"]["n_questions"], 50)
        self.assertEqual(self.hidden_view["counts"]["n_records"], 150)
        self.assertEqual(self.hidden_view["counts"]["n_rag"], 50)
        self.assertEqual(self.hidden_view["counts"]["n_graphrag"], 50)
        self.assertEqual(self.hidden_view["counts"]["n_agentic"], 50)
        self.assertEqual(len(self.hidden_view["questions"]), 50)
        keys = [(row.get("qid"), row.get("pipeline")) for row in records]
        self.assertEqual(len(keys), 150)
        self.assertEqual(len(set(keys)), 150)
        self.assertEqual(self.hidden_view["n_unique_pairs"], 150)

    def test_hidden_accuracy_is_never_calculated_or_displayed(self):
        def collect(payload):
            keys = set()
            if isinstance(payload, dict):
                for key, value in payload.items():
                    keys.add(str(key).lower())
                    keys.update(collect(value))
            elif isinstance(payload, list):
                for item in payload:
                    keys.update(collect(item))
            return keys

        keys = collect(self.hidden_view)
        for marker in ACCURACY_KEY_MARKERS:
            self.assertTrue(
                all(marker not in key for key in keys),
                "hidden view contained " + marker + ": " + str(sorted(keys)),
            )
        self.assertNotIn("overall", self.hidden_view)
        self.assertNotIn("advantage", self.hidden_view)
        self.assertNotIn("failures", self.hidden_view)
        self.assertIn("cannot be calculated", self.hidden_view["no_gold_note"].lower())
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("Official Public Benchmark", html)
        self.assertIn("Hidden Benchmark", html)
        self.assertIn(
            "Hidden accuracy cannot be calculated because there are no gold answers.",
            html,
        )
        hidden_html_blob = html[html.find("function renderHidden"):]
        self.assertNotIn("accuracy_pct", hidden_html_blob)
        self.assertNotIn("normalized_match", hidden_html_blob)

    def test_html_does_not_present_token_efficiency_as_llm_cost(self):
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("not LLM-cost efficiency", html)
        self.assertIn("context_token_count", html)
        self.assertNotIn("token_efficiency_per_1k_tokens", html)

    def test_artifacts_not_modified(self):
        for path, expected in self.hashes.items():
            self.assertEqual(_sha256(path), expected, str(path))


if __name__ == "__main__":
    unittest.main()
