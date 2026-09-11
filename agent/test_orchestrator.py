"""Integration test for the state-driven Agentic GraphRAG orchestrator.

Uses a generic question, not an evaluation qid.
Does not modify TigerGraph, RAG, or GraphRAG.
"""

import json
import sys
import unittest

sys.path.append(".")

from agent.orchestrator import inspect_gaps, investigate, propose_actions
from agent.state import STATUS_ANSWERED, STATUS_RUNNING, InvestigationState


GENERIC_QUESTION = (
    "Who won gold in the women's 57 kg judo event at the 2016 Summer Olympics?"
)


class OrchestratorTests(unittest.TestCase):
    def test_starts_empty_and_proposes_from_gaps(self):
        empty = InvestigationState.create(GENERIC_QUESTION)
        self.assertEqual(empty.iteration_count, 0)
        self.assertEqual(empty.stopping_status, STATUS_RUNNING)
        self.assertEqual(empty.document_ids, [])
        self.assertEqual(empty.evidence, [])
        self.assertEqual(empty.graph_paths, [])
        self.assertEqual(empty.actions, [])
        self.assertFalse(empty.identified_entities["Event"])
        self.assertFalse(empty.identified_entities["Athlete"])

        gaps = inspect_gaps(empty)
        self.assertIn("events", gaps["gaps"])
        self.assertIn("athletes", gaps["gaps"])
        first = propose_actions(empty)
        self.assertTrue(first)
        self.assertIn(first[0]["tool"], {"search_events", "search_entities"})

        venue_state = InvestigationState.create(
            "Who won the event held at a named venue on a given date?"
        )
        venue_state.identified_entities["Venue"] = [{
            "id": "example_venue",
            "type": "Venue",
            "name": "Example Venue",
        }]
        venue_actions = propose_actions(venue_state)
        self.assertTrue(venue_actions)
        self.assertEqual(venue_actions[0]["tool"], "traverse_graph")
        self.assertEqual(venue_actions[0]["args"]["direction"], "in")
        self.assertEqual(venue_actions[0]["args"]["edge_type"], "HELD_AT")

    def test_investigate_accumulates_and_stops(self):
        result = investigate(GENERIC_QUESTION, max_iterations=8)
        print("\nORCHESTRATOR INVESTIGATION TRACE")
        print("=" * 70)
        print("question:", GENERIC_QUESTION)
        print("stopping_status:", result["stopping_status"])
        print("stop_reason:", result["stop_reason"])
        print("final_answer:", result["final_answer"])
        print("sufficient_evidence:", result["sufficient_evidence"])
        print("tools_used:", result["tools_used"])
        print("iterations:", result["state"]["iteration_count"])
        for step in result["trace"]:
            print("-" * 70)
            print("iteration:", step["iteration"])
            print("chosen_action:", step["chosen_action"])
            print("tool_arguments:", json.dumps(step["tool_arguments"], ensure_ascii=False))
            print("reason:", step["reason"])
            print("planner:", step.get("planner"))
            print("missing_before:", step.get("missing_before"))
            print(
                "new_entities:",
                [
                    f"{row.get('type')}:{row.get('id')}|{row.get('name')}"
                    for row in step.get("newly_discovered_entities") or []
                ][:8],
            )
            print("new_documents:", step.get("newly_discovered_documents"))
            print(
                "graph_paths:",
                [
                    f"{row.get('path')} {row.get('from_id')}->{row.get('to_id')}"
                    for row in step.get("graph_paths") or []
                ][:8],
            )
            compact = step.get("tool_result") or {}
            print(
                "tool_result:",
                {
                    "ok": compact.get("ok"),
                    "n_entities": compact.get("n_entities"),
                    "n_events": compact.get("n_events"),
                    "n_documents": compact.get("n_documents"),
                    "n_evidence": compact.get("n_evidence"),
                    "error": compact.get("error"),
                },
            )

        self.assertEqual(result["initial_empty"]["iteration_count"], 0)
        self.assertEqual(result["initial_empty"]["actions"], [])
        self.assertEqual(result["initial_empty"]["document_ids"], [])
        self.assertFalse(result["initial_empty"]["identified_entities"]["Event"])

        tools = result["tools_used"]
        self.assertTrue(tools)
        self.assertGreaterEqual(result["state"]["iteration_count"], 2)
        self.assertEqual(len(result["trace"]), result["state"]["iteration_count"])
        self.assertNotEqual(tools, ["search_entities", "traverse_graph", "retrieve_documents"])

        first_tool = tools[0]
        self.assertIn(first_tool, {"search_events", "search_entities", "traverse_graph"})

        later = tools[1:]
        self.assertTrue(later)
        event_ids = []
        used_discovered_id = False
        for step in result["trace"]:
            for row in step.get("newly_discovered_entities") or []:
                if row.get("type") == "Event" and row.get("id"):
                    event_ids.append(row["id"])
            args = step.get("tool_arguments") or {}
            src_id = args.get("src_id")
            event_id = args.get("event_id")
            if event_ids and (src_id in event_ids or event_id in event_ids):
                used_discovered_id = True
        self.assertTrue(
            used_discovered_id,
            "later tool calls should use an Event id discovered earlier",
        )
        self.assertTrue(
            len(set(tools)) >= 2,
            "orchestrator should choose more than one tool based on state",
        )

        counts = []
        for step in result["trace"]:
            counts.append(step.get("state_counts") or {})
        self.assertGreater(counts[-1].get("events", 0), 0)
        self.assertGreater(
            counts[-1].get("athletes", 0) + counts[-1].get("evidence", 0),
            0,
        )
        self.assertGreater(result["state"]["iteration_count"], 0)
        self.assertTrue(result["state"]["identified_entities"]["Event"])
        self.assertNotEqual(result["stopping_status"], STATUS_RUNNING)
        self.assertIn(
            result["stop_reason"],
            {
                "sufficient evidence for an answer",
                "no useful next action",
                "maximum iteration limit",
            },
        )
        self.assertTrue(result["sufficient_evidence"] or result["state"]["evidence"] or result["state"]["identified_entities"]["Athlete"])
        self.assertEqual(result["stopping_status"], STATUS_ANSWERED)
        self.assertTrue(result["final_answer"])

        required = {
            "iteration",
            "chosen_action",
            "tool_arguments",
            "tool_result",
            "reason",
            "newly_discovered_entities",
            "newly_discovered_documents",
            "graph_paths",
        }
        for step in result["trace"]:
            self.assertTrue(required.issubset(step.keys()))


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("ORCHESTRATOR TEST SUMMARY")
    print("=" * 70)
    print("tests_run:", result.testsRun)
    print("failures:", len(result.failures))
    print("errors:", len(result.errors))
    print("ok:", result.wasSuccessful())
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
