"""Unit/integration tests for investigation state and read-only tools.

Does not modify TigerGraph, RAG, or GraphRAG. Does not use eval question IDs.
"""

import sys
import unittest

sys.path.append(".")

from agent.state import STATUS_ANSWERED, STATUS_RUNNING, InvestigationState
from agent.tools import (
    TOOL_RESULT_FIELDS,
    lookup_evidence,
    retrieve_documents,
    search_entities,
    search_events,
    traverse_graph,
)


def _assert_tool_shape(test, result, tool_name):
    test.assertIsInstance(result, dict)
    test.assertEqual(tuple(result.keys()), TOOL_RESULT_FIELDS)
    test.assertEqual(result["tool"], tool_name)
    test.assertIsInstance(result["args"], dict)
    test.assertIsInstance(result["data"], dict)
    test.assertIsInstance(result["entities"], list)
    test.assertIsInstance(result["events"], list)
    test.assertIsInstance(result["document_ids"], list)
    test.assertIsInstance(result["evidence"], list)
    test.assertIsInstance(result["graph_paths"], list)
    test.assertIsInstance(result["ok"], bool)
    if result["ok"]:
        test.assertIsNone(result["error"])
    else:
        test.assertTrue(result["error"])


class InvestigationStateTests(unittest.TestCase):
    def test_create_and_update(self):
        state = InvestigationState.create(
            "Who won judo at the 2016 Summer Olympics?",
            unresolved=["need event", "need athlete"],
        )
        self.assertEqual(state.original_question, "Who won judo at the 2016 Summer Olympics?")
        self.assertEqual(state.iteration_count, 0)
        self.assertEqual(state.stopping_status, STATUS_RUNNING)
        self.assertIsNone(state.final_answer)
        self.assertEqual(state.unresolved_questions, ["need event", "need athlete"])
        self.assertEqual(state.identified_entities["Event"], [])

        fake = {
            "ok": True,
            "tool": "search_events",
            "args": {"query": "judo 2016"},
            "data": {},
            "entities": [{"id": "E1", "type": "Event", "name": "Judo example"}],
            "events": [{"id": "E1", "type": "Event", "name": "Judo example"}],
            "document_ids": ["D1"],
            "evidence": [{"doc_id": "D1", "title": "Example", "text": "gold: A"}],
            "graph_paths": [{
                "path": "Event-(WON_BY)->Athlete",
                "from_id": "E1",
                "to_id": "A1",
            }],
            "error": None,
        }
        state.ingest_tool_result(fake)
        state.resolve_question("need event")
        state.add_unresolved("confirm gold medalist")
        state.stop(STATUS_ANSWERED, answer="example athlete")

        self.assertEqual(state.iteration_count, 1)
        self.assertEqual(state.actions[0]["tool"], "search_events")
        self.assertEqual(state.document_ids, ["D1"])
        self.assertEqual(state.identified_entities["Event"][0]["id"], "E1")
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(len(state.graph_paths), 1)
        self.assertEqual(state.unresolved_questions, ["need athlete", "confirm gold medalist"])
        self.assertEqual(state.final_answer, "example athlete")
        self.assertEqual(state.stopping_status, STATUS_ANSWERED)
        dumped = state.to_dict()
        self.assertEqual(dumped["original_question"], state.original_question)
        self.assertEqual(dumped["iteration_count"], 1)


class ToolIntegrationTests(unittest.TestCase):
    def test_each_tool_returns_structured_result(self):
        question = "judo at the 2016 Summer Olympics"
        entities = search_entities(question)
        _assert_tool_shape(self, entities, "search_entities")
        self.assertTrue(entities["ok"], entities.get("error"))
        self.assertTrue(entities["entities"])
        types = {row["type"] for row in entities["entities"]}
        self.assertTrue({"OlympicGames", "Sport"} <= types)

        events = search_events(question, top_k=3)
        _assert_tool_shape(self, events, "search_events")
        self.assertTrue(events["ok"], events.get("error"))
        self.assertTrue(events["events"])
        event_id = events["events"][0]["id"]

        won = traverse_graph("Event", event_id, "WON_BY", direction="out")
        _assert_tool_shape(self, won, "traverse_graph")
        self.assertTrue(won["ok"], won.get("error"))
        self.assertTrue(won["graph_paths"])
        self.assertEqual(won["graph_paths"][0]["path"], "Event-(WON_BY)->Athlete")

        held = traverse_graph("Event", event_id, "HELD_AT", direction="out")
        _assert_tool_shape(self, held, "traverse_graph")
        self.assertTrue(held["ok"], held.get("error"))
        venue_id = None
        if held["entities"]:
            venue_id = held["entities"][0]["id"]
            reverse_held = traverse_graph(
                "Venue", venue_id, "HELD_AT", direction="in", limit=5
            )
            _assert_tool_shape(self, reverse_held, "traverse_graph")
            self.assertTrue(reverse_held["ok"], reverse_held.get("error"))
            self.assertTrue(reverse_held["events"])
            self.assertTrue(
                any(
                    row["path"] == "Venue<-(HELD_AT)-Event"
                    for row in reverse_held["graph_paths"]
                )
            )

        docs = retrieve_documents(event_id=event_id, limit=2)
        _assert_tool_shape(self, docs, "retrieve_documents")
        self.assertTrue(docs["ok"], docs.get("error"))
        self.assertTrue(docs["document_ids"])
        self.assertTrue(docs["evidence"])

        evidence = lookup_evidence(doc_id=docs["document_ids"][0])
        _assert_tool_shape(self, evidence, "lookup_evidence")
        self.assertTrue(evidence["ok"], evidence.get("error"))
        self.assertEqual(evidence["document_ids"][0], docs["document_ids"][0])
        self.assertTrue(evidence["evidence"][0]["text"])

        state = InvestigationState.create(question)
        for result in (entities, events, won, held, docs, evidence):
            state.ingest_tool_result(result)
        self.assertGreaterEqual(state.iteration_count, 5)
        self.assertTrue(state.identified_entities["Sport"] or state.identified_entities["Event"])
        self.assertTrue(state.document_ids)
        self.assertTrue(state.evidence)
        self.assertTrue(state.graph_paths)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("AGENT FOUNDATION TEST SUMMARY")
    print("=" * 70)
    print("tests_run:", result.testsRun)
    print("failures:", len(result.failures))
    print("errors:", len(result.errors))
    print("ok:", result.wasSuccessful())
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
