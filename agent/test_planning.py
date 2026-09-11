"""Planning-reliability tests for the Agentic GraphRAG orchestrator.

Generic questions only. Does not use eval qids or hardcoded gold answers.
Does not modify TigerGraph, RAG, or GraphRAG.
"""

import json
import re
import sys
import unittest

sys.path.append(".")

from agent.orchestrator import (
    _aggregation_discovery_complete,
    aggregation_required_tokens,
    _evidence_matches_date,
    _parse_dates,
    evidence_is_sufficient,
    extract_answer,
    inspect_gaps,
    investigate,
    parse_weight_classes,
    propose_actions,
    question_intent,
    ranked_events,
    weight_class_exact_match,
)
from agent.state import InvestigationState
from agent.tools import search_events


TEMPORAL_Q = (
    "Who won gold in the men's 20 kilometres walk athletics event "
    "at the Summer Olympics held immediately before 2016?"
)
AGG_Q = (
    "How many biathlon events at the 2018 Winter Olympics had more than "
    "73 competitors?"
)
VENUE_Q = (
    "Who won the gold medal in the event held at Richmond Olympic Oval "
    "on 14 February 2010?"
)
VENUE_RANGE_014_Q = (
    "Which badminton event was held at Riocentro on 11–19 August at the "
    "2016 Summer Olympics and who won the gold medal?"
)
VENUE_RANGE_015_Q = (
    "Which cycling event was held at London Velopark on 3 to 4 August at "
    "the 2012 Summer Olympics and who won the gold medal?"
)
VENUE_017_Q = (
    "Who won the gold medal in the event held at Royal Artillery Barracks "
    "on 28 July 2012?"
)
VENUE_005_Q = (
    "Who won the gold medal in the event held at Olympic Weightlifting "
    "Gymnasium on 20 September 1988?"
)
VENUE_028_Q = (
    "Who won the gold medal in the event held at Olympic Aquatic Centre "
    "on August 14, 2004 (heats & final)?"
)
PUB098_VENUE_Q = (
    "Who won the gold medal in the event held at Sydney Convention and "
    "Exhibition Centre on 18 September to 1 October 2000?"
)
LOOKUP_Q = (
    "How many nations competed in Judo at the 2016 Summer Olympics "
    "Women's 57 kg?"
)
SUPERLATIVE_Q = (
    "According to the provided corpus, which sailing event at the "
    "2000 Summer Olympics had the highest number of competitors?"
)
ALPINE_SUPERLATIVE_Q = (
    "According to the provided corpus, which alpine skiing event at the "
    "1988 Winter Olympics had the highest number of competitors?"
)
SHOOTING_AGG_Q = (
    "How many shooting events at the 2004 Summer Olympics had more than "
    "37 competitors?"
)
CANOEING_AGG_Q = (
    "According to the provided corpus, how many canoeing events at the "
    "2012 Summer Olympics had more than 25 competitors?"
)
PUB001_AGG_Q = (
    "According to the provided corpus, how many biathlon events at the "
    "2018 Winter Olympics had more than 73 competitors?"
)
PUB003_AGG_Q = (
    "According to the provided corpus, how many shooting events at the "
    "2004 Summer Olympics had more than 37 competitors?"
)
PUB020_AGG_Q = (
    "According to the provided corpus, how many alpine skiing events at "
    "the 2014 Winter Olympics had more than 63 competitors?"
)
PUB006_JUDO_Q = (
    "Who won the gold medal in the women's 57 kg judo event at the "
    "Summer Olympics held immediately before 2020?"
)
PUB048_TAEKWONDO_Q = (
    "Who won the gold medal in the men's 68 kg taekwondo event at the "
    "Summer Olympics held immediately before 2016?"
)
PUB055_TAEKWONDO_Q = (
    "Who won the gold medal in the men's 80 kg taekwondo event at the "
    "Summer Olympics held immediately before 2016?"
)


def _print_trace(title, result):
    def out(*args):
        text = " ".join(str(a) for a in args)
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))
            sys.stdout.buffer.flush()

    out("\n" + title)
    out("=" * 70)
    out("question:", result["question"])
    out("kind:", result.get("investigation_kind"))
    out("stopping_status:", result["stopping_status"])
    out("stop_reason:", result["stop_reason"])
    out("tools_used:", result["tools_used"])
    out("supporting_evidence:", result.get("supporting_evidence"))
    out("n_evidence:", len((result.get("state") or {}).get("evidence") or []))
    out("n_documents:", len((result.get("state") or {}).get("document_ids") or []))
    out("final_answer:", result.get("final_answer"))
    for step in result.get("trace") or []:
        out("-" * 70)
        out("iteration:", step.get("iteration"))
        out("chosen_action:", step.get("chosen_action"))
        out("tool_arguments:", json.dumps(step.get("tool_arguments") or {}, ensure_ascii=False)[:400])
        out("reason:", step.get("reason"))
        out("planner:", step.get("planner"))
        out("missing_before:", step.get("missing_before"))
        paths = [
            f"{row.get('path')} {row.get('from_id')}->{row.get('to_id')}"
            for row in (step.get("graph_paths") or [])[:6]
        ]
        out("graph_paths:", paths)
        out("new_documents:", step.get("newly_discovered_documents"))


class PlanningIntentTests(unittest.TestCase):
    def test_intent_kinds(self):
        self.assertEqual(question_intent(TEMPORAL_Q)["kind"], "temporal")
        self.assertEqual(question_intent(TEMPORAL_Q)["temporal"], "previous")
        self.assertEqual(question_intent(TEMPORAL_Q)["anchor_year"], 2016)
        self.assertEqual(question_intent(AGG_Q)["kind"], "aggregation")
        self.assertEqual(question_intent(AGG_Q)["aggregation"]["op"], ">")
        self.assertEqual(question_intent(AGG_Q)["aggregation"]["value"], 73)
        self.assertEqual(question_intent(VENUE_Q)["kind"], "venue_date")
        self.assertTrue(question_intent(VENUE_Q)["has_date"])
        self.assertEqual(question_intent(LOOKUP_Q)["kind"], "lookup")
        sailing = question_intent(SUPERLATIVE_Q)
        self.assertEqual(sailing["kind"], "superlative")
        self.assertEqual(sailing["superlative"]["direction"], "highest")
        self.assertEqual(sailing["superlative"]["field"], "competitors")
        self.assertEqual(sailing["event_hint"], "sailing")
        self.assertFalse(sailing["asks_how_many"])
        alpine = question_intent(ALPINE_SUPERLATIVE_Q)
        self.assertEqual(alpine["kind"], "superlative")
        self.assertEqual(alpine["event_hint"], "alpine skiing")
        range_014 = question_intent(VENUE_RANGE_014_Q)
        self.assertEqual(range_014["kind"], "venue_date")
        self.assertTrue(range_014["has_date"])
        self.assertEqual(range_014["dates"][0]["start"]["day"], 11)
        self.assertEqual(range_014["dates"][0]["end"]["day"], 19)
        self.assertEqual(range_014["dates"][0]["start"]["month"], 8)
        self.assertEqual(range_014["dates"][0]["start"]["year"], 2016)
        range_015 = question_intent(VENUE_RANGE_015_Q)
        self.assertTrue(range_015["has_date"])
        self.assertEqual(range_015["dates"][0]["start"]["day"], 3)
        self.assertEqual(range_015["dates"][0]["end"]["day"], 4)
        self.assertEqual(range_015["dates"][0]["start"]["year"], 2012)

    def test_aggregation_sport_phrase_from_how_many_events(self):
        canoeing = question_intent(CANOEING_AGG_Q)
        self.assertEqual(canoeing["kind"], "aggregation")
        self.assertEqual(canoeing["event_hint"], "canoeing")
        self.assertEqual(
            aggregation_required_tokens(canoeing),
            ["canoeing", "2012", "summer"],
        )
        self.assertNotIn("competitors", aggregation_required_tokens(canoeing))
        self.assertNotIn("events", aggregation_required_tokens(canoeing))
        self.assertNotIn("olympics", aggregation_required_tokens(canoeing))

        biathlon = question_intent(PUB001_AGG_Q)
        self.assertEqual(biathlon["event_hint"], "biathlon")
        self.assertEqual(
            aggregation_required_tokens(biathlon),
            ["biathlon", "2018", "winter"],
        )
        shooting = question_intent(PUB003_AGG_Q)
        self.assertEqual(shooting["event_hint"], "shooting")
        self.assertEqual(
            aggregation_required_tokens(shooting),
            ["shooting", "2004", "summer"],
        )
        alpine = question_intent(PUB020_AGG_Q)
        self.assertEqual(alpine["event_hint"], "alpine skiing")
        self.assertEqual(
            aggregation_required_tokens(alpine),
            ["alpine skiing", "2014", "winter"],
        )
        figure = question_intent(
            "How many figure skating events at the 2010 Winter Olympics "
            "had more than 20 competitors?"
        )
        self.assertEqual(figure["event_hint"], "figure skating")
        self.assertEqual(
            aggregation_required_tokens(figure),
            ["figure skating", "2010", "winter"],
        )
        bare = question_intent(
            "How many events at the 2012 Summer Olympics had more than "
            "25 competitors?"
        )
        self.assertIsNone(bare["event_hint"])
        self.assertEqual(
            aggregation_required_tokens(bare),
            ["2012", "summer"],
        )

    def test_weight_class_signed_exact_match(self):
        self.assertEqual(parse_weight_classes("80 kg"), [("", 80)])
        self.assertEqual(parse_weight_classes("Men's +80 kg"), [("+", 80)])
        self.assertEqual(parse_weight_classes("Women's -60 kg"), [("-", 60)])
        self.assertEqual(parse_weight_classes("Men's 80kg"), [("", 80)])
        self.assertEqual(parse_weight_classes("-48 kg"), [("-", 48)])

        self.assertTrue(weight_class_exact_match("80 kg", "Men's 80 kg"))
        self.assertFalse(weight_class_exact_match("80 kg", "Men's +80 kg"))
        self.assertTrue(weight_class_exact_match("+80 kg", "Men's +80 kg"))
        self.assertFalse(weight_class_exact_match("+80 kg", "Men's 80 kg"))
        self.assertTrue(weight_class_exact_match("-60 kg", "Men's -60 kg"))
        self.assertTrue(weight_class_exact_match("-48 kg", "Judo – Women's -48 kg"))
        self.assertFalse(weight_class_exact_match("80 kg", "Men's -80 kg"))

        unsigned_q = PUB055_TAEKWONDO_Q
        plus_q = (
            "Who won the gold medal in the men's +80 kg taekwondo event "
            "at the Summer Olympics held immediately before 2016?"
        )
        pool = [
            {
                "id": "Q1339098",
                "name": "Taekwondo at the 2012 Summer Olympics – Men's +80 kg",
            },
            {
                "id": "Q934172",
                "name": "Taekwondo at the 2012 Summer Olympics – Men's 80 kg",
            },
        ]
        unsigned_ranked = ranked_events(
            InvestigationState.create(unsigned_q), unsigned_q, pool=pool
        )
        self.assertEqual(unsigned_ranked[0]["id"], "Q934172")
        plus_ranked = ranked_events(
            InvestigationState.create(plus_q), plus_q, pool=pool
        )
        self.assertEqual(plus_ranked[0]["id"], "Q1339098")

    def test_date_range_parsing_formats(self):
        en = _parse_dates("11–19 August 2016")
        hyphen = _parse_dates("11-19 August 2016")
        spoken = _parse_dates("3 to 4 August 2012")
        detached = _parse_dates("11–19 August at the 2016 Summer Olympics")
        exact = _parse_dates("14 February 2010")
        mdy = _parse_dates("February 14, 2010")
        for rows in (en, hyphen, detached):
            self.assertEqual(rows[0]["start"]["day"], 11)
            self.assertEqual(rows[0]["end"]["day"], 19)
            self.assertEqual(rows[0]["start"]["year"], 2016)
        self.assertEqual(spoken[0]["start"]["day"], 3)
        self.assertEqual(spoken[0]["end"]["day"], 4)
        self.assertEqual(exact[0]["start"]["day"], 14)
        self.assertEqual(exact[0]["end"]["day"], 14)
        self.assertEqual(exact[0]["start"]["month"], 2)
        self.assertEqual(mdy[0]["start"]["day"], 14)
        self.assertEqual(mdy[0]["start"]["month"], 2)
        self.assertEqual(mdy[0]["start"]["year"], 2010)
        intent = {
            "kind": "venue_date",
            "dates": detached,
            "years": [2016],
        }
        self.assertTrue(_evidence_matches_date(
            "games: 2016 Summer\ndate: 11–19 August\ngold: Carolina Marín",
            intent,
        ))
        self.assertFalse(_evidence_matches_date(
            "games: 2016 Summer\ndate: 11–20 August 2016\ngold: Chen Long",
            intent,
        ))
        self.assertFalse(_evidence_matches_date("gold: Chen Long", intent))

    def test_cross_month_date_range_parsing(self):
        def _ymd(part):
            return (part["year"], part["month"], part["day"])

        same = _parse_dates("11–19 August 2016")
        self.assertEqual(len(same), 1)
        self.assertEqual(_ymd(same[0]["start"]), (2016, 8, 11))
        self.assertEqual(_ymd(same[0]["end"]), (2016, 8, 19))

        spoken = _parse_dates("18 September to 1 October 2000")
        hyphen = _parse_dates("18 September - 1 October 2000")
        en = _parse_dates("18 September–1 October 2000")
        for rows in (spoken, hyphen, en):
            self.assertEqual(len(rows), 1, rows)
            self.assertEqual(_ymd(rows[0]["start"]), (2000, 9, 18))
            self.assertEqual(_ymd(rows[0]["end"]), (2000, 10, 1))

        detached = _parse_dates("11–19 August at the 2016 Summer Olympics")
        self.assertEqual(len(detached), 1)
        self.assertEqual(_ymd(detached[0]["start"]), (2016, 8, 11))
        self.assertEqual(_ymd(detached[0]["end"]), (2016, 8, 19))

        range_intent = {
            "kind": "venue_date",
            "dates": spoken,
            "years": [2000],
        }
        self.assertTrue(_evidence_matches_date(
            "games: 2000 Summer\ndate: 18 September to 1 October 2000\n"
            "gold: Bekzat Sattarkhanov",
            range_intent,
        ))
        self.assertFalse(_evidence_matches_date(
            "games: 2000 Summer\ndate: 18 September 2000\ngold: Isabel Fernández",
            range_intent,
        ))

        inside_intent = {
            "kind": "venue_date",
            "dates": _parse_dates("22 September 2000"),
            "years": [2000],
        }
        self.assertTrue(_evidence_matches_date(
            "games: 2000 Summer\ndate: 18 September to 1 October 2000",
            inside_intent,
        ))
        outside_intent = {
            "kind": "venue_date",
            "dates": _parse_dates("2 October 2000"),
            "years": [2000],
        }
        self.assertFalse(_evidence_matches_date(
            "games: 2000 Summer\ndate: 18 September to 1 October 2000",
            outside_intent,
        ))

    def test_gap_proposals_are_state_driven(self):
        temporal_empty = InvestigationState.create(TEMPORAL_Q)
        self.assertIn("target_games", inspect_gaps(temporal_empty)["gaps"])
        first = propose_actions(temporal_empty)
        self.assertEqual(first[0]["tool"], "search_entities")

        venue_empty = InvestigationState.create(VENUE_Q)
        self.assertEqual(propose_actions(venue_empty)[0]["tool"], "search_entities")
        venue_ready = InvestigationState.create(VENUE_Q)
        venue_ready.identified_entities["Venue"] = [{
            "id": "richmond_olympic_oval",
            "type": "Venue",
            "name": "Richmond Olympic Oval",
        }]
        venue_next = propose_actions(venue_ready)
        self.assertEqual(venue_next[0]["tool"], "traverse_graph")
        self.assertEqual(venue_next[0]["args"]["edge_type"], "HELD_AT")
        self.assertEqual(venue_next[0]["args"]["direction"], "in")

        agg_events = InvestigationState.create(AGG_Q)
        agg_events.identified_entities["Event"] = [
            {"id": "E1", "type": "Event", "name": "Biathlon at the 2018 Winter Olympics – Sprint"},
            {"id": "E2", "type": "Event", "name": "Biathlon at the 2018 Winter Olympics – Pursuit"},
        ]
        gaps = inspect_gaps(agg_events)["gaps"]
        self.assertIn("incomplete_candidate_set", gaps)
        agg_expand = propose_actions(agg_events)
        self.assertEqual(agg_expand[0]["tool"], "search_events")
        self.assertEqual(agg_expand[0]["args"]["top_k"], 0)
        self.assertTrue(agg_expand[0]["args"]["required_tokens"])

        agg_events.actions.append({
            "tool": "search_events",
            "args": {
                "query": "biathlon 2018 winter",
                "top_k": 0,
                "required_tokens": ["biathlon", "2018", "winter"],
            },
            "ok": True,
        })
        gaps = inspect_gaps(agg_events)["gaps"]
        self.assertIn("candidate_event_evidence", gaps)
        agg_next = propose_actions(agg_events)
        self.assertEqual(agg_next[0]["tool"], "retrieve_documents")

        superlative_empty = InvestigationState.create(SUPERLATIVE_Q)
        self.assertIn("incomplete_candidate_set", inspect_gaps(superlative_empty)["gaps"])
        superlative_first = propose_actions(superlative_empty)
        self.assertEqual(superlative_first[0]["tool"], "search_events")
        self.assertEqual(superlative_first[0]["args"]["top_k"], 0)
        self.assertIn("sailing", superlative_first[0]["args"]["required_tokens"])
        self.assertIn("2000", superlative_first[0]["args"]["required_tokens"])


class PlanningScenarioTests(unittest.TestCase):
    def test_temporal_previous_olympics(self):
        result = investigate(TEMPORAL_Q)
        _print_trace("SCENARIO temporal previous Olympics", result)
        self.assertEqual(result["investigation_kind"], "temporal")
        self.assertTrue(result["tools_used"])
        self.assertEqual(result["tools_used"][0], "search_entities")
        event_names = " ".join(
            row.get("name") or ""
            for row in result["state"]["identified_entities"].get("Event") or []
        )
        self.assertIn("2012", event_names)
        self.assertTrue(result["supporting_evidence"], "must collect corpus evidence")
        self.assertTrue(result["state"]["evidence"])
        self.assertIn("Chen Ding", result.get("final_answer") or "")
        self.assertIn(
            result["stop_reason"],
            {"sufficient evidence for an answer", "no useful next action", "maximum iteration limit"},
        )

    def test_aggregation_count_condition(self):
        result = investigate(AGG_Q)
        _print_trace("SCENARIO aggregation how many events", result)
        self.assertEqual(result["investigation_kind"], "aggregation")
        self.assertGreaterEqual(len(result["state"]["evidence"]), 2)
        retrieve_calls = [
            step for step in result["trace"]
            if step.get("chosen_action") == "retrieve_documents"
        ]
        self.assertGreaterEqual(len(retrieve_calls), 2)
        complete_search = []
        for step in result["trace"]:
            if step.get("chosen_action") != "search_events":
                continue
            raw_k = (step.get("tool_arguments") or {}).get("top_k")
            try:
                top_k = int(raw_k)
            except (TypeError, ValueError):
                top_k = 8
            if top_k <= 0:
                complete_search.append(step)
        self.assertTrue(
            complete_search,
            "aggregation must run an untruncated Event search before counting",
        )
        self.assertTrue(result["supporting_evidence"])
        self.assertTrue(re.fullmatch(r"\d+", (result.get("final_answer") or "")))
        self.assertIn(
            result["stop_reason"],
            {"sufficient evidence for an answer", "no useful next action", "maximum iteration limit"},
        )

    def test_venue_and_date(self):
        result = investigate(VENUE_Q)
        _print_trace("SCENARIO venue + date", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        paths = [
            row.get("path")
            for step in result["trace"]
            for row in (step.get("graph_paths") or [])
        ]
        self.assertIn("Venue<-(HELD_AT)-Event", paths)
        self.assertTrue(result["supporting_evidence"])
        blob = " ".join(item.get("text") or "" for item in result["state"]["evidence"]).lower()
        self.assertTrue("february" in blob and "14" in blob)
        answer = result.get("final_answer") or ""
        self.assertIn("Martina", answer)
        self.assertIn(
            result["stop_reason"],
            {"sufficient evidence for an answer", "no useful next action", "maximum iteration limit"},
        )

    def test_venue_date_excel_14_august_2016(self):
        question = (
            "Which athlete won gold at the event held at ExCeL on 14 August 2016?"
        )
        intent = question_intent(question)
        self.assertEqual(intent["kind"], "venue_date")
        self.assertTrue(intent["asks_who"])
        self.assertEqual(intent["dates"][0]["start"]["day"], 14)
        self.assertEqual(intent["dates"][0]["start"]["month"], 8)
        self.assertEqual(intent["dates"][0]["start"]["year"], 2016)
        empty = InvestigationState.create(question)
        first = propose_actions(empty)
        self.assertEqual(first[0]["tool"], "search_entities")

        result = investigate(question)
        _print_trace("SCENARIO venue date ExCeL 14 August 2016", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        venue_ids = {
            row.get("id")
            for row in result["state"]["identified_entities"].get("Venue") or []
        }
        self.assertTrue(
            {vid for vid in venue_ids if str(vid).startswith("excel_")},
            venue_ids,
        )
        self.assertEqual(result["tools_used"][0], "search_entities")
        self.assertIn("traverse_graph", result["tools_used"])
        edges = [
            (step.get("tool_arguments") or {}).get("edge_type")
            for step in result["trace"]
        ]
        self.assertIn("HELD_AT", edges)
        self.assertNotEqual(result["tools_used"][:2], ["search_entities", "search_events"])
        paths = [
            row.get("path")
            for step in result["trace"]
            for row in (step.get("graph_paths") or [])
        ]
        self.assertIn("Venue<-(HELD_AT)-Event", paths)
        answer = result.get("final_answer") or ""
        self.assertNotIn("Zhao Shuai", answer)
        date_matched = [
            item.get("title") or ""
            for item in result["state"]["evidence"]
            if "14" in (item.get("text") or "") and "2016" in (item.get("text") or "")
        ]
        if date_matched:
            self.assertTrue(answer)
            self.assertTrue(
                any("excel" in (item.get("text") or "").lower()
                    for item in result["state"]["evidence"]),
            )

    def test_venue_date_does_not_stop_on_first_gold(self):
        intent = question_intent(VENUE_RANGE_014_Q)
        self.assertEqual(intent["kind"], "venue_date")
        state = InvestigationState.create(VENUE_RANGE_014_Q)
        state.identified_entities["Venue"] = [{
            "id": "riocentro_pavilion_4",
            "type": "Venue",
            "name": "Riocentro - Pavilion 4",
        }]
        state.identified_entities["Event"] = [
            {
                "id": "Q25301472",
                "type": "Event",
                "name": "Badminton at the 2016 Summer Olympics – Men's singles",
            },
            {
                "id": "Q25301483",
                "type": "Event",
                "name": "Badminton at the 2016 Summer Olympics – Women's singles",
            },
        ]
        state.evidence.append({
            "doc_id": "Q25301472",
            "title": "Badminton at the 2016 Summer Olympics – Men's singles",
            "text": "date: 11–20 August 2016\ngold: Chen Long",
        })
        state.document_ids.append("Q25301472")
        self.assertFalse(
            evidence_is_sufficient(state, intent),
            "first gold: must not stop venue+date while other events remain",
        )
        nxt = propose_actions(state)
        self.assertEqual(nxt[0]["tool"], "retrieve_documents")
        self.assertEqual(nxt[0]["args"]["event_id"], "Q25301483")

    def test_venue_date_range_pub014(self):
        result = investigate(VENUE_RANGE_014_Q)
        _print_trace("SCENARIO venue date range pub-014", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        answer = result.get("final_answer") or ""
        self.assertIn("Carolina Mar", answer)
        self.assertNotIn("Chen Long", answer)

    def test_venue_date_range_pub015(self):
        result = investigate(VENUE_RANGE_015_Q)
        _print_trace("SCENARIO venue date range pub-015", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        answer = result.get("final_answer") or ""
        self.assertIn("Dani King", answer)
        self.assertIn("Laura Trott", answer)
        self.assertIn("Joanna Rowsell", answer)
        self.assertNotIn("trombergs", answer.lower())

    def test_venue_date_pub017(self):
        result = investigate(VENUE_017_Q)
        _print_trace("SCENARIO venue date pub-017", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        self.assertIn("Yi Siling", result.get("final_answer") or "")

    def test_venue_date_pub005(self):
        result = investigate(VENUE_005_Q)
        _print_trace("SCENARIO venue date pub-005", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        self.assertIn("Süleymanoğlu", result.get("final_answer") or "")

    def test_venue_date_pub028(self):
        result = investigate(VENUE_028_Q)
        _print_trace("SCENARIO venue date pub-028", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        self.assertIn("Michael Phelps", result.get("final_answer") or "")

    def test_venue_date_pub098_cross_month_range(self):
        parsed = _parse_dates(PUB098_VENUE_Q)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["start"]["day"], 18)
        self.assertEqual(parsed[0]["start"]["month"], 9)
        self.assertEqual(parsed[0]["end"]["day"], 1)
        self.assertEqual(parsed[0]["end"]["month"], 10)
        self.assertEqual(parsed[0]["start"]["year"], 2000)
        self.assertEqual(parsed[0]["end"]["year"], 2000)
        result = investigate(PUB098_VENUE_Q)
        _print_trace("SCENARIO venue date pub-098", result)
        self.assertEqual(result["investigation_kind"], "venue_date")
        answer = result.get("final_answer") or ""
        self.assertIn("Bekzat Sattarkhanov", answer)
        self.assertNotIn("Isabel Fernández", answer)
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")

    def test_aggregation_partial_subset_is_not_complete(self):
        intent = question_intent(SHOOTING_AGG_Q)
        self.assertEqual(intent["kind"], "aggregation")
        state = InvestigationState.create(SHOOTING_AGG_Q)
        partial = [
            {"id": "E1", "type": "Event", "name": "Shooting at the 2004 Summer Olympics – Air pistol"},
            {"id": "E2", "type": "Event", "name": "Shooting at the 2004 Summer Olympics – Rifle"},
            {"id": "E3", "type": "Event", "name": "Shooting at the 2004 Summer Olympics – Trap"},
        ]
        state.identified_entities["Event"] = partial
        for row in partial:
            state.document_ids.append(row["id"])
            state.evidence.append({
                "doc_id": row["id"],
                "title": row["name"],
                "text": "competitors: 44",
            })
        self.assertFalse(
            _aggregation_discovery_complete(state, intent),
            "a truncated event list is not a complete aggregation universe",
        )
        self.assertFalse(
            evidence_is_sufficient(state, intent),
            "evidence for a partial candidate set must not stop aggregation",
        )
        self.assertIn("incomplete_candidate_set", inspect_gaps(state, intent)["gaps"])
        nxt = propose_actions(state)
        self.assertEqual(nxt[0]["tool"], "search_events")
        self.assertEqual(nxt[0]["args"]["top_k"], 0)
        self.assertIn("shooting", nxt[0]["args"]["required_tokens"])
        self.assertIn("2004", nxt[0]["args"]["required_tokens"])

    def test_aggregation_stops_only_after_complete_set(self):
        intent = question_intent(SHOOTING_AGG_Q)
        state = InvestigationState.create(SHOOTING_AGG_Q)
        events = [
            {
                "id": "E1",
                "type": "Event",
                "name": "Shooting at the 2004 Summer Olympics – Air pistol",
            },
            {
                "id": "E2",
                "type": "Event",
                "name": "Shooting at the 2004 Summer Olympics – Rifle",
            },
        ]
        state.identified_entities["Event"] = events
        state.actions.append({
            "tool": "search_events",
            "args": {
                "query": "shooting 2004 summer",
                "top_k": 0,
                "required_tokens": ["shooting", "2004", "summer"],
            },
            "ok": True,
        })
        state.evidence.append({
            "doc_id": "E1",
            "title": events[0]["name"],
            "text": "competitors: 44",
        })
        state.document_ids.append("E1")
        self.assertTrue(_aggregation_discovery_complete(state, intent))
        self.assertFalse(
            evidence_is_sufficient(state, intent),
            "must keep collecting until every scoped event has evidence",
        )
        self.assertEqual(propose_actions(state)[0]["tool"], "retrieve_documents")
        self.assertEqual(propose_actions(state)[0]["args"]["event_id"], "E2")

        state.evidence.append({
            "doc_id": "E2",
            "title": events[1]["name"],
            "text": "competitors: 20",
        })
        state.document_ids.append("E2")
        self.assertTrue(evidence_is_sufficient(state, intent))
        self.assertFalse(propose_actions(state) or evidence_is_sufficient(state, intent) is False)

    def test_aggregation_discovers_full_scoped_universe(self):
        universe = search_events(
            "shooting 2004 summer",
            top_k=0,
            required_tokens=["shooting", "2004", "summer"],
        )
        truncated = search_events(
            "shooting 2004 summer",
            top_k=7,
            required_tokens=["shooting", "2004", "summer"],
        )
        self.assertTrue(universe["ok"], universe.get("error"))
        complete_ids = {row["id"] for row in universe["events"]}
        truncated_ids = {row["id"] for row in truncated["events"]}
        self.assertGreater(
            len(complete_ids),
            len(truncated_ids),
            "untruncated sport/year Event search must expand beyond a ranked subset",
        )

        result = investigate(SHOOTING_AGG_Q)
        _print_trace("SCENARIO aggregation complete candidate set", result)
        retrieved = set(result["state"]["document_ids"])
        self.assertTrue(result["supporting_evidence"])
        self.assertEqual(
            complete_ids,
            retrieved,
            "aggregation must retrieve evidence for every scoped Event, not a truncated subset",
        )
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")
        self.assertTrue(re.fullmatch(r"\d+", result.get("final_answer") or ""))
        self.assertTrue(result["sufficient_evidence"])

    def test_simple_lookup(self):
        result = investigate(LOOKUP_Q)
        _print_trace("SCENARIO simple lookup", result)
        self.assertEqual(result["investigation_kind"], "lookup")
        self.assertTrue(result["supporting_evidence"])
        self.assertTrue(result["state"]["evidence"])
        self.assertTrue(re.fullmatch(r"\d+", (result.get("final_answer") or "")))
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")


    def test_superlative_returns_event_name_not_count(self):
        intent = question_intent(SUPERLATIVE_Q)
        self.assertEqual(intent["kind"], "superlative")
        state = InvestigationState.create(SUPERLATIVE_Q)
        events = [
            {
                "id": "E1",
                "type": "Event",
                "name": "Sailing at the 2000 Summer Olympics – Soling",
            },
            {
                "id": "E2",
                "type": "Event",
                "name": "Sailing at the 2000 Summer Olympics – Laser",
            },
        ]
        state.identified_entities["Event"] = events
        state.actions.append({
            "tool": "search_events",
            "args": {
                "query": "sailing 2000 summer",
                "top_k": 0,
                "required_tokens": ["sailing", "2000", "summer"],
            },
            "ok": True,
        })
        state.evidence.append({
            "doc_id": "E1",
            "title": events[0]["name"],
            "text": "competitors: 48",
        })
        state.document_ids.append("E1")
        self.assertFalse(evidence_is_sufficient(state, intent))
        self.assertEqual(propose_actions(state)[0]["tool"], "retrieve_documents")
        state.evidence.append({
            "doc_id": "E2",
            "title": events[1]["name"],
            "text": "competitors: 20",
        })
        state.document_ids.append("E2")
        self.assertTrue(evidence_is_sufficient(state, intent))
        answer = extract_answer(state, intent)
        self.assertEqual(answer, events[0]["name"])
        self.assertFalse(re.fullmatch(r"\d+", answer or ""))

    def test_superlative_highest_competitors(self):
        result = investigate(SUPERLATIVE_Q)
        _print_trace("SCENARIO superlative highest competitors", result)
        self.assertEqual(result["investigation_kind"], "superlative")
        complete_search = []
        for step in result["trace"]:
            if step.get("chosen_action") != "search_events":
                continue
            raw_k = (step.get("tool_arguments") or {}).get("top_k")
            try:
                top_k = int(raw_k)
            except (TypeError, ValueError):
                top_k = 8
            if top_k <= 0:
                complete_search.append(step)
        self.assertTrue(
            complete_search,
            "superlative must run an untruncated Event search before selecting",
        )
        self.assertGreaterEqual(len(result["state"]["evidence"]), 2)
        self.assertTrue(result["supporting_evidence"])
        answer = result.get("final_answer") or ""
        self.assertTrue(answer)
        self.assertFalse(re.fullmatch(r"\d+", answer), answer)
        self.assertIn("sailing", answer.lower())
        self.assertIn("2000", answer)
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")

    def test_superlative_unsported_2004_highest_competitors(self):
        question = (
            "Which event had the highest number of competitors at the "
            "2004 Summer Olympics?"
        )
        intent = question_intent(question)
        self.assertEqual(intent["kind"], "superlative")
        self.assertEqual(intent["superlative"]["direction"], "highest")
        self.assertEqual(intent["superlative"]["field"], "competitors")
        self.assertIsNone(intent["event_hint"])
        self.assertEqual(intent["years"], [2004])
        self.assertEqual(intent["season"], "summer")
        empty = InvestigationState.create(question)
        first = propose_actions(empty)
        self.assertEqual(first[0]["tool"], "search_events")
        self.assertEqual(first[0]["args"]["top_k"], 0)
        self.assertEqual(first[0]["args"]["required_tokens"], ["2004", "summer"])

        result = investigate(question)
        _print_trace("SCENARIO unsported superlative 2004", result)
        self.assertEqual(result["investigation_kind"], "superlative")
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")
        answer = result.get("final_answer") or ""
        self.assertIn("Cycling", answer)
        self.assertIn("road race", answer.lower())
        self.assertIn("2004", answer)
        self.assertFalse(re.fullmatch(r"\d+", answer), answer)
        self.assertNotEqual(answer, "Insufficient evidence")
        self.assertLess(result["state"]["iteration_count"], 48)
        retrieve_calls = [
            step for step in result["trace"]
            if step.get("chosen_action") == "retrieve_documents"
        ]
        self.assertEqual(len(retrieve_calls), 1)
        batch = retrieve_calls[0].get("tool_arguments") or {}
        self.assertGreater(len(batch.get("doc_ids") or []), 1)
        self.assertGreaterEqual(len(result["state"]["evidence"]), 2)
        values = []
        for item in result["state"]["evidence"]:
            match = re.search(
                r"competitors:\s*(\d+)", (item.get("text") or ""), re.I
            )
            if match:
                values.append((int(match.group(1)), item.get("title") or ""))
        self.assertTrue(values)
        winner = max(values, key=lambda row: row[0])
        self.assertIn("road race", winner[1].lower())

    def _assert_aggregation_live(self, question, expected_answer, sport_token):
        intent = question_intent(question)
        self.assertEqual(intent["kind"], "aggregation")
        self.assertEqual(intent["event_hint"], sport_token)
        tokens = aggregation_required_tokens(intent)
        self.assertEqual(tokens[0], sport_token)
        empty = InvestigationState.create(question)
        first = propose_actions(empty)
        self.assertEqual(first[0]["tool"], "search_events")
        self.assertEqual(first[0]["args"]["top_k"], 0)
        self.assertEqual(first[0]["args"]["required_tokens"][0], sport_token)
        result = investigate(question)
        _print_trace("SCENARIO public aggregation " + sport_token, result)
        self.assertEqual(result["investigation_kind"], "aggregation")
        self.assertEqual(result.get("final_answer"), expected_answer)
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")
        complete_search = []
        for step in result["trace"]:
            if step.get("chosen_action") != "search_events":
                continue
            raw_k = (step.get("tool_arguments") or {}).get("top_k")
            try:
                top_k = int(raw_k)
            except (TypeError, ValueError):
                top_k = 8
            if top_k <= 0:
                complete_search.append(step)
        self.assertTrue(complete_search)
        required = (complete_search[0].get("tool_arguments") or {}).get("required_tokens") or []
        self.assertEqual(required[0], sport_token)

    def test_public_aggregation_pub051_canoeing(self):
        self._assert_aggregation_live(CANOEING_AGG_Q, "6", "canoeing")

    def test_public_aggregation_pub001_biathlon(self):
        self._assert_aggregation_live(PUB001_AGG_Q, "5", "biathlon")

    def test_public_aggregation_pub003_shooting(self):
        self._assert_aggregation_live(PUB003_AGG_Q, "8", "shooting")

    def test_public_aggregation_pub020_alpine_skiing(self):
        self._assert_aggregation_live(PUB020_AGG_Q, "4", "alpine skiing")

    def _assert_weight_class_live(self, question, expected_answer):
        intent = question_intent(question)
        self.assertEqual(intent["kind"], "temporal")
        result = investigate(question)
        _print_trace("SCENARIO public weight-class " + expected_answer, result)
        self.assertEqual(result["investigation_kind"], "temporal")
        self.assertEqual(result.get("final_answer"), expected_answer)
        self.assertEqual(result["stop_reason"], "sufficient evidence for an answer")
        titles = " ".join(
            item.get("title") or ""
            for item in (result.get("state") or {}).get("evidence") or []
        )
        classes = parse_weight_classes(question)
        if classes:
            evidence_classes = parse_weight_classes(titles)
            self.assertTrue(
                any(item in evidence_classes for item in classes),
                titles,
            )

    def test_public_weight_class_pub055_taekwondo_80kg(self):
        self.assertTrue(
            weight_class_exact_match(PUB055_TAEKWONDO_Q, "Men's 80 kg")
        )
        self.assertFalse(
            weight_class_exact_match(PUB055_TAEKWONDO_Q, "Men's +80 kg")
        )
        self._assert_weight_class_live(PUB055_TAEKWONDO_Q, "Sebastián Crismanich")

    def test_public_weight_class_pub048_taekwondo_68kg(self):
        self._assert_weight_class_live(PUB048_TAEKWONDO_Q, "Servet Tazegül")

    def test_public_weight_class_pub006_judo_57kg(self):
        self._assert_weight_class_live(PUB006_JUDO_Q, "Rafaela Silva")

    def test_country_hint_from_participant_questions(self):
        shooting = question_intent("who participated from india in shooting")
        self.assertEqual(shooting["kind"], "who_won")
        self.assertTrue(shooting["asks_who"])
        self.assertEqual(shooting["country_hint"], "india")
        self.assertEqual(shooting["event_hint"], "shooting")
        self.assertFalse(shooting["asks_country"])

        athletes = question_intent(
            "which athletes from India participated in shooting"
        )
        self.assertTrue(athletes["asks_who"])
        self.assertEqual(athletes["country_hint"], "india")
        self.assertEqual(athletes["event_hint"], "shooting")

        represented = question_intent("who represented India in shooting")
        self.assertTrue(represented["asks_who"])
        self.assertEqual(represented["country_hint"], "india")

        plain = question_intent(
            "Who won gold in the women's 57 kg judo event at the 2016 Summer Olympics?"
        )
        self.assertTrue(plain["asks_who"])
        self.assertIsNone(plain["country_hint"])

    def test_country_filter_proposes_won_by_then_represents(self):
        question = "who participated from india in shooting"
        state = InvestigationState.create(question)
        first = propose_actions(state)
        self.assertEqual(first[0]["tool"], "search_events")
        self.assertEqual(first[0]["args"].get("required_tokens"), ["shooting"])
        self.assertGreaterEqual(int(first[0]["args"]["top_k"]), 12)

        state.identified_entities["Event"] = [{
            "id": "Q1005423",
            "type": "Event",
            "name": "Shooting at the 2008 Summer Olympics – Men's 25 metre rapid fire pistol",
        }]
        next_actions = propose_actions(state)
        self.assertEqual(next_actions[0]["tool"], "traverse_graph")
        self.assertEqual(next_actions[0]["args"]["edge_type"], "WON_BY")
        self.assertNotEqual(next_actions[0]["tool"], "retrieve_documents")

        state.identified_entities["Athlete"] = [{
            "id": "oleksandr_petriv",
            "type": "Athlete",
            "name": "Oleksandr Petriv",
        }]
        country_actions = propose_actions(state)
        self.assertEqual(country_actions[0]["tool"], "traverse_graph")
        self.assertEqual(country_actions[0]["args"]["edge_type"], "REPRESENTS")
        self.assertEqual(country_actions[0]["args"]["src_id"], "oleksandr_petriv")

    def _assert_india_shooting_live(self, question):
        intent = question_intent(question)
        self.assertEqual(intent["country_hint"], "india")
        result = investigate(question)
        _print_trace("SCENARIO country filter " + question, result)
        answer = result.get("final_answer") or ""
        self.assertIn("Abhinav Bindra", answer)
        self.assertNotIn("Oleksandr Petriv", answer)
        self.assertIn("traverse_graph", result["tools_used"])
        self.assertIn(
            "REPRESENTS",
            " ".join(
                (step.get("tool_arguments") or {}).get("edge_type") or ""
                for step in result["trace"]
            ),
        )
        paths = " ".join(
            row.get("path") or ""
            for row in (result.get("state") or {}).get("graph_paths") or []
        )
        self.assertIn("Athlete-(REPRESENTS)->Country", paths)

    def test_india_shooting_who_participated(self):
        self._assert_india_shooting_live(
            "who participated from india in shooting"
        )

    def test_india_shooting_which_athletes(self):
        self._assert_india_shooting_live(
            "which athletes from India participated in shooting"
        )

    def test_india_shooting_who_represented(self):
        self._assert_india_shooting_live(
            "who represented India in shooting"
        )

    def test_participant_question_without_country_constraint(self):
        question = (
            "Who won gold in the women's 57 kg judo event at the "
            "2016 Summer Olympics?"
        )
        intent = question_intent(question)
        self.assertIsNone(intent["country_hint"])
        result = investigate(question)
        _print_trace("SCENARIO participant without country", result)
        self.assertIn("Silva", result.get("final_answer") or "")
        self.assertNotIn("Bindra", result.get("final_answer") or "")


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("PLANNING TEST SUMMARY")
    print("=" * 70)
    print("tests_run:", result.testsRun)
    print("failures:", len(result.failures))
    print("errors:", len(result.errors))
    print("ok:", result.wasSuccessful())
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
