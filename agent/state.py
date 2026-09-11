"""Investigation state for Agentic GraphRAG.

Read-only working memory. No graph writes. No orchestrator.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional


STATUS_RUNNING = "running"
STATUS_ANSWERED = "answered"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

ENTITY_TYPES = (
    "Document",
    "Event",
    "OlympicGames",
    "Athlete",
    "Sport",
    "Venue",
    "Country",
)


def _empty_entities():
    return {vtype: [] for vtype in ENTITY_TYPES}


def _entity_key(row):
    return (row.get("type"), row.get("id"))


def _merge_entity_list(existing, incoming):
    seen = {_entity_key(row) for row in existing}
    merged = list(existing)
    for row in incoming or []:
        key = _entity_key(row)
        if not row.get("id") or key in seen:
            continue
        seen.add(key)
        merged.append({
            "id": row.get("id"),
            "type": row.get("type"),
            "name": row.get("name"),
        })
    return merged


@dataclass
class InvestigationState:
    original_question: str
    identified_entities: dict = field(default_factory=_empty_entities)
    evidence: list = field(default_factory=list)
    document_ids: list = field(default_factory=list)
    graph_paths: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    unresolved_questions: list = field(default_factory=list)
    iteration_count: int = 0
    final_answer: Optional[str] = None
    stopping_status: str = STATUS_RUNNING

    @classmethod
    def create(cls, question, unresolved=None):
        state = cls(original_question=question)
        if unresolved:
            state.unresolved_questions = list(unresolved)
        return state

    def add_unresolved(self, question):
        if question and question not in self.unresolved_questions:
            self.unresolved_questions.append(question)

    def resolve_question(self, question):
        self.unresolved_questions = [
            item for item in self.unresolved_questions if item != question
        ]

    def record_action(self, tool, args, result):
        self.actions.append({
            "iteration": self.iteration_count + 1,
            "tool": tool,
            "args": dict(args or {}),
            "ok": bool((result or {}).get("ok")),
            "error": (result or {}).get("error"),
        })
        self.iteration_count += 1

    def ingest_tool_result(self, result):
        """Merge a structured tool result into this state and log the call."""
        result = result or {}
        self.record_action(result.get("tool"), result.get("args"), result)

        for row in result.get("entities") or []:
            vtype = row.get("type")
            if vtype not in self.identified_entities:
                self.identified_entities[vtype] = []
            self.identified_entities[vtype] = _merge_entity_list(
                self.identified_entities[vtype], [row]
            )
        event_rows = []
        for row in result.get("events") or []:
            event_rows.append({
                "id": row.get("id"),
                "type": row.get("type") or "Event",
                "name": row.get("name"),
            })
        if event_rows:
            self.identified_entities["Event"] = _merge_entity_list(
                self.identified_entities["Event"], event_rows
            )

        seen_docs = set(self.document_ids)
        for doc_id in result.get("document_ids") or []:
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                self.document_ids.append(doc_id)

        seen_ev = {(item.get("doc_id"), item.get("text")) for item in self.evidence}
        for item in result.get("evidence") or []:
            key = (item.get("doc_id"), item.get("text"))
            if key in seen_ev:
                continue
            seen_ev.add(key)
            self.evidence.append(dict(item))

        seen_paths = {
            (row.get("path"), row.get("from_id"), row.get("to_id"))
            for row in self.graph_paths
        }
        for row in result.get("graph_paths") or []:
            key = (row.get("path"), row.get("from_id"), row.get("to_id"))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            self.graph_paths.append(dict(row))
        return self

    def stop(self, status, answer=None):
        self.stopping_status = status
        if answer is not None:
            self.final_answer = answer
        return self

    def to_dict(self):
        return asdict(self)
