from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rs.llm.agents.base_agent import AgentContext, AgentDecision


@dataclass(frozen=True)
class DecisionMemoryEntry:
    timestamp_utc: str
    handler_name: str
    floor: int | None
    act: int | None
    proposed_command: str
    confidence: float
    explanation: str


class DecisionMemoryStore:
    """Run-scoped memory of accepted LLM decisions for prompt continuity.

    This store is intentionally plain and graph-agnostic. Future LangGraph
    agents can read from and write to it, but the memory model itself stays
    independent of LangGraph runtime details.
    """

    def __init__(self, max_runs: int = 64, max_entries_per_run: int = 8):
        self._max_runs = max_runs
        self._max_entries_per_run = max_entries_per_run
        self._entries_by_run: OrderedDict[str, list[DecisionMemoryEntry]] = OrderedDict()

    def record(self, context: AgentContext, decision: AgentDecision) -> None:
        run_id = self._resolve_run_id(context)
        proposed_command = decision.proposed_command
        if run_id is None or proposed_command is None or decision.fallback_recommended:
            return

        entry = DecisionMemoryEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            handler_name=context.handler_name,
            floor=_coerce_int(context.game_state.get("floor")),
            act=_coerce_int(context.game_state.get("act")),
            proposed_command=str(proposed_command),
            confidence=float(decision.confidence),
            explanation=str(decision.explanation),
        )

        entries = self._entries_by_run.setdefault(run_id, [])
        entries.append(entry)
        if len(entries) > self._max_entries_per_run:
            del entries[:-self._max_entries_per_run]

        self._entries_by_run.move_to_end(run_id)
        while len(self._entries_by_run) > self._max_runs:
            self._entries_by_run.popitem(last=False)

    def get_recent_entries(self, context: AgentContext, limit: int = 3) -> list[DecisionMemoryEntry]:
        run_id = self._resolve_run_id(context)
        if run_id is None:
            return []

        entries = self._entries_by_run.get(run_id, [])
        if limit <= 0:
            return []
        return list(entries[-limit:])

    def build_recent_decisions_summary(self, context: AgentContext, limit: int = 3) -> str:
        entries = self.get_recent_entries(context, limit=limit)
        if not entries:
            return "none"

        parts = []
        for entry in entries:
            location = []
            if entry.act is not None:
                location.append(f"A{entry.act}")
            if entry.floor is not None:
                location.append(f"F{entry.floor}")
            location_text = " ".join(location)
            location_prefix = f"{location_text} " if location_text else ""
            parts.append(
                f"{location_prefix}{entry.handler_name} -> {entry.proposed_command} "
                f"({entry.confidence:.2f}, {entry.explanation})"
            )
        return " | ".join(parts)

    def clear(self) -> None:
        self._entries_by_run.clear()

    def _resolve_run_id(self, context: AgentContext) -> str | None:
        run_id = context.extras.get("run_id")
        if run_id is None:
            return None
        text = str(run_id).strip()
        return None if text == "" else text


def _coerce_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
