from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rs.machine.state import GameState


class BattleRuntimeAdapter(Protocol):
    """Runtime bridge for a single battle session."""

    def current_state(self) -> GameState:
        raise Exception("must be implemented by runtime adapter")

    def execute(self, commands: list[str]) -> GameState:
        raise Exception("must be implemented by runtime adapter")


@dataclass
class BattleSessionResult:
    handled: bool
    final_state: GameState | None = None
    session_id: str = ""
    executed_commands: list[list[str]] = field(default_factory=list)
    steps: int = 0
    summary: str = ""
