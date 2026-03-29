from __future__ import annotations

from typing import Any

from rs.game.event import Event
from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_strategy_name
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_event_options(screen_state: dict[str, Any]) -> list[dict[str, Any]]:
    options = screen_state.get("options")
    if not isinstance(options, list):
        return []

    normalized_options: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        choice_index = option.get("choice_index", index)
        try:
            normalized_index = int(choice_index)
        except (TypeError, ValueError):
            normalized_index = index
        normalized_options.append(
            {
                "choice_index": normalized_index,
                "label": str(option.get("label", "")).strip(),
                "text": str(option.get("text", "")).strip(),
                "disabled": bool(option.get("disabled", False)),
            }
        )
    return normalized_options


def build_event_agent_context(state: GameState, handler_name: str) -> AgentContext:
    """Build compact event advisor context from current game state."""
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    event = state.get_event()
    event_name = event.value if isinstance(event, Event) else str(event)
    game_state = state.game_state()
    screen_state = state.screen_state()
    run_summary = get_cached_run_summary(state)

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "event_name": event_name,
            "floor": state.floor(),
            "act": game_state.get("act"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "character_class": game_state.get("class"),
            "event_options": _build_event_options(screen_state),
        },
        extras={
            "run_id": run_summary["run_id"],
            "strategy_name": get_current_strategy_name(),
            "relic_names": run_summary["relic_names"],
            "deck_size": run_summary["deck_size"],
            "run_memory_summary": run_summary["run_memory_summary"],
        },
    )
