from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.integration.stsdb_enrichment import enrich_card_entries, enrich_relic_names
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


_CAMPFIRE_OPTIONS = ("rest", "smith", "dig", "toke", "lift", "recall")
_BOSS_REST_FLOORS = {15, 32, 49}


def _build_campfire_options(choice_list: list[Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for index, choice in enumerate(choice_list):
        options.append({
            "choice_index": index,
            "choice_token": str(choice).strip().lower(),
        })
    return options


def _build_option_flags(choice_list: list[Any]) -> dict[str, bool]:
    normalized = {str(choice).strip().lower() for choice in choice_list}
    return {option: option in normalized for option in _CAMPFIRE_OPTIONS}


def _build_relic_counters(state: GameState) -> dict[str, int]:
    counters: dict[str, int] = {}
    for relic in state.get_relics():
        name = str(relic.get("name", "")).strip().lower()
        if name == "":
            continue
        try:
            counters[name] = int(relic.get("counter", -1))
        except (TypeError, ValueError):
            counters[name] = -1
    return counters


def build_campfire_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    screen_state = state.screen_state()
    choice_list = state.get_choice_list().copy()
    run_summary = get_cached_run_summary(state)

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=choice_list,
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "room_phase": game_state.get("room_phase"),
            "room_type": game_state.get("room_type"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
        },
        extras={
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "deck_size": run_summary["deck_size"],
            "deck_profile": run_summary["deck_profile"],
            "deck_card_entries": enrich_card_entries(run_summary["deck_card_entries"], game_state.get("class", "")),
            "relic_names": run_summary["relic_names"],
            "relic_summaries": enrich_relic_names(run_summary["relic_names"]),
            "held_potion_names": run_summary["held_potion_names"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "campfire_options": _build_campfire_options(choice_list),
            "campfire_option_flags": _build_option_flags(choice_list),
            "campfire_has_rested": bool(screen_state.get("has_rested", False)),
            "relic_counters": _build_relic_counters(state),
            "is_boss_rest_site": state.floor() in _BOSS_REST_FLOORS,
            "screen_state": screen_state,
            "game_state_ref": state,
        },
    )
