from __future__ import annotations

from typing import Any

from rs.game.screen_type import ScreenType
from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def is_astrolabe_transform_state(state: GameState) -> bool:
    screen_state = state.screen_state()
    return (
        state.screen_type() == ScreenType.GRID.value
        and str(state.game_state().get("room_type", "")).strip() == "TreasureRoomBoss"
        and int(screen_state.get("num_cards", 0) or 0) == 3
    )


def _normalize_relic_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_boss_relic_options(state: GameState) -> tuple[list[dict[str, Any]], bool]:
    screen_relics = state.screen_state().get("relics", [])
    if not isinstance(screen_relics, list):
        screen_relics = []

    choice_list = state.get_choice_list()
    mismatch = False
    options: list[dict[str, Any]] = []
    for index, choice in enumerate(choice_list):
        choice_name = _normalize_relic_name(choice)
        screen_relic = screen_relics[index] if index < len(screen_relics) and isinstance(screen_relics[index], dict) else {}
        screen_name = _normalize_relic_name(screen_relic.get("name"))
        if screen_name and choice_name and screen_name != choice_name:
            mismatch = True
        options.append({
            "choice_index": index,
            "choice_name": choice_name,
            "screen_relic_name": screen_name,
            "screen_relic_id": str(screen_relic.get("id", "")).strip(),
        })
    return options, mismatch


def build_boss_reward_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    run_summary = get_cached_run_summary(state)
    boss_relic_options, choice_metadata_mismatch = _build_boss_relic_options(state)

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
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
            "deck_card_entries": run_summary["deck_card_entries"],
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "boss_relic_options": boss_relic_options,
            "skip_available": "skip" in [str(command) for command in available_commands],
            "choice_metadata_mismatch": choice_metadata_mismatch,
            "screen_state": state.screen_state(),
            "game_state_ref": state,
        },
    )
