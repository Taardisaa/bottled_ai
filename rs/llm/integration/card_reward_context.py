from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_choice_card_summaries(state: GameState) -> list[dict[str, Any]]:
    screen_cards = state.screen_state().get("cards")
    if not isinstance(screen_cards, list):
        return []

    summaries: list[dict[str, Any]] = []
    for idx, card in enumerate(screen_cards):
        if not isinstance(card, dict):
            continue
        summaries.append({
            "index": idx,
            "name": str(card.get("name", "")).strip().lower(),
            "type": card.get("type"),
            "rarity": card.get("rarity"),
            "cost": card.get("cost"),
            "upgrades": card.get("upgrades", 0),
            "exhausts": bool(card.get("exhausts", False)),
            "has_target": bool(card.get("has_target", False)),
        })
    return summaries


def build_card_reward_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    run_summary = get_cached_run_summary(state)

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
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "deck_card_name_counts": run_summary["deck_card_name_counts"],
            "deck_card_entries": run_summary["deck_card_entries"],
            "deck_profile": run_summary["deck_profile"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "choice_card_summaries": _build_choice_card_summaries(state),
            "reward_screen_flags": {
                "bowl_available": bool(state.screen_state().get("bowl_available", False)),
                "skip_available": bool(state.screen_state().get("skip_available", False)),
            },
        },
    )
