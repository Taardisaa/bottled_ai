from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_card_rows(cards: Any) -> list[dict[str, Any]]:
    if not isinstance(cards, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        rows.append({
            "choice_index": index,
            "name": str(card.get("name", "")).strip().lower(),
            "id": str(card.get("id", "")).strip(),
            "type": card.get("type"),
            "cost": card.get("cost"),
            "upgrades": card.get("upgrades", 0),
            "rarity": card.get("rarity"),
        })
    return rows


def _grid_mode(screen_state: dict[str, Any]) -> str:
    if screen_state.get("for_purge"):
        return "purge"
    if screen_state.get("for_transform"):
        return "transform"
    if screen_state.get("for_upgrade"):
        return "upgrade"
    return "unknown"


def build_grid_select_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    screen_state = state.screen_state()
    run_summary = get_cached_run_summary(state)
    selectable_cards = _build_card_rows(screen_state.get("cards", []))
    selected_cards = _build_card_rows(screen_state.get("selected_cards", []))
    num_cards = int(screen_state.get("num_cards", 0) or 0)
    confirm_up = bool(screen_state.get("confirm_up", False))
    # When confirm_up is True the selection is complete — the game is waiting for confirm/cancel.
    # selected_cards may still be empty in this state, so derive picks_remaining from confirm_up.
    raw_picks_remaining = max(0, num_cards - len(selected_cards))
    picks_remaining = 0 if confirm_up else raw_picks_remaining

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
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
        },
        extras={
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "grid_mode": _grid_mode(screen_state),
            "for_purge": bool(screen_state.get("for_purge", False)),
            "for_transform": bool(screen_state.get("for_transform", False)),
            "for_upgrade": bool(screen_state.get("for_upgrade", False)),
            "confirm_up": confirm_up,
            "any_number": bool(screen_state.get("any_number", False)),
            "selectable_cards": selectable_cards,
            "selected_cards": selected_cards,
            "num_cards": num_cards,
            "picks_remaining": picks_remaining,
            "deck_profile": run_summary["deck_profile"],
            "deck_card_entries": run_summary["deck_card_entries"],
            "relic_names": run_summary["relic_names"],
            "run_memory_summary": run_summary["run_memory_summary"],
        },
    )
