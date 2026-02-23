from __future__ import annotations

from rs.llm.agents.base_agent import AgentContext
from rs.machine.state import GameState


def _build_card_entries_with_counts(state: GameState) -> list[dict[str, int | str]]:
    counts: dict[tuple[str, int], int] = {}
    for card in state.deck.cards:
        key = (card.name.strip().lower(), int(card.upgrades))
        counts[key] = counts.get(key, 0) + 1

    entries: list[dict[str, int | str]] = []
    for (card_name, upgrade_times), count in counts.items():
        entries.append({
            "name": card_name,
            "upgrade_times": upgrade_times,
            "count": count,
        })
    return entries


def build_card_reward_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    deck_card_counts = state.get_deck_card_list_by_name_with_upgrade_stripped()
    deck_card_entries = _build_card_entries_with_counts(state)

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "room_phase": game_state.get("room_phase"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
        },
        extras={
            "deck_size": len(state.deck.cards),
            "relic_names": [relic["name"] for relic in state.get_relics()],
            "deck_card_name_counts": deck_card_counts,
            "deck_card_entries": deck_card_entries,
        },
    )
