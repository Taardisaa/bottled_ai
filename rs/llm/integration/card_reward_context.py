from __future__ import annotations

from typing import Any

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


def _build_deck_profile(state: GameState) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    rarity_counts: dict[str, int] = {}
    cost_buckets = {
        "zero_cost": 0,
        "one_cost": 0,
        "two_cost": 0,
        "three_plus_cost": 0,
        "x_cost": 0,
        "unplayable": 0,
    }
    upgraded_cards = 0
    exhaust_cards = 0
    targeted_cards = 0

    for card in state.deck.cards:
        type_key = card.type.value
        type_counts[type_key] = type_counts.get(type_key, 0) + 1

        rarity_key = card.rarity.value
        rarity_counts[rarity_key] = rarity_counts.get(rarity_key, 0) + 1

        if card.cost == -1:
            cost_buckets["x_cost"] += 1
        elif card.cost < 0:
            cost_buckets["unplayable"] += 1
        elif card.cost == 0:
            cost_buckets["zero_cost"] += 1
        elif card.cost == 1:
            cost_buckets["one_cost"] += 1
        elif card.cost == 2:
            cost_buckets["two_cost"] += 1
        else:
            cost_buckets["three_plus_cost"] += 1

        if card.upgrades > 0:
            upgraded_cards += 1
        if card.exhausts:
            exhaust_cards += 1
        if card.has_target:
            targeted_cards += 1

    return {
        "total_cards": len(state.deck.cards),
        "type_counts": type_counts,
        "rarity_counts": rarity_counts,
        "cost_buckets": cost_buckets,
        "upgraded_cards": upgraded_cards,
        "exhaust_cards": exhaust_cards,
        "targeted_cards": targeted_cards,
    }


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
            "room_type": game_state.get("room_type"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
        },
        extras={
            "deck_size": len(state.deck.cards),
            "relic_names": [relic["name"] for relic in state.get_relics()],
            "held_potion_names": state.get_held_potion_names(),
            "potions_full": state.are_potions_full(),
            "deck_card_name_counts": deck_card_counts,
            "deck_card_entries": deck_card_entries,
            "deck_profile": _build_deck_profile(state),
            "choice_card_summaries": _build_choice_card_summaries(state),
            "reward_screen_flags": {
                "bowl_available": bool(state.screen_state().get("bowl_available", False)),
                "skip_available": bool(state.screen_state().get("skip_available", False)),
            },
        },
    )
