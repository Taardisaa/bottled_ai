from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.machine.state import GameState


def _build_deck_profile(state: GameState) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    cost_buckets = {
        "zero_cost": 0,
        "one_cost": 0,
        "two_cost": 0,
        "three_plus_cost": 0,
        "x_cost": 0,
        "unplayable": 0,
    }
    upgraded_cards = 0

    for card in state.deck.cards:
        type_key = card.type.value
        type_counts[type_key] = type_counts.get(type_key, 0) + 1

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

    return {
        "total_cards": len(state.deck.cards),
        "type_counts": type_counts,
        "cost_buckets": cost_buckets,
        "upgraded_cards": upgraded_cards,
    }


def _build_shop_offer_summaries(screen_state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cards = screen_state.get("cards")
    relics = screen_state.get("relics")
    potions = screen_state.get("potions")

    card_summaries: list[dict[str, Any]] = []
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            card_summaries.append({
                "name": str(card.get("name", "")).strip().lower(),
                "type": card.get("type"),
                "rarity": card.get("rarity"),
                "cost": card.get("cost"),
                "price": card.get("price"),
                "upgrades": card.get("upgrades", 0),
                "exhausts": bool(card.get("exhausts", False)),
            })

    relic_summaries: list[dict[str, Any]] = []
    if isinstance(relics, list):
        for relic in relics:
            if not isinstance(relic, dict):
                continue
            relic_summaries.append({
                "name": str(relic.get("name", "")).strip(),
                "price": relic.get("price"),
            })

    potion_summaries: list[dict[str, Any]] = []
    if isinstance(potions, list):
        for potion in potions:
            if not isinstance(potion, dict):
                continue
            potion_summaries.append({
                "name": str(potion.get("name", "")).strip(),
                "price": potion.get("price"),
                "requires_target": bool(potion.get("requires_target", False)),
            })

    return {
        "cards": card_summaries,
        "relics": relic_summaries,
        "potions": potion_summaries,
    }


def build_shop_purchase_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    screen_state = state.screen_state()
    offer_summaries = _build_shop_offer_summaries(screen_state)

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "gold": game_state.get("gold"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "room_type": game_state.get("room_type"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
        },
        extras={
            "has_removable_curse": state.deck.contains_curses_we_can_remove(),
            "deck_size": len(state.deck.cards),
            "deck_profile": _build_deck_profile(state),
            "relic_names": [relic["name"] for relic in state.get_relics()],
            "held_potion_names": state.get_held_potion_names(),
            "potions_full": state.are_potions_full(),
            "purge_cost": screen_state.get("purge_cost"),
            "purge_available": bool(screen_state.get("purge_available", False)),
            "offer_summaries": offer_summaries,
        },
    )
