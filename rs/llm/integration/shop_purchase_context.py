from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.integration.stsdb_enrichment import (
    enrich_relic_names,
    query_card_description,
    query_potion_description,
    query_relic_description,
)
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_shop_offer_summaries(screen_state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cards = screen_state.get("cards")
    relics = screen_state.get("relics")
    potions = screen_state.get("potions")

    card_summaries: list[dict[str, Any]] = []
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            entry: dict[str, Any] = {
                "name": str(card.get("name", "")).strip().lower(),
                "type": card.get("type"),
                "rarity": card.get("rarity"),
                "cost": card.get("cost"),
                "price": card.get("price"),
                "upgrades": card.get("upgrades", 0),
                "exhausts": bool(card.get("exhausts", False)),
            }
            desc = query_card_description(entry["name"], entry["upgrades"])
            if desc:
                entry["description"] = desc
            card_summaries.append(entry)

    relic_summaries: list[dict[str, Any]] = []
    if isinstance(relics, list):
        for relic in relics:
            if not isinstance(relic, dict):
                continue
            relic_entry: dict[str, Any] = {
                "name": str(relic.get("name", "")).strip(),
                "price": relic.get("price"),
            }
            rdesc = query_relic_description(relic_entry["name"])
            if rdesc:
                relic_entry["description"] = rdesc
            relic_summaries.append(relic_entry)

    potion_summaries: list[dict[str, Any]] = []
    if isinstance(potions, list):
        for potion in potions:
            if not isinstance(potion, dict):
                continue
            potion_entry: dict[str, Any] = {
                "name": str(potion.get("name", "")).strip(),
                "price": potion.get("price"),
                "requires_target": bool(potion.get("requires_target", False)),
            }
            pdesc = query_potion_description(potion_entry["name"])
            if pdesc:
                potion_entry["description"] = pdesc
            potion_summaries.append(potion_entry)

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
    run_summary = get_cached_run_summary(state)

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
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "has_removable_curse": state.deck.contains_curses_we_can_remove(),
            "deck_size": run_summary["deck_size"],
            "deck_profile": run_summary["deck_profile"],
            "relic_names": run_summary["relic_names"],
            "relic_summaries": enrich_relic_names(run_summary["relic_names"]),
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "purge_cost": screen_state.get("purge_cost"),
            "purge_available": bool(screen_state.get("purge_available", False)),
            "offer_summaries": offer_summaries,
        },
    )
