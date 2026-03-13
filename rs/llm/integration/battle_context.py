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

    return {
        "total_cards": len(state.deck.cards),
        "type_counts": type_counts,
        "cost_buckets": cost_buckets,
    }


def _build_monster_summaries(state: GameState) -> list[dict[str, Any]]:
    monsters = []
    for monster in state.get_monsters():
        powers = monster.get("powers", [])
        power_names = [str(power.get("id", "")) for power in powers if isinstance(power, dict)]
        monsters.append({
            "name": monster.get("name"),
            "current_hp": monster.get("current_hp"),
            "max_hp": monster.get("max_hp"),
            "block": monster.get("block"),
            "intent": monster.get("intent"),
            "is_gone": bool(monster.get("is_gone", False)),
            "power_names": power_names,
        })
    return monsters


def _build_player_power_summaries(state: GameState) -> list[dict[str, Any]]:
    combat_state = state.combat_state()
    if combat_state is None:
        return []

    player = combat_state.get("player", {})
    powers = player.get("powers", [])
    return [
        {
            "id": power.get("id"),
            "amount": power.get("amount"),
        }
        for power in powers
        if isinstance(power, dict)
    ]


def build_battle_meta_agent_context(
        state: GameState,
        handler_name: str,
        deterministic_profile: str,
        available_profiles: list[str],
) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    combat_state = state.combat_state() or {}
    player = combat_state.get("player", {})
    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=[],
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "room_type": game_state.get("room_type"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "turn": combat_state.get("turn"),
            "player_block": player.get("block"),
            "player_energy": player.get("energy"),
        },
        extras={
            "deterministic_profile": deterministic_profile,
            "available_profiles": available_profiles,
            "monster_summaries": _build_monster_summaries(state),
            "player_power_summaries": _build_player_power_summaries(state),
            "relic_names": [relic["name"] for relic in state.get_relics()],
            "held_potion_names": state.get_held_potion_names(),
            "deck_profile": _build_deck_profile(state),
        },
    )
