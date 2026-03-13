from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


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
    run_summary = get_cached_run_summary(state)
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
            "run_id": run_summary["run_id"],
            "deterministic_profile": deterministic_profile,
            "available_profiles": available_profiles,
            "monster_summaries": _build_monster_summaries(state),
            "player_power_summaries": _build_player_power_summaries(state),
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "deck_profile": run_summary["deck_profile"],
            "run_memory_summary": run_summary["run_memory_summary"],
        },
    )
