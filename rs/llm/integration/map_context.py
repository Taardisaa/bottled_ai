from __future__ import annotations

from typing import Any

from rs.game.map import Map
from rs.game.path import PathHandlerConfig
from rs.llm.agents.base_agent import AgentContext
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_sorted_path_summaries(
        choice_list: list[str],
        game_map: Map,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    choice_lookup = {choice.split("=", 1)[1]: choice for choice in choice_list if "=" in choice}
    choice_indices = {choice: idx for idx, choice in enumerate(choice_list)}
    for idx, path in enumerate(game_map.paths):
        if not path.rooms:
            continue

        first_room = path.rooms[0]
        first_room_x = first_room.id.split("_", 1)[0]
        choice_label = choice_lookup.get(first_room_x, f"x={first_room_x}")
        summaries.append({
            "rank": idx,
            "choice_label": choice_label,
            "choice_command": f"choose {choice_indices.get(choice_label, 0)}",
            "first_room_id": first_room.id,
            "first_room_symbol": first_room.type.value,
            "rooms": [room.type.value for room in path.rooms],
            "room_counts": {room_type.value: count for room_type, count in path.room_count.items()},
            "reward": round(path.reward, 4),
            "survivability": round(path.survivability, 4),
            "reward_survivability": round(path.reward_survivability, 4),
            "ending_hp_estimate": round(path.hp, 4),
            "ending_gold_estimate": round(path.gold, 4),
        })
    return summaries


def _first_room_index(rooms: list[str], symbol: str) -> int | None:
    for idx, room in enumerate(rooms):
        if room == symbol:
            return idx
    return None


def _build_choice_path_overviews(sorted_path_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overview_by_choice: dict[str, dict[str, Any]] = {}
    for summary in sorted_path_summaries:
        choice_label = str(summary["choice_label"])
        rooms = list(summary["rooms"])
        current = overview_by_choice.get(choice_label)
        if current is None or float(summary["reward_survivability"]) > float(current["reward_survivability"]):
            overview_by_choice[choice_label] = {
                "choice_label": choice_label,
                "choice_command": summary["choice_command"],
                "reward_survivability": summary["reward_survivability"],
                "survivability": summary["survivability"],
                "reward": summary["reward"],
                "room_counts": dict(summary["room_counts"]),
                "rooms": rooms,
                "shop_distance": _first_room_index(rooms, "$"),
                "elite_distance": _first_room_index(rooms, "E"),
                "campfire_distance": _first_room_index(rooms, "R"),
                "question_distance": _first_room_index(rooms, "?"),
            }
    return sorted(overview_by_choice.values(), key=lambda item: item["reward_survivability"])


def _resolve_current_position(state: GameState) -> str:
    node = state.game_state()["screen_state"]["current_node"]
    return f"{node['x']}_{node['y']}"


def build_map_agent_context(
        state: GameState,
        handler_name: str,
        config: PathHandlerConfig | None = None,
) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    chosen_config = PathHandlerConfig() if config is None else config
    current_position = _resolve_current_position(state)
    game_map = Map(state.get_map(), current_position, state.game_state()["floor"])
    game_map.sort_paths_by_reward_to_survivability(state, chosen_config)

    sorted_path_summaries = _build_sorted_path_summaries(state.get_choice_list(), game_map)
    choice_path_overviews = _build_choice_path_overviews(sorted_path_summaries)
    deterministic_choice_index = game_map.get_path_choice_from_choices(state.get_choice_list())
    deterministic_choice_command = f"choose {deterministic_choice_index}"
    run_summary = get_cached_run_summary(state)

    game_state = state.game_state()
    screen_state = state.screen_state()
    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "room_type": game_state.get("room_type"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
            "current_position": current_position,
        },
        extras={
            "run_id": run_summary["run_id"],
            "deck_profile": run_summary["deck_profile"],
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "next_nodes": screen_state.get("next_nodes", []),
            "boss_available": bool(screen_state.get("boss_available", False)),
            "first_node_chosen": bool(screen_state.get("first_node_chosen", False)),
            "sorted_path_summaries": sorted_path_summaries,
            "choice_path_overviews": choice_path_overviews,
            "deterministic_best_command": deterministic_choice_command,
        },
    )
