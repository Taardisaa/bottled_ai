from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx

from rs.game.map import Map
from rs.game.path import PathHandlerConfig
from rs.game.room import RoomType
from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
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


def _node_id(x_coord: Any, y_coord: Any) -> str:
    return f"{x_coord}_{y_coord}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_map_graph(raw_map: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for room in raw_map:
        if not isinstance(room, dict):
            continue
        node_id = _node_id(room.get("x"), room.get("y"))
        symbol = str(room.get("symbol", "")).strip() or "?"
        room_type_name = RoomType(symbol).name if symbol in {member.value for member in RoomType} else symbol
        graph.add_node(
            node_id,
            node_id=node_id,
            x=_safe_int(room.get("x")),
            y=_safe_int(room.get("y")),
            room_symbol=symbol,
            room_type=room_type_name,
        )

    for room in raw_map:
        if not isinstance(room, dict):
            continue
        parent_id = _node_id(room.get("x"), room.get("y"))
        for child in room.get("children", []):
            if not isinstance(child, dict):
                continue
            child_id = _node_id(child.get("x"), child.get("y"))
            if child_id in graph:
                graph.add_edge(parent_id, child_id)

    boss_room_id = _resolve_boss_room_id(raw_map)
    if boss_room_id is not None and boss_room_id not in graph:
        boss_x, boss_y = boss_room_id.split("_", 1)
        graph.add_node(
            boss_room_id,
            node_id=boss_room_id,
            x=_safe_int(boss_x),
            y=_safe_int(boss_y),
            room_symbol=RoomType.BOSS.value,
            room_type=RoomType.BOSS.name,
        )

    if boss_room_id is not None:
        for room in raw_map:
            if not isinstance(room, dict):
                continue
            parent_id = _node_id(room.get("x"), room.get("y"))
            for child in room.get("children", []):
                if not isinstance(child, dict):
                    continue
                child_id = _node_id(child.get("x"), child.get("y"))
                if child_id == boss_room_id:
                    graph.add_edge(parent_id, boss_room_id)

    start_children = [
        _node_id(room.get("x"), room.get("y"))
        for room in raw_map
        if isinstance(room, dict) and _safe_int(room.get("y"), -999) == 0
    ]
    graph.add_node(
        "0_-1",
        node_id="0_-1",
        x=0,
        y=-1,
        room_symbol="START",
        room_type="START",
    )
    for child_id in start_children:
        if child_id in graph:
            graph.add_edge("0_-1", child_id)

    return graph


def _resolve_boss_room_id(raw_map: list[dict[str, Any]]) -> str | None:
    for room in reversed(raw_map):
        if not isinstance(room, dict):
            continue
        children = room.get("children", [])
        if isinstance(children, list) and children:
            first_child = children[0]
            if isinstance(first_child, dict) and "x" in first_child and "y" in first_child:
                return _node_id(first_child.get("x"), first_child.get("y"))
    return None


def _collect_descendant_paths(graph: nx.DiGraph, start_node_id: str) -> list[list[str]]:
    if start_node_id not in graph:
        return []

    descendants: list[list[str]] = []

    def _walk(node_id: str, path: list[str]) -> None:
        children = list(graph.successors(node_id))
        if not children:
            descendants.append(path.copy())
            return
        for child_id in children:
            path.append(child_id)
            _walk(child_id, path)
            path.pop()

    _walk(start_node_id, [start_node_id])
    return descendants


def _room_names_for_path(graph: nx.DiGraph, node_path: list[str]) -> list[str]:
    return [str(graph.nodes[node_id].get("room_type", "UNKNOWN")) for node_id in node_path if node_id in graph]


def _room_counts_for_names(room_names: list[str]) -> dict[str, int]:
    tracked_room_types = [
        RoomType.MONSTER.name,
        RoomType.QUESTION.name,
        RoomType.ELITE.name,
        RoomType.CAMPFIRE.name,
        RoomType.TREASURE.name,
        RoomType.SHOP.name,
        RoomType.BOSS.name,
    ]
    counts = Counter(room_names)
    return {room_type: int(counts.get(room_type, 0)) for room_type in tracked_room_types}


def _first_room_distance(room_names: list[str], room_name: str) -> int | None:
    for index, value in enumerate(room_names):
        if value == room_name:
            return index
    return None


def _build_choice_graph_families(
        graph: nx.DiGraph,
        current_position: str,
        choice_list: list[str],
) -> list[dict[str, Any]]:
    if current_position not in graph:
        current_position = "0_-1"

    choice_lookup = {choice.split("=", 1)[1]: (choice, idx) for idx, choice in enumerate(choice_list) if "=" in choice}
    families: list[dict[str, Any]] = []
    for child_id in graph.successors(current_position):
        child_x = str(graph.nodes[child_id].get("x"))
        choice_label, choice_index = choice_lookup.get(child_x, (f"x={child_x}", len(families)))
        node_paths = _collect_descendant_paths(graph, child_id)
        path_shapes = [_build_structural_path_summary(graph, node_path, f"choose {choice_index}") for node_path in node_paths]
        families.append(
            {
                "choice_index": choice_index,
                "choice_label": choice_label,
                "choice_command": f"choose {choice_index}",
                "path_shapes": path_shapes,
            }
        )

    families.sort(key=lambda family: int(family["choice_index"]))
    return families


def _build_structural_path_summary(
        graph: nx.DiGraph,
        node_path: list[str],
        choice_command: str,
) -> dict[str, Any]:
    room_names = _room_names_for_path(graph, node_path)
    room_counts = _room_counts_for_names(room_names)
    return {
        "choice_command": choice_command,
        "rooms": room_names,
        "room_counts": room_counts,
        "path_length": len(room_names),
        "first_shop_distance": _first_room_distance(room_names, RoomType.SHOP.name),
        "first_campfire_distance": _first_room_distance(room_names, RoomType.CAMPFIRE.name),
        "first_elite_distance": _first_room_distance(room_names, RoomType.ELITE.name),
    }


def _range_dict(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values)}


def _shared_prefix(room_sequences: list[list[str]]) -> list[str]:
    if not room_sequences:
        return []
    prefix = list(room_sequences[0])
    for sequence in room_sequences[1:]:
        next_prefix: list[str] = []
        for left, right in zip(prefix, sequence):
            if left != right:
                break
            next_prefix.append(left)
        prefix = next_prefix
        if not prefix:
            break
    return prefix


def _build_branch_shape_summary(path_shapes: list[dict[str, Any]]) -> str:
    path_count = len(path_shapes)
    if path_count <= 1:
        return "single descendant path"

    room_sequences = [list(shape["rooms"]) for shape in path_shapes]
    shared_prefix = _shared_prefix(room_sequences)
    if not shared_prefix:
        next_room_types = sorted({sequence[0] for sequence in room_sequences if sequence})
        return f"{path_count} descendant paths; branches immediately into {', '.join(next_room_types)}"

    divergence_index = len(shared_prefix)
    next_room_types = sorted(
        {
            sequence[divergence_index]
            for sequence in room_sequences
            if len(sequence) > divergence_index
        }
    )
    prefix_text = " > ".join(shared_prefix[:3])
    if next_room_types:
        return f"{path_count} descendant paths; shared prefix {prefix_text}; then branches into {', '.join(next_room_types)}"
    return f"{path_count} descendant paths; shared prefix {prefix_text}"


def _build_choice_branch_summaries(choice_families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in choice_families:
        path_shapes = [shape for shape in family.get("path_shapes", []) if isinstance(shape, dict)]
        if not path_shapes:
            continue

        room_count_key_map = {
            "monster_count_range": RoomType.MONSTER.name,
            "event_count_range": RoomType.QUESTION.name,
            "shop_count_range": RoomType.SHOP.name,
            "campfire_count_range": RoomType.CAMPFIRE.name,
            "elite_count_range": RoomType.ELITE.name,
            "treasure_count_range": RoomType.TREASURE.name,
        }
        summary = {
            "choice_index": family["choice_index"],
            "choice_label": family["choice_label"],
            "choice_command": family["choice_command"],
            "path_count": len(path_shapes),
            "branch_shape_summary": _build_branch_shape_summary(path_shapes),
            "first_shop_distance_range": _range_dict(
                [value for value in (shape.get("first_shop_distance") for shape in path_shapes) if isinstance(value, int)]
            ),
            "first_campfire_distance_range": _range_dict(
                [value for value in (shape.get("first_campfire_distance") for shape in path_shapes) if isinstance(value, int)]
            ),
            "first_elite_distance_range": _range_dict(
                [value for value in (shape.get("first_elite_distance") for shape in path_shapes) if isinstance(value, int)]
            ),
        }
        for summary_key, room_name in room_count_key_map.items():
            counts = [int(shape["room_counts"].get(room_name, 0)) for shape in path_shapes]
            summary[summary_key] = _range_dict(counts)
        summaries.append(summary)

    return summaries


def _representative_candidate_paths(path_shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path_shapes:
        return []

    def _distance_key(path: dict[str, Any], field: str) -> tuple[int, tuple[str, ...]]:
        distance = path.get(field)
        normalized_distance = int(distance) if isinstance(distance, int) else 999
        return normalized_distance, tuple(path.get("rooms", []))

    sorted_by_rooms = sorted(path_shapes, key=lambda item: tuple(item.get("rooms", [])))
    candidates = [sorted_by_rooms[0]]

    for field in ("first_shop_distance", "first_elite_distance", "first_campfire_distance"):
        candidates.append(min(path_shapes, key=lambda item: _distance_key(item, field)))

    candidates.append(min(path_shapes, key=lambda item: (int(item.get("path_length", 0)), tuple(item.get("rooms", [])))))

    unique_candidates: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for candidate in candidates:
        signature = tuple(candidate.get("rooms", []))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_candidates.append(candidate)
        if len(unique_candidates) >= 3:
            break
    return unique_candidates


def _build_choice_representative_paths(choice_families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representative_groups: list[dict[str, Any]] = []
    for family in choice_families:
        path_shapes = [shape for shape in family.get("path_shapes", []) if isinstance(shape, dict)]
        representative_paths = _representative_candidate_paths(path_shapes)[:1]
        compact_paths = [
            {
                "rooms": list(path.get("rooms", [])),
                "path_length": path.get("path_length"),
                "first_shop_distance": path.get("first_shop_distance"),
                "first_campfire_distance": path.get("first_campfire_distance"),
                "first_elite_distance": path.get("first_elite_distance"),
            }
            for path in representative_paths
        ]
        representative_groups.append(
            {
                "choice_index": family["choice_index"],
                "choice_label": family["choice_label"],
                "choice_command": family["choice_command"],
                "representative_paths": compact_paths,
            }
        )
    return representative_groups


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
    map_graph = _build_map_graph(state.get_map())
    choice_families = _build_choice_graph_families(map_graph, current_position, state.get_choice_list())

    sorted_path_summaries = _build_sorted_path_summaries(state.get_choice_list(), game_map)
    choice_path_overviews = _build_choice_path_overviews(sorted_path_summaries)
    choice_branch_summaries = _build_choice_branch_summaries(choice_families)
    choice_representative_paths = _build_choice_representative_paths(choice_families)
    deterministic_choice_index = game_map.get_path_choice_from_choices(state.get_choice_list())
    deterministic_choice_command = f"choose {deterministic_choice_index}"
    run_summary = get_cached_run_summary(state)
    raw_deck_profile = run_summary["deck_profile"]
    compact_deck_profile = {
        "total_cards": raw_deck_profile.get("total_cards"),
        "type_counts": raw_deck_profile.get("type_counts", {}),
        "upgraded_cards": raw_deck_profile.get("upgraded_cards"),
    }
    if "exhaust_cards" in raw_deck_profile:
        compact_deck_profile["exhaust_cards"] = raw_deck_profile.get("exhaust_cards")

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
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
        },
        extras={
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "deck_profile": compact_deck_profile,
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "next_nodes": screen_state.get("next_nodes", []),
            "boss_available": bool(screen_state.get("boss_available", False)),
            "first_node_chosen": bool(screen_state.get("first_node_chosen", False)),
            "sorted_path_summaries": sorted_path_summaries,
            "choice_path_overviews": choice_path_overviews,
            "choice_branch_summaries": choice_branch_summaries,
            "choice_representative_paths": choice_representative_paths,
            "map_graph_metadata": {
                "node_count": map_graph.number_of_nodes(),
                "edge_count": map_graph.number_of_edges(),
            },
            "deterministic_best_command": deterministic_choice_command,
        },
    )
