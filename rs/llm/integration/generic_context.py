from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.machine.state import GameState


def build_generic_unhandled_agent_context(state: GameState, handler_name: str = "GenericHandler") -> AgentContext:
    available_commands_raw = state.json.get("available_commands")
    available_commands = [str(command) for command in available_commands_raw] if isinstance(available_commands_raw, list) else []
    game_state = state.game_state()
    choice_list = state.get_choice_list().copy()
    screen_state = state.screen_state() if isinstance(game_state.get("screen_state"), dict) else {}

    generic_payload = {
        "screen": {
            "screen_type": state.screen_type(),
            "screen_name": game_state.get("screen_name"),
            "is_screen_up": game_state.get("is_screen_up"),
            "room_phase": game_state.get("room_phase"),
            "room_type": game_state.get("room_type"),
            "action_phase": game_state.get("action_phase"),
        },
        "commands": {
            "available_commands": available_commands,
            "choice_list": choice_list,
            "ready_for_command": state.json.get("ready_for_command"),
            "in_game": state.json.get("in_game"),
        },
        "game_state": {
            "floor": game_state.get("floor"),
            "act": game_state.get("act"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "keys": game_state.get("keys"),
        },
        "resources": {
            "deck_size": len(game_state.get("deck", [])) if isinstance(game_state.get("deck"), list) else 0,
            "relic_count": len(game_state.get("relics", [])) if isinstance(game_state.get("relics"), list) else 0,
            "potion_slots": len(game_state.get("potions", [])) if isinstance(game_state.get("potions"), list) else 0,
            "map_node_count": len(game_state.get("map", [])) if isinstance(game_state.get("map"), list) else 0,
        },
        "screen_state": dict(screen_state),
    }

    sectioned_schema_explanations = {
        "screen": (
            "Describes where the UI currently is. screen_type/screen_name/room_phase/room_type/action_phase "
            "indicate which interaction mode and progression phase are active."
        ),
        "commands": (
            "Lists executable protocol commands and explicit choices. available_commands are command verbs; "
            "choice_list is the selectable option list aligned to choose indexes."
        ),
        "game_state": (
            "Run-level summary fields. floor/act/class/hp/gold/keys provide compact progress and resource context."
        ),
        "resources": (
            "Compact inventory sizes and counts derived from game_state collections (deck/relics/potions/map)."
        ),
        "screen_state": (
            "Screen-specific raw payload. Structure varies by screen_type and contains detailed local data."
        ),
    }

    field_dictionary = {
        "available_commands": {
            "type": "list[str]",
            "meaning": "Executable command verbs accepted by the game protocol right now.",
            "constraints": "Returned command name must exist here (or mapped alias like proceed/confirm).",
        },
        "choice_list": {
            "type": "list[str]",
            "meaning": "Selectable options for choose command.",
            "constraints": "choose indexes are 0-based and must be within [0, len(choice_list)-1].",
        },
        "screen_type": {
            "type": "str",
            "meaning": "Primary UI mode identifier.",
            "constraints": "Determines shape/semantics of screen_state and expected legal actions.",
        },
        "screen_state": {
            "type": "dict[str, Any]",
            "meaning": "Detailed payload specific to the active screen_type.",
            "constraints": "Schema is variable; use together with screen_type.",
        },
        "room_phase": {
            "type": "str | None",
            "meaning": "High-level room progression stage (e.g. EVENT/COMBAT/COMPLETE).",
            "constraints": "May be absent for some screens.",
        },
        "action_phase": {
            "type": "str | None",
            "meaning": "Low-level action gate from protocol state machine.",
            "constraints": "WAITING_ON_USER usually means next command is expected.",
        },
        "keys": {
            "type": "dict[str, bool] | None",
            "meaning": "Act key acquisition flags.",
            "constraints": "Optional; includes emerald/ruby/sapphire when present.",
        },
    }

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=available_commands,
        choice_list=choice_list,
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "room_phase": game_state.get("room_phase"),
            "room_type": game_state.get("room_type"),
            "character_class": game_state.get("class"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "screen_name": game_state.get("screen_name"),
            "action_phase": game_state.get("action_phase"),
            "ascension_level": game_state.get("ascension_level"),
        },
        extras={
            "generic_payload": generic_payload,
            "sectioned_schema_explanations": sectioned_schema_explanations,
            "field_dictionary": field_dictionary,
            "raw_game_state_keys": sorted([str(key) for key in game_state.keys()]),
        },
    )
