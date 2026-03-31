from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rs.utils.type_utils import is_int_string


@dataclass
class ValidationResult:
    """Result object produced by command validation."""

    is_valid: bool
    code: str
    message: str
    normalized_command: str = ""
    reason_details: dict[str, Any] = field(default_factory=dict)
    source_validator: str = "generic"


def _validate_choose(args: list[str], context: Any) -> ValidationResult:
    """Validate `choose` command syntax and references.

    Args:
        args: Tokens after the command name.
        context: Agent context carrying `choice_list`.

    Returns:
        ValidationResult: Validation outcome.
    """
    if len(args) != 1:
        return ValidationResult(False, "bad_syntax", "choose requires exactly one argument")

    choice = args[0]
    if is_int_string(choice):
        idx = int(choice)
        if idx < 0 or idx >= len(context.choice_list):
            return ValidationResult(
                False,
                "index_out_of_range",
                "choose index out of range",
                reason_details={"choice_list_len": len(context.choice_list), "index": idx},
            )
        return ValidationResult(True, "ok", "valid")

    if choice not in context.choice_list:
        return ValidationResult(False, "choice_not_found", "choose option not found in choice_list")
    return ValidationResult(True, "ok", "valid")


def _validate_wait(args: list[str]) -> ValidationResult:
    """Validate `wait` command syntax.

    Args:
        args: Tokens after the command name.

    Returns:
        ValidationResult: Validation outcome.
    """
    if len(args) != 1 or not is_int_string(args[0]):
        return ValidationResult(False, "bad_syntax", "wait requires integer duration")
    if int(args[0]) < 0:
        return ValidationResult(False, "bad_syntax", "wait duration must be non-negative")
    return ValidationResult(True, "ok", "valid")


def _validate_play(args: list[str], context: Any) -> ValidationResult:
    """Validate `play` command syntax and index bounds.

    Args:
        args: Tokens after the command name.
        context: Agent context containing optional combat bounds in `extras`.

    Returns:
        ValidationResult: Validation outcome.
    """
    if len(args) not in {1, 2}:
        return ValidationResult(False, "bad_syntax", "play requires card index and optional target index")
    if not is_int_string(args[0]):
        return ValidationResult(False, "bad_syntax", "play card index must be integer")

    hand_size = int(context.extras.get("hand_size", 0))
    card_idx = int(args[0])
    if hand_size > 0 and (card_idx < 0 or card_idx > hand_size):
        return ValidationResult(
            False,
            "index_out_of_range",
            "play card index out of hand bounds",
            reason_details={"hand_size": hand_size, "card_index": card_idx},
        )

    if len(args) == 2:
        if not is_int_string(args[1]):
            return ValidationResult(False, "bad_syntax", "play target index must be integer")
        alive_monsters = int(context.extras.get("alive_monster_count", 0))
        target = int(args[1])
        if alive_monsters > 0 and (target < 0 or target >= alive_monsters):
            return ValidationResult(
                False,
                "index_out_of_range",
                "play target index out of bounds",
                reason_details={"alive_monster_count": alive_monsters, "target_index": target},
            )

    return ValidationResult(True, "ok", "valid")


def _validate_potion(args: list[str], context: Any) -> ValidationResult:
    """Validate `potion` command syntax and slot bounds.

    Args:
        args: Tokens after the command name.
        context: Agent context containing optional potion bounds in `extras`.

    Returns:
        ValidationResult: Validation outcome.
    """
    if len(args) not in {2, 3}:
        return ValidationResult(False, "bad_syntax", "potion command must be 'potion use|discard <idx> [target]' ")
    if args[0] not in {"use", "discard"}:
        return ValidationResult(False, "bad_syntax", "potion action must be use or discard")
    if not is_int_string(args[1]):
        return ValidationResult(False, "bad_syntax", "potion index must be integer")

    potion_slots = int(context.extras.get("potion_slots", 0))
    potion_idx = int(args[1])
    if potion_slots > 0 and (potion_idx < 0 or potion_idx >= potion_slots):
        return ValidationResult(
            False,
            "index_out_of_range",
            "potion index out of range",
            reason_details={"potion_slots": potion_slots, "potion_index": potion_idx},
        )

    if len(args) == 3 and not is_int_string(args[2]):
        return ValidationResult(False, "bad_syntax", "potion target index must be integer")

    return ValidationResult(True, "ok", "valid")


def _battle_state_from_context(context: Any) -> Any:
    state = getattr(context, "extras", {}).get("game_state_ref")
    if state is None:
        return None
    return state


def _alive_target_indexes_for_battle(state: Any) -> list[int]:
    indexes: list[int] = []
    if state is None:
        return indexes
    for index, monster in enumerate(state.get_monsters()):
        if not bool(monster.get("is_gone", False)):
            indexes.append(index)
    return indexes


def _selection_card_count_for_battle(context: Any) -> int:
    try:
        return int(getattr(context, "extras", {}).get("selection_card_count", 0))
    except (TypeError, ValueError):
        return 0


def _validate_choose_battle(args: list[str], context: Any) -> ValidationResult:
    if len(args) != 1 or not is_int_string(args[0]):
        return ValidationResult(
            False,
            "bad_syntax",
            "battle choose requires exactly one integer index",
            source_validator="battle",
        )
    index = int(args[0])
    selection_count = _selection_card_count_for_battle(context)
    if selection_count > 0:
        if index < 0 or index >= selection_count:
            return ValidationResult(
                False,
                "index_out_of_range",
                "battle choose index out of selection bounds",
                reason_details={"selection_count": selection_count, "index": index},
                source_validator="battle",
            )
        return ValidationResult(True, "ok", "valid", source_validator="battle")

    choice_list = getattr(context, "choice_list", [])
    if index < 0 or index >= len(choice_list):
        return ValidationResult(
            False,
            "index_out_of_range",
            "battle choose index out of choice bounds",
            reason_details={"choice_list_len": len(choice_list), "index": index},
            source_validator="battle",
        )
    return ValidationResult(True, "ok", "valid", source_validator="battle")


def _validate_play_battle(args: list[str], context: Any) -> ValidationResult:
    state = _battle_state_from_context(context)
    if state is None:
        return ValidationResult(
            False,
            "missing_state",
            "battle context missing game_state_ref",
            source_validator="battle",
        )
    if len(args) not in {1, 2}:
        return ValidationResult(False, "bad_syntax", "play requires card index and optional target index", source_validator="battle")
    if not is_int_string(args[0]):
        return ValidationResult(False, "bad_syntax", "play card index must be integer", source_validator="battle")

    card_index = int(args[0])
    if card_index < 1 or card_index > len(state.hand.cards):
        return ValidationResult(
            False,
            "index_out_of_range",
            "play card index out of hand bounds",
            reason_details={"hand_size": len(state.hand.cards), "card_index": card_index},
            source_validator="battle",
        )

    card = state.hand.cards[card_index - 1]
    if not bool(card.is_playable):
        return ValidationResult(False, "command_not_available", "selected card is not currently playable", source_validator="battle")

    alive_targets = _alive_target_indexes_for_battle(state)
    if card.has_target:
        if len(args) != 2 or not is_int_string(args[1]):
            return ValidationResult(False, "bad_syntax", "targeted play requires target index", source_validator="battle")
        target_index = int(args[1])
        if target_index not in alive_targets:
            return ValidationResult(
                False,
                "index_out_of_range",
                "play target is not a live monster index",
                reason_details={"alive_target_indexes": alive_targets, "target_index": target_index},
                source_validator="battle",
            )
    elif len(args) == 2:
        return ValidationResult(False, "bad_syntax", "untargeted card cannot include a target index", source_validator="battle")

    return ValidationResult(True, "ok", "valid", source_validator="battle")


def _validate_potion_battle(args: list[str], context: Any) -> ValidationResult:
    state = _battle_state_from_context(context)
    if state is None:
        return ValidationResult(
            False,
            "missing_state",
            "battle context missing game_state_ref",
            source_validator="battle",
        )
    if len(args) not in {2, 3}:
        return ValidationResult(False, "bad_syntax", "potion command must be 'potion use|discard <idx> [target]'", source_validator="battle")
    if args[0] not in {"use", "discard"}:
        return ValidationResult(False, "bad_syntax", "potion action must be use or discard", source_validator="battle")
    if not is_int_string(args[1]):
        return ValidationResult(False, "bad_syntax", "potion slot must be integer", source_validator="battle")

    potion_index = int(args[1])
    potions = state.get_potions()
    if potion_index < 0 or potion_index >= len(potions):
        return ValidationResult(
            False,
            "index_out_of_range",
            "potion slot out of range",
            reason_details={"potion_slots": len(potions), "potion_index": potion_index},
            source_validator="battle",
        )
    potion = potions[potion_index]
    if str(potion.get("id", "")).strip() == "Potion Slot":
        return ValidationResult(False, "command_not_available", "cannot use or discard an empty potion slot", source_validator="battle")

    if len(args) == 3:
        if not is_int_string(args[2]):
            return ValidationResult(False, "bad_syntax", "potion target must be integer", source_validator="battle")
        target_index = int(args[2])
        alive_targets = _alive_target_indexes_for_battle(state)
        if target_index not in alive_targets:
            return ValidationResult(
                False,
                "index_out_of_range",
                "potion target is not a live monster index",
                reason_details={"alive_target_indexes": alive_targets, "target_index": target_index},
                source_validator="battle",
            )
    return ValidationResult(True, "ok", "valid", source_validator="battle")


def validate_command(context: Any, command: str, mode: str = "generic") -> ValidationResult:
    """Validate proposed command against available commands and basic syntax.

    Args:
        context: Current decision context.
        command: Proposed command string.

    Returns:
        ValidationResult: Structured validation status and error code.
    """
    command = command.strip()
    if command == "":
        return ValidationResult(False, "empty_command", "command is empty")

    tokens = command.split()
    command_name = tokens[0]
    args = tokens[1:]

    availability_aliases: dict[str, set[str]] = {
        "return": {"leave", "cancel"},
        "proceed": {"proceed", "confirm"},
    }
    required = availability_aliases.get(command_name, {command_name})
    if not any(req in context.available_commands for req in required):
        return ValidationResult(False, "command_not_available", f"{command_name} is not in available_commands")

    if command_name in {"end", "skip", "confirm", "proceed", "return", "leave", "cancel"}:
        if args:
            return ValidationResult(False, "bad_syntax", f"{command_name} does not accept arguments")
        return ValidationResult(True, "ok", "valid")

    if mode == "battle":
        if command_name == "choose":
            result = _validate_choose_battle(args, context)
            result.normalized_command = command
            return result
        if command_name == "play":
            result = _validate_play_battle(args, context)
            result.normalized_command = command
            return result
        if command_name == "potion":
            result = _validate_potion_battle(args, context)
            result.normalized_command = command
            return result

    if command_name == "choose":
        result = _validate_choose(args, context)
        result.normalized_command = command
        return result

    if command_name == "wait":
        result = _validate_wait(args)
        result.normalized_command = command
        return result

    if command_name == "play":
        result = _validate_play(args, context)
        result.normalized_command = command
        return result

    if command_name == "potion":
        result = _validate_potion(args, context)
        result.normalized_command = command
        return result

    return ValidationResult(False, "unknown_command", f"unknown command syntax: {command_name}")


def validate_command_batch(context: Any, commands: list[str], mode: str = "generic") -> dict[str, Any]:
    normalized_commands = [str(command).strip() for command in commands if str(command).strip()]
    if not normalized_commands:
        return {"is_valid": False, "errors": ["empty_command_batch"], "commands": []}

    errors: list[dict[str, Any]] = []
    for command in normalized_commands:
        validation = validate_command(context, command, mode=mode)
        if not validation.is_valid:
            errors.append({
                "command": command,
                "code": validation.code,
                "message": validation.message,
                "reason_details": dict(validation.reason_details),
                "source_validator": validation.source_validator,
            })

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "commands": normalized_commands,
        "summary": "valid" if not errors else f"{len(errors)} validation errors",
        "mode": mode,
    }
