from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.utils.type_utils import is_int_string


@dataclass
class ValidationResult:
    """Result object produced by command validation."""

    is_valid: bool
    code: str
    message: str


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
            return ValidationResult(False, "index_out_of_range", "choose index out of range")
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
        return ValidationResult(False, "index_out_of_range", "play card index out of hand bounds")

    if len(args) == 2:
        if not is_int_string(args[1]):
            return ValidationResult(False, "bad_syntax", "play target index must be integer")
        alive_monsters = int(context.extras.get("alive_monster_count", 0))
        target = int(args[1])
        if alive_monsters > 0 and (target < 0 or target >= alive_monsters):
            return ValidationResult(False, "index_out_of_range", "play target index out of bounds")

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
        return ValidationResult(False, "index_out_of_range", "potion index out of range")

    if len(args) == 3 and not is_int_string(args[2]):
        return ValidationResult(False, "bad_syntax", "potion target index must be integer")

    return ValidationResult(True, "ok", "valid")


def validate_command(context: Any, command: str) -> ValidationResult:
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

    if command_name == "choose":
        return _validate_choose(args, context)

    if command_name == "wait":
        return _validate_wait(args)

    if command_name == "play":
        return _validate_play(args, context)

    if command_name == "potion":
        return _validate_potion(args, context)

    return ValidationResult(False, "unknown_command", f"unknown command syntax: {command_name}")
