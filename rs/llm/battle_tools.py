from __future__ import annotations

from typing import Any

from rs.calculator.executor import get_best_battle_action
from rs.calculator.interfaces.comparator_interface import ComparatorInterface
from rs.llm.agents.base_agent import AgentContext, AgentTool
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.validator import ValidationResult, validate_command
from rs.machine.state import GameState


def _state_from_context(context: AgentContext) -> GameState:
    state = context.extras.get("game_state_ref")
    if not isinstance(state, GameState):
        raise ValueError("battle context missing GameState reference")
    return state


def _runtime_from_context(context: AgentContext) -> Any:
    runtime = context.extras.get("battle_runtime")
    if runtime is None:
        raise ValueError("battle context missing runtime adapter")
    return runtime


def _alive_target_indexes(state: GameState) -> list[int]:
    indexes: list[int] = []
    for index, monster in enumerate(state.get_monsters()):
        if not bool(monster.get("is_gone", False)):
            indexes.append(index)
    return indexes


def _selection_card_count(context: AgentContext) -> int:
    try:
        return int(context.extras.get("selection_card_count", 0))
    except (TypeError, ValueError):
        return 0


def _validate_choose_for_battle(context: AgentContext, args: list[str]) -> ValidationResult:
    selection_count = _selection_card_count(context)
    if selection_count > 0:
        if len(args) != 1 or not str(args[0]).isdigit():
            return ValidationResult(False, "bad_syntax", "battle choose requires exactly one integer index")
        index = int(args[0])
        if index < 0 or index >= selection_count:
            return ValidationResult(False, "index_out_of_range", "battle choose index out of selection bounds")
        return ValidationResult(True, "ok", "valid")
    return validate_command(context, f"choose {' '.join(args)}")


def _validate_play_for_battle(context: AgentContext, args: list[str]) -> ValidationResult:
    state = _state_from_context(context)
    if len(args) not in {1, 2}:
        return ValidationResult(False, "bad_syntax", "play requires card index and optional target index")
    if not str(args[0]).isdigit():
        return ValidationResult(False, "bad_syntax", "play card index must be integer")

    card_index = int(args[0])
    if card_index < 1 or card_index > len(state.hand.cards):
        return ValidationResult(False, "index_out_of_range", "play card index out of hand bounds")

    card = state.hand.cards[card_index - 1]
    if not bool(card.is_playable):
        return ValidationResult(False, "command_not_available", "selected card is not currently playable")

    if card.has_target:
        if len(args) != 2 or not str(args[1]).isdigit():
            return ValidationResult(False, "bad_syntax", "targeted play requires target index")
        target_index = int(args[1])
        if target_index not in _alive_target_indexes(state):
            return ValidationResult(False, "index_out_of_range", "play target is not a live monster index")
    elif len(args) == 2:
        return ValidationResult(False, "bad_syntax", "untargeted card cannot include a target index")

    return ValidationResult(True, "ok", "valid")


def _validate_potion_for_battle(context: AgentContext, args: list[str]) -> ValidationResult:
    state = _state_from_context(context)
    if len(args) not in {2, 3}:
        return ValidationResult(False, "bad_syntax", "potion command must be 'potion use|discard <idx> [target]'")
    if args[0] not in {"use", "discard"}:
        return ValidationResult(False, "bad_syntax", "potion action must be use or discard")
    if not str(args[1]).isdigit():
        return ValidationResult(False, "bad_syntax", "potion slot must be integer")

    potion_index = int(args[1])
    if potion_index < 0 or potion_index >= len(state.get_potions()):
        return ValidationResult(False, "index_out_of_range", "potion slot out of range")

    potion = state.get_potions()[potion_index]
    if str(potion.get("id", "")).strip() == "Potion Slot":
        return ValidationResult(False, "command_not_available", "cannot use or discard an empty potion slot")

    if len(args) == 3:
        if not str(args[2]).isdigit():
            return ValidationResult(False, "bad_syntax", "potion target must be integer")
        target_index = int(args[2])
        if target_index not in _alive_target_indexes(state):
            return ValidationResult(False, "index_out_of_range", "potion target is not a live monster index")

    return ValidationResult(True, "ok", "valid")


def validate_battle_commands(context: AgentContext, commands: list[str]) -> dict[str, Any]:
    normalized_commands = [str(command).strip() for command in commands if str(command).strip()]
    if not normalized_commands:
        return {"is_valid": False, "errors": ["empty_command_batch"], "commands": []}

    errors: list[dict[str, Any]] = []
    for command in normalized_commands:
        tokens = command.split()
        name = tokens[0]
        args = tokens[1:]
        if name == "choose":
            validation = _validate_choose_for_battle(context, args)
        elif name == "play":
            validation = _validate_play_for_battle(context, args)
        elif name == "potion":
            validation = _validate_potion_for_battle(context, args)
        else:
            validation = validate_command(context, command)

        if not validation.is_valid:
            errors.append({
                "command": command,
                "code": validation.code,
                "message": validation.message,
            })

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "commands": normalized_commands,
        "summary": "valid" if not errors else f"{len(errors)} validation errors",
    }


class LowestEnemyHpComparator(ComparatorInterface):
    def does_challenger_defeat_the_best(self, best, challenger, original) -> bool:
        def _score(candidate: Any) -> tuple[int, int, int, int]:
            total_enemy_hp = sum(max(0, monster.current_hp) for monster in candidate.monsters if not monster.is_gone)
            live_enemy_count = sum(1 for monster in candidate.monsters if not monster.is_gone and monster.current_hp > 0)
            player_hp = max(0, candidate.player.current_hp)
            player_block = max(0, candidate.player.block)
            return total_enemy_hp, live_enemy_count, -player_hp, -player_block

        return _score(challenger) < _score(best)


class EnumerateLegalActionsTool(AgentTool):
    name = "enumerate_legal_actions"

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_context(context)
        commands: list[str] = []
        categories = {
            "play": [],
            "choose": [],
            "potion": [],
            "other": [],
        }

        if "play" in context.available_commands:
            for hand_index, card in enumerate(state.hand.cards, start=1):
                if not bool(card.is_playable):
                    continue
                if card.has_target:
                    for target_index in _alive_target_indexes(state):
                        command = f"play {hand_index} {target_index}"
                        commands.append(command)
                        categories["play"].append(command)
                else:
                    command = f"play {hand_index}"
                    commands.append(command)
                    categories["play"].append(command)

        if "choose" in context.available_commands:
            selection_count = _selection_card_count(context)
            if selection_count > 0:
                for index in range(selection_count):
                    command = f"choose {index}"
                    commands.append(command)
                    categories["choose"].append(command)
            else:
                for index, choice in enumerate(context.choice_list):
                    command = f"choose {index}"
                    commands.append(command)
                    categories["choose"].append(f"{command} ({choice})")

        if "potion" in context.available_commands:
            for index, potion in enumerate(state.get_potions()):
                if str(potion.get("id", "")).strip() == "Potion Slot":
                    continue
                discard_command = f"potion discard {index}"
                commands.append(discard_command)
                categories["potion"].append(discard_command)
                use_command = f"potion use {index}"
                commands.append(use_command)
                categories["potion"].append(use_command)
                for target_index in _alive_target_indexes(state):
                    targeted_command = f"potion use {index} {target_index}"
                    commands.append(targeted_command)
                    categories["potion"].append(targeted_command)

        for simple_command in ["end", "confirm", "proceed", "return", "cancel", "leave", "skip"]:
            if simple_command in context.available_commands:
                commands.append(simple_command)
                categories["other"].append(simple_command)
        if "wait" in context.available_commands:
            commands.append("wait 30")
            categories["other"].append("wait 30")

        return {
            "commands": commands,
            "categories": categories,
            "summary": f"enumerated {len(commands)} legal actions",
        }


class AnalyzeWithCalculatorTool(AgentTool):
    name = "analyze_with_calculator"

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_context(context)
        max_path_count = int(payload.get("max_path_count", 250))
        if max_path_count < 1:
            max_path_count = 1
        try:
            action = get_best_battle_action(state, LowestEnemyHpComparator(), max_path_count=max_path_count)
        except Exception as exc:
            return {
                "recommended_commands": [],
                "summary": "calculator_failed",
                "error": str(exc),
            }

        commands = list(action.commands) if action is not None else []
        return {
            "recommended_commands": commands,
            "summary": "calculator_recommendation" if commands else "calculator_no_action",
        }


class ValidateBattleCommandTool(AgentTool):
    name = "validate_battle_command"

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        if "commands" in payload and isinstance(payload["commands"], list):
            commands = [str(command) for command in payload["commands"]]
        elif "command" in payload:
            commands = [str(payload["command"])]
        else:
            commands = []
        return validate_battle_commands(context, commands)


class ExecuteBattleCommandTool(AgentTool):
    name = "execute_battle_command"

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        if "commands" in payload and isinstance(payload["commands"], list):
            commands = [str(command) for command in payload["commands"]]
        elif "command" in payload:
            commands = [str(payload["command"])]
        else:
            commands = []

        validation = validate_battle_commands(context, commands)
        if not bool(validation.get("is_valid")):
            return {
                "executed": False,
                "commands": commands,
                "validation": validation,
                "summary": "execute_rejected_by_validation",
            }

        runtime = _runtime_from_context(context)
        next_state = runtime.execute(validation["commands"])
        return {
            "executed": True,
            "commands": validation["commands"],
            "state": next_state,
            "validation": validation,
            "summary": f"executed {len(validation['commands'])} commands",
        }


class RetrieveBattleExperienceTool(AgentTool):
    name = "retrieve_battle_experience"

    def __init__(self, langmem_service: LangMemService | None = None):
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._langmem_service.build_context_memory(context)
        result["summary"] = "retrieved battle experience"
        return result
