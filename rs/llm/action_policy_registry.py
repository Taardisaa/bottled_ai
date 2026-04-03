from __future__ import annotations

from dataclasses import dataclass

from rs.llm.agents.base_agent import AgentContext
from rs.llm.validator import ValidationResult, validate_command


@dataclass
class PolicyResolution:
    commands: list[str] | None
    validation: ValidationResult


class ActionPolicyRegistry:
    """Central command normalization and sequencing policy."""

    def resolve(self, context: AgentContext, proposed_command: str | None) -> PolicyResolution:
        if proposed_command is None:
            return PolicyResolution(
                commands=None,
                validation=ValidationResult(False, "empty_command", "command is empty"),
            )
        normalized = str(proposed_command).strip()
        if normalized == "":
            return PolicyResolution(
                commands=None,
                validation=ValidationResult(False, "empty_command", "command is empty"),
            )
        if self._uses_duplicate_choice_token(context, normalized):
            return PolicyResolution(
                commands=None,
                validation=ValidationResult(
                    False,
                    "ambiguous_choice_token",
                    "choose token appears multiple times in choice_list",
                ),
            )

        validation = validate_command(context, normalized)
        if not validation.is_valid:
            return PolicyResolution(commands=None, validation=validation)

        commands = self._commands_for_context(context, normalized)
        if commands is None:
            return PolicyResolution(
                commands=None,
                validation=ValidationResult(
                    False,
                    "unsupported_command_mapping",
                    "no execution mapping for validated command",
                ),
            )
        return PolicyResolution(commands=commands, validation=validation)

    def _commands_for_context(self, context: AgentContext, proposed_command: str) -> list[str] | None:
        if context.handler_name == "EventHandler":
            return [proposed_command, "wait 30"]

        if context.handler_name == "ShopPurchaseHandler":
            if proposed_command == "return":
                return ["return", "proceed"]
            return [proposed_command, "wait 30"]

        if context.handler_name == "CardRewardHandler":
            return [proposed_command, "wait 30"]

        if context.handler_name == "MapHandler":
            return [proposed_command]

        if context.handler_name == "CombatRewardHandler":
            if proposed_command.startswith("potion use ") or proposed_command.startswith("potion discard "):
                return ["wait 30", proposed_command]
            return [proposed_command]

        if context.handler_name == "BossRewardHandler":
            if proposed_command == "skip":
                return ["skip", "proceed"]
            return [proposed_command]

        if context.handler_name == "AstrolabeTransformHandler":
            return [proposed_command]

        if context.handler_name == "GridSelectHandler":
            if proposed_command in ("confirm", "proceed"):
                return [proposed_command]
            return [proposed_command, "wait 30"]

        if context.handler_name in {"CampfireHandler", "ChestHandler", "HandSelectHandler", "GenericHandler"}:
            return [proposed_command]

        return None

    def _uses_duplicate_choice_token(self, context: AgentContext, proposed_command: str) -> bool:
        tokens = proposed_command.split()
        if len(tokens) != 2 or tokens[0] != "choose":
            return False
        choice = tokens[1].strip().lower()
        if choice.isdigit():
            return False
        matches = [token for token in context.choice_list if str(token).strip().lower() == choice]
        return len(matches) > 1
