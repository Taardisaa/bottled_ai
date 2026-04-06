from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rs.calculator.executor import get_best_battle_action
from rs.calculator.interfaces.comparator_interface import ComparatorInterface
from rs.llm.agents.base_agent import AgentContext
from rs.llm.langmem_service import LangMemService
from rs.llm.validator import validate_command_batch
from rs.machine.state import GameState


def _state_from_context(context: AgentContext) -> GameState:
    state = context.extras.get("game_state_ref")
    if not isinstance(state, GameState):
        raise ValueError("battle context missing GameState reference")
    return state


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


class LowestEnemyHpComparator(ComparatorInterface):
    def does_challenger_defeat_the_best(self, best, challenger, original) -> bool:
        def _score(candidate: Any) -> tuple[int, int, int, int]:
            total_enemy_hp = sum(max(0, monster.current_hp) for monster in candidate.monsters if not monster.is_gone)
            live_enemy_count = sum(1 for monster in candidate.monsters if not monster.is_gone and monster.current_hp > 0)
            player_hp = max(0, candidate.player.current_hp)
            player_block = max(0, candidate.player.block)
            return total_enemy_hp, live_enemy_count, -player_hp, -player_block

        return _score(challenger) < _score(best)


# ---------------------------------------------------------------------------
# Implementation classes — called directly by the guardrail fallback.
# Not part of the public LangGraph tool API; use the @tool functions below.
# ---------------------------------------------------------------------------

class EnumerateLegalActionsTool:
    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        state = _state_from_context(context)
        commands: list[str] = []
        categories: dict[str, list[str]] = {"play": [], "choose": [], "potion": [], "other": []}

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


class AnalyzeWithCalculatorTool:
    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        max_path_count = int(payload.get("max_path_count", 250))
        if max_path_count < 1:
            max_path_count = 1
        try:
            action = get_best_battle_action(
                _state_from_context(context), LowestEnemyHpComparator(), max_path_count=max_path_count
            )
        except Exception as exc:
            return {"recommended_commands": [], "summary": "calculator_failed", "error": str(exc)}
        commands = list(action.commands) if action is not None else []
        return {
            "recommended_commands": commands,
            "summary": "calculator_recommendation" if commands else "calculator_no_action",
        }


class ValidateBattleCommandTool:
    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        if "commands" in payload and isinstance(payload["commands"], list):
            commands = [str(c) for c in payload["commands"]]
        elif "command" in payload:
            commands = [str(payload["command"])]
        else:
            commands = []
        return validate_command_batch(context, commands, mode="battle")


# ---------------------------------------------------------------------------
# LangGraph @tool functions — bound to ChatModel via bind_tools() and
# dispatched by ToolNode. The `state` argument is injected automatically by
# ToolNode; the LLM never provides it.
# ---------------------------------------------------------------------------

@tool
def enumerate_legal_actions(state: Annotated[dict, InjectedState] = None) -> dict:
    """List currently legal battle commands for the exact current state.
    Returns commands grouped by category (play, choose, potion, other)."""
    context = (state or {}).get("current_context")
    if context is None:
        return {"commands": [], "categories": {}, "summary": "no_context"}
    return EnumerateLegalActionsTool().run(context, {})


@tool
def analyze_with_calculator(max_path_count: int = 250, state: Annotated[dict, InjectedState] = None) -> dict:
    """Ask the rs.calculator engine for a recommended command batch.
    Returns recommended_commands list. Use max_path_count to limit search depth."""
    context = (state or {}).get("current_context")
    if context is None:
        return {"recommended_commands": [], "summary": "no_context"}
    return AnalyzeWithCalculatorTool().run(context, {"max_path_count": max_path_count})


@tool
def validate_battle_command(commands: list[str], state: Annotated[dict, InjectedState] = None) -> dict:
    """Validate a command batch against the current battle state before submitting.
    Returns is_valid and error details."""
    context = (state or {}).get("current_context")
    if context is None:
        return {"is_valid": False, "errors": ["no_context"]}
    return ValidateBattleCommandTool().run(context, {"commands": commands})


@tool
def submit_battle_commands(commands: list[str], reasoning: str = "") -> str:
    """Submit the final command batch to execute in battle.
    Call this once you are confident in your decision. Do not call other tools after this.

    Args:
        commands: The command(s) to execute.
        reasoning: Brief explanation of your decision (what you observed, what you chose, and why).
    """
    return "submitted"


def make_retrieve_battle_experience_tool(langmem_service: LangMemService):
    """Factory that creates a retrieve_battle_experience tool closed over the given LangMemService."""

    @tool
    def retrieve_battle_experience(state: Annotated[dict, InjectedState] = None) -> dict:
        """Retrieve relevant LangMem episodic and semantic battle memories.
        Call this if you need more context about past battles."""
        context = (state or {}).get("current_context")
        if context is None:
            return {
                "retrieved_episodic_memories": "none",
                "retrieved_semantic_memories": "none",
                "summary": "no_context",
            }
        result = langmem_service.build_context_memory(context)
        result["summary"] = "retrieved battle experience"
        return result

    return retrieve_battle_experience
