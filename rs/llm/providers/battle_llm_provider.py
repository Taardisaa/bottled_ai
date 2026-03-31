from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Protocol

from pydantic import BaseModel, Field, field_validator

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


_PROMPT_TEMPLATE = """
You are the battle subagent for a Slay the Spire bot.

You are controlling exactly one battle session. Decide the next best step for the current text-only state.
You may either:
1. request exactly one tool call, or
2. return the executable command batch for the current screen.

Rules:
- Use tools only when they materially help.
- Do not invent unavailable commands.
- For play commands, use the protocol form "play <hand_index> [target_index]".
- For hand-select or grid screens, use "choose <index>" and include "confirm" / "wait 30" when needed.
- Return only one tool call per response.
- Prefer short, factual explanations.

Available tools:
{tool_descriptions}

Current battle payload:
{battle_payload}
""".strip()


@dataclass
class BattleDirective:
    mode: str
    explanation: str
    confidence: float
    tool_name: str | None = None
    tool_payload: Dict[str, Any] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BattleDirectiveSchema(BaseModel):
    mode: str = "action"
    explanation: str = ""
    confidence: float = 0.0
    tool_name: str | None = None
    tool_payload: dict[str, Any] = Field(default_factory=dict)
    commands: list[str] = Field(default_factory=list)

    @field_validator("tool_payload", mode="before")
    @classmethod
    def _normalize_tool_payload(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return {}


class BattleDirectiveProvider(Protocol):
    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            tool_descriptions: dict[str, str],
            validation_feedback: dict[str, Any] | None = None,
    ) -> BattleDirective:
        ...


class BattleLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            tool_descriptions: dict[str, str],
            validation_feedback: dict[str, Any] | None = None,
    ) -> BattleDirective:
        prompt = self._build_prompt(context, working_memory, tool_descriptions, validation_feedback)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as exc:
            return BattleDirective(
                mode="stop",
                explanation="llm_utils_unavailable",
                confidence=0.0,
                metadata={"provider_error": str(exc)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=BattleDirectiveSchema,
            temperature=self.temperature,
            cache_namespace="battle_subagent",
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = BattleDirectiveSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, BattleDirectiveSchema):
            return BattleDirective(
                mode="stop",
                explanation="llm_non_schema_response",
                confidence=0.0,
                metadata={"token_total": token_total},
            )

        confidence = response.confidence
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        mode = str(response.mode or "action").strip().lower()
        if mode not in {"tool", "action", "stop"}:
            mode = "stop"

        commands = [str(command).strip() for command in response.commands if str(command).strip()]
        tool_name = str(response.tool_name).strip() if response.tool_name is not None else None
        if tool_name == "":
            tool_name = None

        return BattleDirective(
            mode=mode,
            explanation=str(response.explanation or "battle_subagent_step"),
            confidence=confidence,
            tool_name=tool_name,
            tool_payload=dict(response.tool_payload or {}),
            commands=commands,
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            tool_descriptions: dict[str, str],
            validation_feedback: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "handler_name": context.handler_name,
            "screen_type": context.screen_type,
            "available_commands": context.available_commands,
            "choice_list": context.choice_list,
            "game_state": context.game_state,
            "battle_state": {
                "hand_cards": context.extras.get("hand_cards", []),
                "selection_cards": context.extras.get("selection_cards", []),
                "monster_summaries": context.extras.get("monster_summaries", []),
                "potion_summaries": context.extras.get("potion_summaries", []),
                "player_energy": context.extras.get("player_energy"),
                "player_block": context.extras.get("player_block"),
                "player_powers": context.extras.get("player_powers", []),
            },
            "run_memory_summary": context.extras.get("run_memory_summary", ""),
            "retrieved_episodic_memories": context.extras.get("retrieved_episodic_memories", "none"),
            "retrieved_semantic_memories": context.extras.get("retrieved_semantic_memories", "none"),
            "battle_working_memory": {
                "recent_step_summaries": working_memory.get("recent_step_summaries", [])[-6:],
                "recent_tool_results": working_memory.get("recent_tool_results", [])[-4:],
                "executed_command_batches": working_memory.get("executed_command_batches", [])[-4:],
            },
            "validation_feedback": dict(validation_feedback or {}),
        }
        return _PROMPT_TEMPLATE.format(
            tool_descriptions=json.dumps(tool_descriptions, ensure_ascii=False, indent=2, sort_keys=True),
            battle_payload=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
