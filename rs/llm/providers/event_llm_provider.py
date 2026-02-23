from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "event_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class EventLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class EventLlmProvider:
    """Model-backed event decision provider.

    This provider delegates to `rs.utils.llm_utils.ask_llm_once` and expects a
    structured EventDecisionSchema response.
    """

    def __init__(self, model: str | None = None, temperature: float = 1.0):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext) -> EventLlmProposal:
        prompt = self._build_prompt(context)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as e:
            return EventLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(e)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=EventDecisionSchema,
            temperature=self.temperature,
            cache_namespace="event_advisor_agent",
            two_layer_struct_convert=False,
        )

        if isinstance(response, dict):
            try:
                response = EventDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, EventDecisionSchema):
            return EventLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_non_schema_response",
                metadata={"token_total": token_total},
            )

        proposed_command = response.proposed_command
        if proposed_command is not None:
            proposed_command = str(proposed_command).strip()
            if proposed_command == "":
                proposed_command = None

        confidence_raw = response.confidence
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        explanation = str(response.explanation or "llm_event_policy")
        metadata = {
            "token_total": token_total,
            "provider": "llm_utils.ask_llm_once",
            "model": self.model,
        }

        return EventLlmProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=explanation,
            metadata=metadata,
        )

    def _build_prompt(self, context: AgentContext) -> str:
        event_name = context.game_state.get("event_name", "unknown")
        floor = context.game_state.get("floor", "unknown")
        act = context.game_state.get("act", "unknown")
        hp = context.game_state.get("current_hp", "unknown")
        max_hp = context.game_state.get("max_hp", "unknown")
        gold = context.game_state.get("gold", "unknown")

        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            choice_list=context.choice_list,
            event_name=event_name,
            floor=floor,
            act=act,
            current_hp=hp,
            max_hp=max_hp,
            gold=gold,
            extras=context.extras,
        )
