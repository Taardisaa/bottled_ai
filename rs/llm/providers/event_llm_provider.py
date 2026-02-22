from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Dict

pydantic_module = None
try:
    pydantic_module = importlib.import_module("pydantic")
    HAS_PYDANTIC = True
except Exception:
    HAS_PYDANTIC = False

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


@dataclass
class EventLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventLlmProvider:
    """Model-backed event decision provider.

    This provider delegates to `rs.utils.llm_utils.ask_llm_once` and expects a
    structured dictionary response with at least `proposed_command`.
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

        struct_param: Any = dict
        if HAS_PYDANTIC and pydantic_module is not None:
            struct_param = pydantic_module.create_model(
                "EventDecisionSchema",
                proposed_command=(str | None, None),
                confidence=(float, 0.0),
                explanation=(str, ""),
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=struct_param,
            temperature=self.temperature,
            cache_namespace="event_advisor_agent",
        )

        if response is not None and hasattr(response, "model_dump"):
            response = response.model_dump()

        if not isinstance(response, dict):
            return EventLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_non_dict_response",
                metadata={"token_total": token_total},
            )

        proposed_command = response.get("proposed_command")
        if proposed_command is not None:
            proposed_command = str(proposed_command).strip()
            if proposed_command == "":
                proposed_command = None

        confidence_raw = response.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        explanation = str(response.get("explanation", "llm_event_policy"))
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

        return (
            "You are selecting a Slay the Spire event command. "
            "Return ONLY a JSON object with keys: "
            "proposed_command (string or null), confidence (0..1), explanation (short string).\n"
            "Rules:\n"
            "- proposed_command must be one command from available_commands.\n"
            "- Prefer safe conservative choices when uncertain.\n"
            "- If uncertain, set proposed_command to null and low confidence.\n"
            f"Handler: {context.handler_name}\n"
            f"Screen: {context.screen_type}\n"
            f"Available commands: {context.available_commands}\n"
            f"Choices: {context.choice_list}\n"
            f"Event: {event_name}\n"
            f"Floor: {floor}, Act: {act}, HP: {hp}/{max_hp}, Gold: {gold}\n"
            f"Extras: {context.extras}\n"
        )
