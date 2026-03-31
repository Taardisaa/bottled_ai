from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "generic_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class GenericLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenericDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class GenericLlmProvider:
    """Model-backed generic decision provider for unhandled states."""

    def __init__(self, model: str | None = None, temperature: float = 1.0):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> GenericLlmProposal:
        prompt = self._build_prompt(context, validation_feedback)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as error:
            return GenericLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(error)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=GenericDecisionSchema,
            temperature=self.temperature,
            cache_namespace="generic_advisor_agent",
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = GenericDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, GenericDecisionSchema):
            return GenericLlmProposal(
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

        try:
            confidence = float(response.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return GenericLlmProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or "llm_generic_policy"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> str:
        generic_payload = context.extras.get("generic_payload", {})
        sectioned_explanations = context.extras.get("sectioned_schema_explanations", {})
        field_dictionary = context.extras.get("field_dictionary", {})
        raw_game_state_keys = context.extras.get("raw_game_state_keys", [])
        validation_feedback_block = ""
        if validation_feedback is not None:
            validation_feedback_block = (
                "## Validation Feedback From Previous Rejected Proposal\n"
                f"{self._to_pretty_json(validation_feedback)}\n\n"
                "If validation feedback is not null, correct the previous command "
                "and return a command that satisfies the feedback constraints.\n"
            )

        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands_json=self._to_pretty_json(context.available_commands),
            choice_list_json=self._to_pretty_json(context.choice_list),
            game_state_summary_json=self._to_pretty_json(context.game_state),
            generic_payload_json=self._to_pretty_json(generic_payload),
            sectioned_schema_explanations_json=self._to_pretty_json(sectioned_explanations),
            field_dictionary_json=self._to_pretty_json(field_dictionary),
            raw_game_state_keys_json=self._to_pretty_json(raw_game_state_keys),
            validation_feedback_block=validation_feedback_block,
        )

    def _to_pretty_json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)
