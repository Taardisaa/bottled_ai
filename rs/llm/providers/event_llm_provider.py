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
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
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
        run_memory_summary = context.extras.get("run_memory_summary", "")
        recent_llm_decisions = context.extras.get("recent_llm_decisions", "none")
        memory_context_block = self._build_memory_context_block(
            context.extras.get("retrieved_episodic_memories", "none"),
            context.extras.get("retrieved_semantic_memories", "none"),
        )
        current_priorities = self._format_list_field(context.extras.get("current_priorities"), default="none")
        risk_flags = self._format_list_field(context.extras.get("risk_flags"), default="stable")
        deck_direction = str(context.extras.get("deck_direction", "unknown") or "unknown")
        event_options_text = self._format_event_options(
            context.game_state.get("event_options"),
            context.choice_list,
        )

        return PROMPT_TEMPLATE.format(
            event_options_text=event_options_text,
            event_name=event_name,
            floor=floor,
            act=act,
            current_hp=hp,
            max_hp=max_hp,
            gold=gold,
            run_memory_summary=run_memory_summary,
            recent_llm_decisions=recent_llm_decisions,
            memory_context_block=memory_context_block,
            current_priorities=current_priorities,
            risk_flags=risk_flags,
            deck_direction=deck_direction,
        )

    def _format_event_options(self, event_options: Any, choice_list: list[str]) -> str:
        if isinstance(event_options, list) and event_options:
            rendered_options: list[str] = []
            for fallback_index, option in enumerate(event_options):
                if not isinstance(option, dict):
                    continue
                choice_index = option.get("choice_index", fallback_index)
                label = str(option.get("label", "")).strip()
                text = str(option.get("text", "")).strip()
                disabled = bool(option.get("disabled", False))
                status = "disabled" if disabled else "enabled"
                rendered_options.append(
                    f"- {choice_index} | {status} | label=\"{label}\" | text=\"{text}\""
                )
            if rendered_options:
                return "\n".join(rendered_options)

        if choice_list:
            return "\n".join(f"- {index} | enabled | choice=\"{choice}\"" for index, choice in enumerate(choice_list))
        return "- none"

    def _format_list_field(self, value: Any, default: str) -> str:
        if isinstance(value, list):
            normalized_values = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(normalized_values) if normalized_values else default
        value_text = str(value or "").strip()
        return value_text if value_text else default

    def _build_memory_context_block(self, episodic: Any, semantic: Any) -> str:
        lines: list[str] = []
        for label, value in (
            ("Retrieved episodic memories", episodic),
            ("Retrieved semantic memories", semantic),
        ):
            value_text = str(value or "").strip()
            if value_text == "" or value_text.lower() == "none":
                continue
            lines.append(f"{label}: {value_text}")
        return "\n".join(lines)
