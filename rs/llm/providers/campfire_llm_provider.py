from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Protocol

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "campfire_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class CampfireCommandProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CampfireDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class CampfireCommandProvider(Protocol):
    def propose(self, context: AgentContext, working_memory: dict[str, Any]) -> CampfireCommandProposal:
        ...


class CampfireLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 0.35):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext, working_memory: dict[str, Any]) -> CampfireCommandProposal:
        prompt = self._build_prompt(context, working_memory)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as exc:
            return CampfireCommandProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(exc)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=CampfireDecisionSchema,
            temperature=self.temperature,
            cache_namespace="campfire_subagent",
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = CampfireDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, CampfireDecisionSchema):
            return CampfireCommandProposal(
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

        confidence = response.confidence
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return CampfireCommandProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or "campfire_subagent"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(self, context: AgentContext, working_memory: dict[str, Any]) -> str:
        payload = {
            "handler_name": context.handler_name,
            "screen_type": context.screen_type,
            "available_commands": list(context.available_commands),
            "choice_list": list(context.choice_list),
            "game_state": dict(context.game_state),
            "campfire_options": context.extras.get("campfire_options", []),
            "campfire_option_flags": context.extras.get("campfire_option_flags", {}),
            "campfire_has_rested": context.extras.get("campfire_has_rested", False),
            "is_boss_rest_site": context.extras.get("is_boss_rest_site", False),
            "relic_names": context.extras.get("relic_names", []),
            "relic_counters": context.extras.get("relic_counters", {}),
            "deck_profile": context.extras.get("deck_profile", {}),
            "deck_card_entries": context.extras.get("deck_card_entries", []),
            "run_memory_summary": context.extras.get("run_memory_summary", ""),
            "recent_llm_decisions": context.extras.get("recent_llm_decisions", "none"),
            "retrieved_episodic_memories": context.extras.get("retrieved_episodic_memories", "none"),
            "retrieved_semantic_memories": context.extras.get("retrieved_semantic_memories", "none"),
            "working_memory": {
                "recent_step_summaries": list(working_memory.get("recent_step_summaries", []))[-6:],
                "executed_command_batches": [list(batch) for batch in working_memory.get("executed_command_batches", [])][-4:],
                "decision_loop_count": int(working_memory.get("decision_loop_count", 0)),
            },
        }
        return PROMPT_TEMPLATE.format(
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )
