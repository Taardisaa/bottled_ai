from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "card_reward_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class CardRewardLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CardRewardDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class CardRewardLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 1.0):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext) -> CardRewardLlmProposal:
        prompt = self._build_prompt(context)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as e:
            return CardRewardLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(e)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=CardRewardDecisionSchema,
            temperature=self.temperature,
            cache_namespace="card_reward_advisor_agent",
        )

        if isinstance(response, dict):
            try:
                response = CardRewardDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, CardRewardDecisionSchema):
            return CardRewardLlmProposal(
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

        return CardRewardLlmProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or "llm_card_reward_policy"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(self, context: AgentContext) -> str:
        floor = context.game_state.get("floor", "unknown")
        act = context.game_state.get("act", "unknown")
        hp = context.game_state.get("current_hp", "unknown")
        max_hp = context.game_state.get("max_hp", "unknown")
        room_phase = context.game_state.get("room_phase", "unknown")
        deck_size = context.extras.get("deck_size", "unknown")
        relic_names = context.extras.get("relic_names", [])

        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            choice_list=context.choice_list,
            room_phase=room_phase,
            floor=floor,
            act=act,
            current_hp=hp,
            max_hp=max_hp,
            deck_size=deck_size,
            relic_names=relic_names,
        )
