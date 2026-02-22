from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "shop_purchase_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class ShopPurchaseLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShopPurchaseDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class ShopPurchaseLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 1.0):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext) -> ShopPurchaseLlmProposal:
        prompt = self._build_prompt(context)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as e:
            return ShopPurchaseLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(e)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=ShopPurchaseDecisionSchema,
            temperature=self.temperature,
            cache_namespace="shop_purchase_advisor_agent",
        )

        if isinstance(response, dict):
            try:
                response = ShopPurchaseDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, ShopPurchaseDecisionSchema):
            return ShopPurchaseLlmProposal(
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

        return ShopPurchaseLlmProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or "llm_shop_purchase_policy"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(self, context: AgentContext) -> str:
        floor = context.game_state.get("floor", "unknown")
        act = context.game_state.get("act", "unknown")
        gold = context.game_state.get("gold", "unknown")
        removable_curse = context.extras.get("has_removable_curse", False)
        deck_size = context.extras.get("deck_size", "unknown")

        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            choice_list=context.choice_list,
            floor=floor,
            act=act,
            gold=gold,
            has_removable_curse=removable_curse,
            deck_size=deck_size,
        )
