from __future__ import annotations

from dataclasses import dataclass, field
import json
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
            two_layer_struct_convert=False,
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
        current_hp = context.game_state.get("current_hp", "unknown")
        max_hp = context.game_state.get("max_hp", "unknown")
        room_type = context.game_state.get("room_type", "unknown")
        character_class = context.game_state.get("character_class", "unknown")
        ascension_level = context.game_state.get("ascension_level", "unknown")
        act_boss = context.game_state.get("act_boss", "unknown")
        removable_curse = context.extras.get("has_removable_curse", False)
        deck_size = context.extras.get("deck_size", "unknown")
        deck_profile = context.extras.get("deck_profile", {})
        relic_names = context.extras.get("relic_names", [])
        held_potion_names = context.extras.get("held_potion_names", [])
        potions_full = context.extras.get("potions_full", False)
        run_memory_summary = context.extras.get("run_memory_summary", "")
        recent_llm_decisions = context.extras.get("recent_llm_decisions", "none")
        purge_cost = context.extras.get("purge_cost", "unknown")
        purge_available = context.extras.get("purge_available", False)
        offer_summaries = context.extras.get("offer_summaries", {})

        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            choice_list=context.choice_list,
            floor=floor,
            act=act,
            gold=gold,
            current_hp=current_hp,
            max_hp=max_hp,
            room_type=room_type,
            character_class=character_class,
            ascension_level=ascension_level,
            act_boss=act_boss,
            has_removable_curse=removable_curse,
            deck_size=deck_size,
            deck_profile=json.dumps(deck_profile, sort_keys=True),
            run_memory_summary=run_memory_summary,
            recent_llm_decisions=recent_llm_decisions,
            relic_names=json.dumps(relic_names, sort_keys=True),
            held_potion_names=json.dumps(held_potion_names, sort_keys=True),
            potions_full=potions_full,
            purge_cost=purge_cost,
            purge_available=purge_available,
            shop_card_offers=json.dumps(offer_summaries.get("cards", []), sort_keys=True),
            shop_relic_offers=json.dumps(offer_summaries.get("relics", []), sort_keys=True),
            shop_potion_offers=json.dumps(offer_summaries.get("potions", []), sort_keys=True),
        )
