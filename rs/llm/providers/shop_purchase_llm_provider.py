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

    def propose(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> ShopPurchaseLlmProposal:
        prompt = self._build_prompt(context, validation_feedback)
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
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
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

    def _build_prompt(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> str:
        floor = context.game_state.get("floor", "unknown")
        act = context.game_state.get("act", "unknown")
        gold = context.game_state.get("gold", "unknown")
        current_hp = context.game_state.get("current_hp", "unknown")
        max_hp = context.game_state.get("max_hp", "unknown")
        character_class = context.game_state.get("character_class", "unknown")
        ascension_level = context.game_state.get("ascension_level", "unknown")
        act_boss = context.game_state.get("act_boss", "unknown")
        removable_curse = context.extras.get("has_removable_curse", False)
        deck_size = context.extras.get("deck_size", "unknown")
        deck_profile = self._compact_deck_profile(context.extras.get("deck_profile", {}))
        relic_names = context.extras.get("relic_names", [])
        run_memory_summary = context.extras.get("run_memory_summary", "")
        recent_llm_decisions = context.extras.get("recent_llm_decisions", "none")
        memory_context_block = self._build_memory_context_block(
            context.extras.get("retrieved_episodic_memories", "none"),
            context.extras.get("retrieved_semantic_memories", "none"),
        )
        purge_cost = context.extras.get("purge_cost", "unknown")
        purge_available = context.extras.get("purge_available", False)
        offer_summaries = context.extras.get("offer_summaries", {})
        shop_options_text = self._format_choice_list(context.choice_list)
        potion_context_lines = self._build_potion_context_lines(
            offer_summaries.get("potions", []),
            context.extras.get("held_potion_names", []),
            context.extras.get("potions_full", False),
        )
        validation_feedback_block = ""
        if validation_feedback is not None:
            validation_feedback_block = (
                "Validation feedback from previous rejected proposal:\n"
                f"{json.dumps(validation_feedback, sort_keys=True)}\n"
                "If feedback is present, correct the command to satisfy it.\n"
            )

        return PROMPT_TEMPLATE.format(
            shop_options_text=shop_options_text,
            floor=floor,
            act=act,
            gold=gold,
            current_hp=current_hp,
            max_hp=max_hp,
            character_class=character_class,
            ascension_level=ascension_level,
            act_boss=act_boss,
            has_removable_curse=removable_curse,
            deck_size=deck_size,
            deck_profile=json.dumps(deck_profile, sort_keys=True),
            run_memory_summary=run_memory_summary,
            recent_llm_decisions=recent_llm_decisions,
            memory_context_block=memory_context_block,
            relic_names=json.dumps(relic_names, sort_keys=True),
            potion_context_lines=potion_context_lines,
            purge_cost=purge_cost,
            purge_available=purge_available,
            shop_card_offers=json.dumps(offer_summaries.get("cards", []), sort_keys=True),
            shop_relic_offers=json.dumps(offer_summaries.get("relics", []), sort_keys=True),
            shop_potion_offers=json.dumps(offer_summaries.get("potions", []), sort_keys=True),
            validation_feedback_block=validation_feedback_block,
        )

    def _format_choice_list(self, choice_list: list[Any]) -> str:
        if not choice_list:
            return "- none"
        return "\n".join(f"- {index} | option=\"{str(choice).strip()}\"" for index, choice in enumerate(choice_list))

    def _compact_deck_profile(self, deck_profile: Any) -> dict[str, Any]:
        if not isinstance(deck_profile, dict):
            return {}
        compacted: dict[str, Any] = {}
        for key in ("total_cards", "type_counts", "upgraded_cards", "exhaust_cards"):
            if key in deck_profile:
                compacted[key] = deck_profile[key]
        return compacted

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

    def _build_potion_context_lines(self, potion_offers: Any, held_potion_names: Any, potions_full: Any) -> str:
        if not isinstance(potion_offers, list) or not potion_offers:
            return ""
        return (
            f"Held potions: {json.dumps(held_potion_names or [], sort_keys=True)}\n"
            f"Potion slots full: {bool(potions_full)}\n"
        )
