from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "battle_meta_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class BattleMetaLlmProposal:
    comparator_profile: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BattleMetaDecisionSchema(BaseModel):
    comparator_profile: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class BattleMetaLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 0.3):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext) -> BattleMetaLlmProposal:
        prompt = self._build_prompt(context)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as e:
            return BattleMetaLlmProposal(
                comparator_profile=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(e)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=BattleMetaDecisionSchema,
            temperature=self.temperature,
            cache_namespace="battle_meta_advisor_agent",
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = BattleMetaDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, BattleMetaDecisionSchema):
            return BattleMetaLlmProposal(
                comparator_profile=None,
                confidence=0.0,
                explanation="llm_non_schema_response",
                metadata={"token_total": token_total},
            )

        comparator_profile = response.comparator_profile
        if comparator_profile is not None:
            comparator_profile = str(comparator_profile).strip().lower()
            if comparator_profile == "":
                comparator_profile = None

        try:
            confidence = float(response.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return BattleMetaLlmProposal(
            comparator_profile=comparator_profile,
            confidence=confidence,
            explanation=str(response.explanation or "llm_battle_meta_policy"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(self, context: AgentContext) -> str:
        memory_context_block = self._build_memory_context_block(
            context.extras.get("retrieved_episodic_memories", "none"),
            context.extras.get("retrieved_semantic_memories", "none"),
        )
        return PROMPT_TEMPLATE.format(
            floor=context.game_state.get("floor", "unknown"),
            act=context.game_state.get("act", "unknown"),
            room_type=context.game_state.get("room_type", "unknown"),
            character_class=context.game_state.get("character_class", "unknown"),
            ascension_level=context.game_state.get("ascension_level", "unknown"),
            current_hp=context.game_state.get("current_hp", "unknown"),
            max_hp=context.game_state.get("max_hp", "unknown"),
            turn=context.game_state.get("turn", "unknown"),
            player_block=context.game_state.get("player_block", "unknown"),
            player_energy=context.game_state.get("player_energy", "unknown"),
            run_memory_summary=context.extras.get("run_memory_summary", ""),
            recent_llm_decisions=context.extras.get("recent_llm_decisions", "none"),
            memory_context_block=memory_context_block,
            deterministic_profile=context.extras.get("deterministic_profile", "unknown"),
            available_profiles=json.dumps(context.extras.get("available_profiles", []), sort_keys=True),
            monster_summaries=json.dumps(context.extras.get("monster_summaries", []), sort_keys=True),
            player_power_summaries=json.dumps(context.extras.get("player_power_summaries", []), sort_keys=True),
            relic_names=json.dumps(context.extras.get("relic_names", []), sort_keys=True),
            deck_profile=json.dumps(self._compact_deck_profile(context.extras.get("deck_profile", {})), sort_keys=True),
        )

    def _build_memory_context_block(self, episodic: Any, semantic: Any) -> str:
        lines: list[str] = []
        for label, value in (
            ("Retrieved episodic memories", episodic),
            ("Retrieved semantic memories", semantic),
        ):
            value_text = str(value or "").strip()
            if value_text == "" or value_text.lower() == "none":
                continue
            lines.append(f"- {label}: {value_text}")
        return "\n".join(lines)

    def _compact_deck_profile(self, deck_profile: Any) -> dict[str, Any]:
        if not isinstance(deck_profile, dict):
            return {}
        compacted: dict[str, Any] = {}
        for key in ("total_cards", "type_counts", "upgraded_cards", "exhaust_cards"):
            if key in deck_profile:
                compacted[key] = deck_profile[key]
        return compacted
