from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


PROMPT_TEMPLATE = (Path(__file__).resolve().parent / "prompts" / "map_decision_prompt.txt").read_text(
    encoding="utf-8"
)


@dataclass
class MapLlmProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MapDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class MapLlmProvider:
    def __init__(self, model: str | None = None, temperature: float = 0.6):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature

    def propose(self, context: AgentContext) -> MapLlmProposal:
        prompt = self._build_prompt(context)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as e:
            return MapLlmProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(e)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=MapDecisionSchema,
            temperature=self.temperature,
            cache_namespace="map_advisor_agent",
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = MapDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, MapDecisionSchema):
            return MapLlmProposal(
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

        return MapLlmProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or "llm_map_policy"),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
            },
        )

    def _build_prompt(self, context: AgentContext) -> str:
        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            choice_list=context.choice_list,
            floor=context.game_state.get("floor", "unknown"),
            act=context.game_state.get("act", "unknown"),
            current_hp=context.game_state.get("current_hp", "unknown"),
            max_hp=context.game_state.get("max_hp", "unknown"),
            gold=context.game_state.get("gold", "unknown"),
            room_type=context.game_state.get("room_type", "unknown"),
            character_class=context.game_state.get("character_class", "unknown"),
            ascension_level=context.game_state.get("ascension_level", "unknown"),
            act_boss=context.game_state.get("act_boss", "unknown"),
            current_position=context.game_state.get("current_position", "unknown"),
            run_memory_summary=context.extras.get("run_memory_summary", ""),
            recent_llm_decisions=context.extras.get("recent_llm_decisions", "none"),
            retrieved_episodic_memories=context.extras.get("retrieved_episodic_memories", "none"),
            retrieved_semantic_memories=context.extras.get("retrieved_semantic_memories", "none"),
            langmem_status=context.extras.get("langmem_status", "disabled_by_config"),
            relic_names=json.dumps(context.extras.get("relic_names", []), sort_keys=True),
            held_potion_names=json.dumps(context.extras.get("held_potion_names", []), sort_keys=True),
            potions_full=context.extras.get("potions_full", False),
            deck_profile=json.dumps(context.extras.get("deck_profile", {}), sort_keys=True),
            next_nodes=json.dumps(context.extras.get("next_nodes", []), sort_keys=True),
            boss_available=context.extras.get("boss_available", False),
            first_node_chosen=context.extras.get("first_node_chosen", False),
            deterministic_best_command=context.extras.get("deterministic_best_command", "unknown"),
            choice_path_overviews=json.dumps(context.extras.get("choice_path_overviews", []), sort_keys=True),
            sorted_path_summaries=json.dumps(context.extras.get("sorted_path_summaries", []), sort_keys=True),
        )
