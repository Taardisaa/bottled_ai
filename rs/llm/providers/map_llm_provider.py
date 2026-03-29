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
        langmem_status = self._normalize_langmem_status(context.extras.get("langmem_status", "disabled_by_config"))
        current_priorities = self._format_list_field(context.extras.get("current_priorities"), default="none")
        risk_flags = self._format_list_field(context.extras.get("risk_flags"), default="stable")
        deck_direction = str(context.extras.get("deck_direction", "unknown") or "unknown")
        run_hypotheses = self._format_list_field(context.extras.get("run_hypotheses"), default="none")
        map_options_text = self._format_choice_list(context.choice_list)
        choice_branch_summaries_text = self._format_choice_branch_summaries(
            context.extras.get("choice_branch_summaries", []),
        )
        choice_representative_paths_text = self._format_choice_representative_paths(
            context.extras.get("choice_representative_paths", []),
        )
        return PROMPT_TEMPLATE.format(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=context.available_commands,
            map_options_text=map_options_text,
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
            langmem_status=langmem_status,
            current_priorities=current_priorities,
            risk_flags=risk_flags,
            deck_direction=deck_direction,
            run_hypotheses=run_hypotheses,
            relic_names=json.dumps(context.extras.get("relic_names", []), sort_keys=True),
            held_potion_names=json.dumps(context.extras.get("held_potion_names", []), sort_keys=True),
            potions_full=context.extras.get("potions_full", False),
            deck_profile=json.dumps(context.extras.get("deck_profile", {}), sort_keys=True),
            boss_available=context.extras.get("boss_available", False),
            first_node_chosen=context.extras.get("first_node_chosen", False),
            deterministic_best_command=context.extras.get("deterministic_best_command", "unknown"),
            choice_branch_summaries_text=choice_branch_summaries_text,
            choice_representative_paths_text=choice_representative_paths_text,
        )

    def _format_choice_list(self, choice_list: list[Any]) -> str:
        if not choice_list:
            return "- none"
        return "\n".join(f"- {index} | route=\"{str(choice).strip()}\"" for index, choice in enumerate(choice_list))

    def _normalize_langmem_status(self, status: Any) -> str:
        status_text = str(status or "").strip().lower()
        if status_text == "" or status_text == "disabled_by_config":
            return "disabled"
        if "unavailable" in status_text or "error" in status_text or "failed" in status_text:
            return "unavailable"
        return "ready"

    def _format_list_field(self, value: Any, default: str) -> str:
        if isinstance(value, list):
            normalized_values = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(normalized_values) if normalized_values else default
        value_text = str(value or "").strip()
        return value_text if value_text else default

    def _format_choice_branch_summaries(self, branch_summaries: Any) -> str:
        if not isinstance(branch_summaries, list) or not branch_summaries:
            return "- none"

        rendered_summaries: list[str] = []
        for summary in branch_summaries:
            if not isinstance(summary, dict):
                continue
            rendered_summaries.append(
                (
                    f"- {summary.get('choice_index', '?')} | {summary.get('choice_label', 'unknown')} | "
                    f"{summary.get('choice_command', 'unknown')} | paths={summary.get('path_count', 0)} | "
                    f"monsters={self._format_range(summary.get('monster_count_range'))} | "
                    f"events={self._format_range(summary.get('event_count_range'))} | "
                    f"shops={self._format_range(summary.get('shop_count_range'))} | "
                    f"campfires={self._format_range(summary.get('campfire_count_range'))} | "
                    f"elites={self._format_range(summary.get('elite_count_range'))} | "
                    f"treasure={self._format_range(summary.get('treasure_count_range'))} | "
                    f"first_shop={self._format_range(summary.get('first_shop_distance_range'))} | "
                    f"first_campfire={self._format_range(summary.get('first_campfire_distance_range'))} | "
                    f"first_elite={self._format_range(summary.get('first_elite_distance_range'))} | "
                    f"shape={summary.get('branch_shape_summary', 'unknown')}"
                )
            )
        return "\n".join(rendered_summaries) if rendered_summaries else "- none"

    def _format_choice_representative_paths(self, representative_groups: Any) -> str:
        if not isinstance(representative_groups, list) or not representative_groups:
            return "- none"

        rendered_groups: list[str] = []
        for group in representative_groups:
            if not isinstance(group, dict):
                continue
            representative_paths = group.get("representative_paths", [])
            if not isinstance(representative_paths, list) or not representative_paths:
                continue
            rendered_groups.append(
                f"- {group.get('choice_index', '?')} | {group.get('choice_label', 'unknown')} | {group.get('choice_command', 'unknown')}:"
            )
            for path in representative_paths:
                if not isinstance(path, dict):
                    continue
                rooms = " > ".join(str(room) for room in path.get("rooms", [])) or "none"
                rendered_groups.append(
                    (
                        f"  - rooms={rooms} | room_counts={json.dumps(path.get('room_counts', {}), sort_keys=True)} | "
                        f"path_length={path.get('path_length', 'unknown')} | "
                        f"first_shop={path.get('first_shop_distance', 'none')} | "
                        f"first_campfire={path.get('first_campfire_distance', 'none')} | "
                        f"first_elite={path.get('first_elite_distance', 'none')}"
                    )
                )
        return "\n".join(rendered_groups) if rendered_groups else "- none"

    def _format_range(self, value: Any) -> str:
        if not isinstance(value, dict):
            return "none"
        if "min" not in value or "max" not in value:
            return "none"
        if value["min"] == value["max"]:
            return str(value["min"])
        return f"{value['min']}-{value['max']}"
