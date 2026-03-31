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

    def propose(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> MapLlmProposal:
        prompt = self._build_prompt(context, validation_feedback)
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

    def _build_prompt(
            self,
            context: AgentContext,
            validation_feedback: Dict[str, Any] | None = None,
    ) -> str:
        map_options_text = self._format_choice_list(context.choice_list)
        choice_branch_summaries_text = self._format_choice_branch_summaries(
            context.extras.get("choice_branch_summaries", []),
        )
        choice_representative_paths_text = self._format_choice_representative_paths(
            context.extras.get("choice_representative_paths", []),
        )
        episodic_memories = self._optional_memory_line(
            "Retrieved episodic memories",
            context.extras.get("retrieved_episodic_memories", "none"),
        )
        semantic_memories = self._optional_memory_line(
            "Retrieved semantic memories",
            context.extras.get("retrieved_semantic_memories", "none"),
        )
        validation_feedback_block = ""
        if validation_feedback is not None:
            validation_feedback_block = (
                "Validation feedback from previous rejected proposal:\n"
                f"{json.dumps(validation_feedback, sort_keys=True)}\n"
                "If feedback is present, correct the command to satisfy it.\n"
            )
        return PROMPT_TEMPLATE.format(
            map_options_text=map_options_text,
            floor=context.game_state.get("floor", "unknown"),
            act=context.game_state.get("act", "unknown"),
            current_hp=context.game_state.get("current_hp", "unknown"),
            max_hp=context.game_state.get("max_hp", "unknown"),
            gold=context.game_state.get("gold", "unknown"),
            character_class=context.game_state.get("character_class", "unknown"),
            ascension_level=context.game_state.get("ascension_level", "unknown"),
            act_boss=context.game_state.get("act_boss", "unknown"),
            run_memory_summary=context.extras.get("run_memory_summary", ""),
            recent_llm_decisions=context.extras.get("recent_llm_decisions", "none"),
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            relic_names=json.dumps(context.extras.get("relic_names", []), sort_keys=True),
            deck_profile=json.dumps(context.extras.get("deck_profile", {}), sort_keys=True),
            boss_available=context.extras.get("boss_available", False),
            first_node_chosen=context.extras.get("first_node_chosen", False),
            deterministic_best_command=context.extras.get("deterministic_best_command", "unknown"),
            choice_branch_summaries_text=choice_branch_summaries_text,
            choice_representative_paths_text=choice_representative_paths_text,
            validation_feedback_block=validation_feedback_block,
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
                        f"  - rooms={rooms} | "
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

    def _optional_memory_line(self, label: str, value: Any) -> str:
        normalized = str(value or "").strip()
        if normalized == "" or normalized.lower() == "none":
            return ""
        return f"{label}: {normalized}\n"
