from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Protocol

from pydantic import BaseModel

from rs.llm.agents.base_agent import AgentContext
from rs.utils.config import config


COMBAT_REWARD_PROMPT = (Path(__file__).resolve().parent / "prompts" / "combat_reward_decision_prompt.txt").read_text(
    encoding="utf-8"
)
BOSS_REWARD_PROMPT = (Path(__file__).resolve().parent / "prompts" / "boss_reward_decision_prompt.txt").read_text(
    encoding="utf-8"
)
ASTROLABE_TRANSFORM_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "astrolabe_transform_decision_prompt.txt"
).read_text(encoding="utf-8")
GRID_SELECT_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "grid_select_decision_prompt.txt"
).read_text(encoding="utf-8")


@dataclass
class RewardCommandProposal:
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RewardDecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class RewardCommandProvider(Protocol):
    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> RewardCommandProposal:
        ...


class BaseRewardLlmProvider:
    def __init__(
            self,
            *,
            model: str | None = None,
            temperature: float = 0.4,
            prompt_template: str,
            cache_namespace: str,
            provider_name: str,
    ):
        self.model = config.fast_llm_model if model is None else model
        self.temperature = temperature
        self._prompt_template = prompt_template
        self._cache_namespace = cache_namespace
        self._provider_name = provider_name

    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> RewardCommandProposal:
        prompt = self._build_prompt(context, working_memory, validation_feedback)
        try:
            from rs.utils.llm_utils import ask_llm_once
        except Exception as exc:
            return RewardCommandProposal(
                proposed_command=None,
                confidence=0.0,
                explanation="llm_utils_unavailable",
                metadata={"provider_error": str(exc)},
            )

        response, token_total = ask_llm_once(
            message=prompt,
            model=self.model,
            struct=RewardDecisionSchema,
            temperature=self.temperature,
            cache_namespace=self._cache_namespace,
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
        )

        if isinstance(response, dict):
            try:
                response = RewardDecisionSchema.model_validate(response)
            except Exception:
                response = None

        if not isinstance(response, RewardDecisionSchema):
            return RewardCommandProposal(
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

        return RewardCommandProposal(
            proposed_command=proposed_command,
            confidence=confidence,
            explanation=str(response.explanation or self._provider_name),
            metadata={
                "token_total": token_total,
                "provider": "llm_utils.ask_llm_once",
                "model": self.model,
                "provider_name": self._provider_name,
            },
        )

    def _build_prompt(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> str:
        payload = self._build_payload(context, working_memory, validation_feedback)
        return self._prompt_template.format(
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )

    def _build_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise Exception("must be implemented by children")

    def _base_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "handler_name": context.handler_name,
            "screen_type": context.screen_type,
            "available_commands": list(context.available_commands),
            "choice_list": list(context.choice_list),
            "choice_token_counts": self._choice_token_counts(context.choice_list),
            "game_state": dict(context.game_state),
            "run_memory_summary": context.extras.get("run_memory_summary", ""),
            "recent_llm_decisions": context.extras.get("recent_llm_decisions", "none"),
            "retrieved_episodic_memories": context.extras.get("retrieved_episodic_memories", "none"),
            "retrieved_semantic_memories": context.extras.get("retrieved_semantic_memories", "none"),
            "working_memory": {
                "recent_step_summaries": list(working_memory.get("recent_step_summaries", []))[-6:],
                "executed_command_batches": [list(batch) for batch in working_memory.get("executed_command_batches", [])][-4:],
                "decision_loop_count": int(working_memory.get("decision_loop_count", 0)),
            },
            "validation_feedback": dict(validation_feedback or {}),
        }

    def _choice_token_counts(self, choice_list: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for raw_choice in choice_list:
            choice = str(raw_choice).strip().lower()
            if choice == "":
                continue
            counts[choice] = counts.get(choice, 0) + 1
        return counts


class CombatRewardLlmProvider(BaseRewardLlmProvider):
    def __init__(self, model: str | None = None, temperature: float = 0.4):
        super().__init__(
            model=model,
            temperature=temperature,
            prompt_template=COMBAT_REWARD_PROMPT,
            cache_namespace="combat_reward_subagent",
            provider_name="combat_reward_subagent",
        )

    def _build_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(context, working_memory, validation_feedback)
        llm_choice_list = context.extras.get("llm_choice_list")
        if isinstance(llm_choice_list, list):
            payload["choice_list"] = [str(choice) for choice in llm_choice_list]
            payload["choice_token_counts"] = self._choice_token_counts(payload["choice_list"])
        payload.update({
            "reward_summaries": context.extras.get("reward_summaries", []),
            "held_potion_names": context.extras.get("held_potion_names", []),
            "reward_potion_names": context.extras.get("reward_potion_names", []),
            "potions_full": context.extras.get("potions_full", False),
            "deck_profile": context.extras.get("deck_profile", {}),
            "relic_names": context.extras.get("relic_names", []),
            "has_card_reward_row": context.extras.get("has_card_reward_row", False),
            "non_card_reward_count": context.extras.get("non_card_reward_count", 0),
        })
        return payload

    def _build_prompt(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> str:
        payload = self._build_payload(context, working_memory, validation_feedback)
        structured_sections_text = self._build_structured_sections(context, payload)

        prompt = self._prompt_template.format(
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return f"{structured_sections_text}\n\n{prompt}"

    def _build_structured_sections(self, context: AgentContext, payload: dict[str, Any]) -> str:
        sections: list[str] = ["CombatReward structured context:"]
        sections.extend(self._run_snapshot_lines(context, payload))
        sections.append("Reward rows:")
        sections.extend(self._reward_row_lines(payload))
        sections.extend(self._command_availability_lines(payload))
        sections.extend(self._potion_capacity_lines(payload))
        sections.extend(self._deterministic_handoff_lines(payload))
        return "\n".join(sections)

    def _run_snapshot_lines(self, context: AgentContext, payload: dict[str, Any]) -> list[str]:
        floor = context.game_state.get("floor", "unknown")
        act = context.game_state.get("act", "unknown")
        hp = context.game_state.get("current_hp", "unknown")
        max_hp = context.game_state.get("max_hp", "unknown")
        deck_profile = payload.get("deck_profile", {})
        relic_names = payload.get("relic_names", [])
        return [
            f"- Run snapshot: floor={floor}, act={act}, hp={hp}/{max_hp}",
            f"- Deck profile: {json.dumps(deck_profile, ensure_ascii=False, sort_keys=True)}",
            f"- Relics: {json.dumps(relic_names, ensure_ascii=False)}",
        ]

    def _reward_row_lines(self, payload: dict[str, Any]) -> list[str]:
        reward_rows = payload.get("reward_summaries", [])
        lines: list[str] = []
        for row in reward_rows if isinstance(reward_rows, list) else []:
            if not isinstance(row, dict):
                continue
            summary_line = str(row.get("reward_summary_line", "")).strip()
            if summary_line != "":
                lines.append(f"- {summary_line}")
                continue
            idx = row.get("choice_index", "?")
            token = row.get("choice_token", "")
            reward_type = row.get("reward_type", "UNKNOWN")
            lines.append(f"- idx={idx} token='{token}' type={reward_type}")
        if not lines:
            return ["- none"]
        return lines

    def _command_availability_lines(self, payload: dict[str, Any]) -> list[str]:
        available_commands = payload.get("available_commands", [])
        choice_token_counts = payload.get("choice_token_counts", {})
        return [
            "Command availability:",
            f"- available_commands={json.dumps(available_commands, ensure_ascii=False)}",
            f"- choice_token_counts={json.dumps(choice_token_counts, ensure_ascii=False, sort_keys=True)}",
        ]

    def _potion_capacity_lines(self, payload: dict[str, Any]) -> list[str]:
        return [
            "Potion capacity:",
            f"- potions_full={bool(payload.get('potions_full', False))}",
            f"- held_potion_names={json.dumps(payload.get('held_potion_names', []), ensure_ascii=False)}",
            f"- reward_potion_names={json.dumps(payload.get('reward_potion_names', []), ensure_ascii=False)}",
        ]

    def _deterministic_handoff_lines(self, payload: dict[str, Any]) -> list[str]:
        return [
            "Deterministic handoff:",
            f"- has_card_reward_row={bool(payload.get('has_card_reward_row', False))}",
            f"- non_card_reward_count={int(payload.get('non_card_reward_count', 0))}",
        ]


class BossRewardLlmProvider(BaseRewardLlmProvider):
    def __init__(self, model: str | None = None, temperature: float = 0.35):
        super().__init__(
            model=model,
            temperature=temperature,
            prompt_template=BOSS_REWARD_PROMPT,
            cache_namespace="boss_reward_subagent",
            provider_name="boss_reward_subagent",
        )

    def _build_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(context, working_memory, validation_feedback)
        payload.update({
            "boss_relic_options": context.extras.get("boss_relic_options", []),
            "choice_metadata_mismatch": context.extras.get("choice_metadata_mismatch", False),
            "skip_available": context.extras.get("skip_available", False),
            "deck_profile": context.extras.get("deck_profile", {}),
            "deck_card_entries": context.extras.get("deck_card_entries", []),
            "relic_names": context.extras.get("relic_names", []),
            "held_potion_names": context.extras.get("held_potion_names", []),
        })
        return payload


class AstrolabeTransformLlmProvider(BaseRewardLlmProvider):
    def __init__(self, model: str | None = None, temperature: float = 0.35):
        super().__init__(
            model=model,
            temperature=temperature,
            prompt_template=ASTROLABE_TRANSFORM_PROMPT,
            cache_namespace="astrolabe_transform_subagent",
            provider_name="astrolabe_transform_subagent",
        )

    def _build_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(context, working_memory, validation_feedback)
        payload.update({
            "selectable_cards": context.extras.get("selectable_cards", []),
            "selected_cards": context.extras.get("selected_cards", []),
            "picks_remaining": context.extras.get("picks_remaining", 0),
            "deck_profile": context.extras.get("deck_profile", {}),
        })
        return payload


class GridSelectLlmProvider(BaseRewardLlmProvider):
    def __init__(self, model: str | None = None, temperature: float = 0.35):
        super().__init__(
            model=model,
            temperature=temperature,
            prompt_template=GRID_SELECT_PROMPT,
            cache_namespace="grid_select_subagent",
            provider_name="grid_select_subagent",
        )

    def _build_payload(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._base_payload(context, working_memory, validation_feedback)
        payload.update({
            "grid_mode": context.extras.get("grid_mode", "unknown"),
            "for_purge": context.extras.get("for_purge", False),
            "for_transform": context.extras.get("for_transform", False),
            "for_upgrade": context.extras.get("for_upgrade", False),
            "confirm_up": context.extras.get("confirm_up", False),
            "selectable_cards": context.extras.get("selectable_cards", []),
            "selected_cards": context.extras.get("selected_cards", []),
            "num_cards": context.extras.get("num_cards", 0),
            "picks_remaining": context.extras.get("picks_remaining", 0),
            "deck_profile": context.extras.get("deck_profile", {}),
            "deck_card_entries": context.extras.get("deck_card_entries", []),
            "relic_names": context.extras.get("relic_names", []),
        })
        return payload
