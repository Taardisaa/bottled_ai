from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel
from stsdb import query_card

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
            two_layer_struct_convert=config.llm_two_layer_struct_convert,
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
        character_class = context.game_state.get("character_class", "unknown")
        ascension_level = context.game_state.get("ascension_level", "unknown")
        act_boss = context.game_state.get("act_boss", "unknown")
        deck_size = context.extras.get("deck_size", "unknown")
        relic_names = context.extras.get("relic_names", [])
        deck_card_counts = context.extras.get("deck_card_name_counts", {})
        deck_profile = self._compact_deck_profile(context.extras.get("deck_profile", {}))
        run_memory_summary = context.extras.get("run_memory_summary", "")
        recent_llm_decisions = context.extras.get("recent_llm_decisions", "none")
        memory_context_block = self._build_memory_context_block(
            context.extras.get("retrieved_episodic_memories", "none"),
            context.extras.get("retrieved_semantic_memories", "none"),
        )
        choice_card_summaries = context.extras.get("choice_card_summaries", [])
        reward_screen_flags = context.extras.get("reward_screen_flags", {})
        card_details = self._build_card_details(context)
        reward_options_text = self._format_reward_options(choice_card_summaries, context.choice_list)

        return PROMPT_TEMPLATE.format(
            reward_options_text=reward_options_text,
            room_phase=room_phase,
            floor=floor,
            act=act,
            current_hp=hp,
            max_hp=max_hp,
            character_class=character_class,
            ascension_level=ascension_level,
            act_boss=act_boss,
            deck_size=deck_size,
            relic_names=json.dumps(relic_names, sort_keys=True),
            deck_card_counts=json.dumps(deck_card_counts, sort_keys=True),
            deck_profile=json.dumps(deck_profile, sort_keys=True),
            run_memory_summary=run_memory_summary,
            recent_llm_decisions=recent_llm_decisions,
            memory_context_block=memory_context_block,
            choice_card_summaries=json.dumps(choice_card_summaries, sort_keys=True),
            reward_screen_flags=json.dumps(reward_screen_flags, sort_keys=True),
            choice_card_details=json.dumps(card_details["choice"], sort_keys=True),
            deck_card_details=json.dumps(card_details["deck"], sort_keys=True),
        )

    def _format_reward_options(self, choice_card_summaries: list[dict[str, Any]], choice_list: list[Any]) -> str:
        if isinstance(choice_card_summaries, list) and choice_card_summaries:
            rendered_options: list[str] = []
            for fallback_index, card in enumerate(choice_card_summaries):
                if not isinstance(card, dict):
                    continue
                index = card.get("index", fallback_index)
                name = str(card.get("name", "")).strip()
                card_type = str(card.get("type", "")).strip()
                rarity = str(card.get("rarity", "")).strip()
                cost = card.get("cost", "unknown")
                upgrades = card.get("upgrades", 0)
                rendered_options.append(
                    f"- {index} | card=\"{name}\" | type={card_type or 'unknown'} | "
                    f"rarity={rarity or 'unknown'} | cost={cost} | upgrades={upgrades}"
                )
            if rendered_options:
                return "\n".join(rendered_options)

        if choice_list:
            return "\n".join(f"- {index} | option=\"{str(choice).strip()}\"" for index, choice in enumerate(choice_list))
        return "- none"

    def _build_card_details(self, context: AgentContext) -> dict[str, list[dict[str, Any]]]:
        choice_entries = self._build_entries_from_names(context.choice_list)
        deck_entries = self._coerce_card_entries(context.extras.get("deck_card_entries"), limit=16, sort_by_count=True)

        return {
            "choice": self._query_card_detail_rows_from_entries(choice_entries),
            "deck": self._query_card_detail_rows_from_entries(deck_entries),
        }

    def _build_entries_from_names(self, card_names: list[Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for raw_name in card_names:
            name, upgrade_times = self._extract_card_name_and_upgrades(raw_name)
            if name == "":
                continue
            entries.append({"name": name, "upgrade_times": upgrade_times})
        return entries

    def _coerce_card_entries(self, raw_entries: Any, limit: int, sort_by_count: bool) -> list[dict[str, Any]]:
        if not isinstance(raw_entries, list):
            return []

        entries: list[dict[str, Any]] = [entry for entry in raw_entries if isinstance(entry, dict)]
        if sort_by_count:
            entries = sorted(entries, key=lambda entry: -int(entry.get("count", 1) or 1))
        return entries[:limit]

    def _query_card_detail_rows_from_entries(self, entries: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            normalized, parsed_upgrade_times = self._extract_card_name_and_upgrades(entry.get("name", ""))
            entry_upgrade_times = entry.get("upgrade_times", parsed_upgrade_times)
            try:
                upgrade_times = int(entry_upgrade_times)
            except (TypeError, ValueError):
                upgrade_times = parsed_upgrade_times
            if upgrade_times < 0:
                upgrade_times = 0

            key = (normalized, upgrade_times)
            if normalized == "" or key in seen:
                continue
            seen.add(key)

            row: dict[str, Any] = {
                "name": normalized,
                "upgrade_times": upgrade_times,
            }
            if "count" in entry:
                row["count"] = entry.get("count")

            try:
                row["info"] = self._compact_card_info(query_card(normalized, upgrade_times=upgrade_times))
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
        return rows

    def _extract_card_name_and_upgrades(self, card_name: Any) -> tuple[str, int]:
        text = str(card_name).strip().lower()
        if text == "":
            return "", 0

        if "+" not in text:
            return text, 0

        if text.endswith("+"):
            return text[:-1].strip(), 1

        left, right = text.rsplit("+", 1)
        if right.isdigit():
            return left.strip(), int(right)

        return text, 0

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

    def _compact_deck_profile(self, deck_profile: Any) -> dict[str, Any]:
        if not isinstance(deck_profile, dict):
            return {}
        compacted: dict[str, Any] = {}
        for key in ("total_cards", "type_counts", "upgraded_cards", "exhaust_cards"):
            if key in deck_profile:
                compacted[key] = deck_profile[key]
        return compacted

    def _compact_card_info(self, raw_info: Any) -> Any:
        if not isinstance(raw_info, dict):
            return raw_info
        compacted: dict[str, Any] = {}
        for key in ("name", "type", "rarity", "cost", "description"):
            if key in raw_info:
                compacted[key] = raw_info[key]
        return compacted

    
