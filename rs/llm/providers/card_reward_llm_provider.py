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
            two_layer_struct_convert=False,
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
        deck_card_counts = context.extras.get("deck_card_name_counts", {})
        hand_card_names = context.extras.get("hand_card_names", [])
        card_details = self._build_card_details(context)

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
            deck_card_counts=deck_card_counts,
            hand_card_names=hand_card_names,
            choice_card_details=json.dumps(card_details["choice"], sort_keys=True),
            deck_card_details=json.dumps(card_details["deck"], sort_keys=True),
            hand_card_details=json.dumps(card_details["hand"], sort_keys=True),
            card_db_status="available",
        )

    def _build_card_details(self, context: AgentContext) -> dict[str, list[dict[str, Any]]]:
        choice_entries = self._build_entries_from_names(context.choice_list)
        deck_entries = self._coerce_card_entries(context.extras.get("deck_card_entries"), limit=16, sort_by_count=True)
        hand_entries = self._coerce_card_entries(context.extras.get("hand_card_entries"), limit=10, sort_by_count=False)

        return {
            "choice": self._query_card_detail_rows_from_entries(choice_entries),
            "deck": self._query_card_detail_rows_from_entries(deck_entries),
            "hand": self._query_card_detail_rows_from_entries(hand_entries),
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
                row["info"] = query_card(normalized, upgrade_times=upgrade_times)
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

    
