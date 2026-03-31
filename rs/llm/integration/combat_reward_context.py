from __future__ import annotations

from typing import Any

from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def _build_reward_summaries(state: GameState) -> list[dict[str, Any]]:
    screen_state = state.screen_state()
    rewards = screen_state.get("rewards", [])
    if not isinstance(rewards, list):
        rewards = []

    summaries: list[dict[str, Any]] = []
    choice_list = state.get_choice_list()
    for index, reward in enumerate(rewards):
        if not isinstance(reward, dict):
            continue

        choice_token = choice_list[index] if index < len(choice_list) else ""
        reward_type = str(reward.get("reward_type", "")).strip().upper()
        summary: dict[str, Any] = {
            "choice_index": index,
            "choice_token": str(choice_token).strip().lower(),
            "reward_type": reward_type,
        }

        if reward_type in {"GOLD", "STOLEN_GOLD"}:
            summary["gold"] = reward.get("gold")
        elif reward_type == "POTION":
            potion = reward.get("potion", {})
            if isinstance(potion, dict):
                summary["potion_name"] = str(potion.get("name", "")).strip().lower()
                summary["potion_id"] = str(potion.get("id", "")).strip()
                summary["potion_can_use"] = bool(potion.get("can_use", False))
                summary["potion_can_discard"] = bool(potion.get("can_discard", False))
                summary["potion_requires_target"] = bool(potion.get("requires_target", False))
        elif reward_type == "RELIC":
            relic = reward.get("relic", {})
            if isinstance(relic, dict):
                summary["relic_name"] = str(relic.get("name", "")).strip().lower()
                summary["relic_id"] = str(relic.get("id", "")).strip()
        elif reward_type == "SAPPHIRE_KEY":
            link = reward.get("link", reward.get("linked_relic", {}))
            if isinstance(link, dict):
                summary["linked_relic_name"] = str(link.get("name", "")).strip().lower()
                summary["linked_relic_id"] = str(link.get("id", "")).strip()
        elif reward_type == "CARD":
            summary["card_reward_entry_only"] = True
            summary["delegates_to_handler"] = "CardRewardHandler"

        summary["reward_summary_line"] = _build_reward_summary_line(summary)

        summaries.append(summary)
    return summaries


def _mask_llm_reward_summaries(reward_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(summary) for summary in reward_summaries if str(summary.get("reward_type", "")).upper() != "CARD"]


def _extract_card_reward_metadata(reward_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    card_rows = [summary for summary in reward_summaries if str(summary.get("reward_type", "")).upper() == "CARD"]
    card_choice_indexes = [int(row.get("choice_index")) for row in card_rows if isinstance(row.get("choice_index"), int)]
    card_choice_tokens = [str(row.get("choice_token", "")).strip().lower() for row in card_rows]
    return {
        "has_card_reward_row": len(card_rows) > 0,
        "card_reward_choice_index": card_choice_indexes[0] if card_choice_indexes else None,
        "card_reward_choice_indexes": card_choice_indexes,
        "card_reward_choice_tokens": [token for token in card_choice_tokens if token != ""],
        "card_reward_count": len(card_rows),
    }


def _build_llm_choice_list(choice_list: list[str], card_choice_indexes: list[int]) -> list[str]:
    card_index_set = set(card_choice_indexes)
    return [
        choice
        for index, choice in enumerate(choice_list)
        if index not in card_index_set
    ]


def _build_reward_summary_line(summary: dict[str, Any]) -> str:
    index = summary.get("choice_index", "?")
    token = summary.get("choice_token", "")
    reward_type = str(summary.get("reward_type", "UNKNOWN")).strip().upper() or "UNKNOWN"

    base = f"idx={index} token='{token}' type={reward_type}"
    if reward_type in {"GOLD", "STOLEN_GOLD"}:
        return f"{base} gold={summary.get('gold')}"
    if reward_type == "RELIC":
        return (
            f"{base} relic_name='{summary.get('relic_name', '')}' "
            f"relic_id='{summary.get('relic_id', '')}'"
        )
    if reward_type == "POTION":
        return (
            f"{base} potion_name='{summary.get('potion_name', '')}' "
            f"can_use={bool(summary.get('potion_can_use', False))} "
            f"can_discard={bool(summary.get('potion_can_discard', False))} "
            f"requires_target={bool(summary.get('potion_requires_target', False))}"
        )
    if reward_type == "SAPPHIRE_KEY":
        return (
            f"{base} linked_relic_name='{summary.get('linked_relic_name', '')}' "
            f"linked_relic_id='{summary.get('linked_relic_id', '')}'"
        )
    if reward_type == "CARD":
        delegate_to = str(summary.get("delegates_to_handler", "CardRewardHandler")).strip() or "CardRewardHandler"
        return f"{base} card_reward_entry_only=true delegates_to_handler={delegate_to}"
    return base


def build_combat_reward_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    run_summary = get_cached_run_summary(state)
    all_reward_summaries = _build_reward_summaries(state)
    reward_summaries = _mask_llm_reward_summaries(all_reward_summaries)
    card_metadata = _extract_card_reward_metadata(all_reward_summaries)
    llm_choice_list = _build_llm_choice_list(
        state.get_choice_list().copy(),
        card_metadata["card_reward_choice_indexes"],
    )

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "room_phase": game_state.get("room_phase"),
            "room_type": game_state.get("room_type"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
            "character_class": game_state.get("class"),
            "ascension_level": game_state.get("ascension_level"),
            "act_boss": game_state.get("act_boss"),
        },
        extras={
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "deck_size": run_summary["deck_size"],
            "deck_profile": run_summary["deck_profile"],
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "reward_summaries": reward_summaries,
            "all_reward_summaries": all_reward_summaries,
            "llm_choice_list": llm_choice_list,
            "non_card_reward_count": len(reward_summaries),
            "has_card_reward_row": card_metadata["has_card_reward_row"],
            "card_reward_choice_index": card_metadata["card_reward_choice_index"],
            "card_reward_choice_indexes": card_metadata["card_reward_choice_indexes"],
            "card_reward_choice_tokens": card_metadata["card_reward_choice_tokens"],
            "card_reward_count": card_metadata["card_reward_count"],
            "reward_potion_names": state.get_reward_potion_names(),
            "screen_state": state.screen_state(),
            "game_state_ref": state,
        },
    )
