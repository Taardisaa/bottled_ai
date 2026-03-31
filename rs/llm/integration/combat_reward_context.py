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
            summary["card_choices"] = [str(choice).strip().lower() for choice in choice_list]

        summaries.append(summary)
    return summaries


def build_combat_reward_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()
    run_summary = get_cached_run_summary(state)
    reward_summaries = _build_reward_summaries(state)

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
            "reward_potion_names": state.get_reward_potion_names(),
            "screen_state": state.screen_state(),
            "game_state_ref": state,
        },
    )
