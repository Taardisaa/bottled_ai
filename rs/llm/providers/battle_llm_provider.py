from __future__ import annotations

import json
from typing import Any

from rs.llm.agents.base_agent import AgentContext


_PROMPT_TEMPLATE = """
You are the battle subagent for a Slay the Spire bot.

You are controlling exactly one battle session. Decide the next best step for the current battle state.

Rules:
- Use enumerate_legal_actions to discover legal commands.
- For play commands, use "play <hand_index> [target_index]".
- For hand-select or grid screens, use "choose <index>" and include "confirm" / "wait 30" when needed.
- Submit exactly ONE command at a time via submit_battle_commands. After each command, you will receive updated state. Only submit "end" when you have no more playable cards for your remaining energy.
- Prefer short, factual explanations.

Current battle state:
{battle_payload}
""".strip()


def _compact_game_state(game_state: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "floor": game_state.get("floor"),
        "act": game_state.get("act"),
        "hp": f"{game_state.get('current_hp')}/{game_state.get('max_hp')}",
        "gold": game_state.get("gold"),
        "class": game_state.get("character_class"),
        "turn": game_state.get("turn"),
    }
    ascension = game_state.get("ascension_level")
    if ascension:
        compact["ascension"] = ascension
    current_action = game_state.get("current_action")
    if current_action:
        compact["current_action"] = current_action
    return compact


def _compact_history(working_memory: dict[str, Any]) -> list[str]:
    batches = working_memory.get("executed_command_batches", [])[-4:]
    return [cmd for batch in batches for cmd in batch]


def build_battle_prompt(context: AgentContext, working_memory: dict[str, Any]) -> str:
    gs = _compact_game_state(context.game_state)

    hand = context.extras.get("hand_cards", [])
    monsters = context.extras.get("monster_summaries", [])
    potions = context.extras.get("potion_summaries", [])
    player_powers = context.extras.get("player_powers", [])

    payload: dict[str, Any] = {
        "game": gs,
        "energy": context.extras.get("player_energy"),
        "block": context.extras.get("player_block"),
        "hand": hand,
        "monsters": monsters,
    }

    if potions:
        payload["potions"] = potions
    if player_powers:
        payload["powers"] = player_powers

    selection_cards = context.extras.get("selection_cards", [])
    choice_list = context.choice_list
    if selection_cards:
        payload["selection_cards"] = selection_cards
    if choice_list:
        payload["choice_list"] = choice_list
    screen_type = context.screen_type
    if screen_type and screen_type != "NONE":
        payload["screen_type"] = screen_type

    context_parts: list[str] = []
    run_mem = str(context.extras.get("run_memory_summary", "")).strip()
    if run_mem:
        context_parts.append(run_mem)
    ep_mem = str(context.extras.get("retrieved_episodic_memories", "none")).strip()
    if ep_mem and ep_mem != "none":
        context_parts.append(ep_mem)
    sem_mem = str(context.extras.get("retrieved_semantic_memories", "none")).strip()
    if sem_mem and sem_mem != "none":
        context_parts.append(sem_mem)
    if context_parts:
        payload["context"] = " | ".join(context_parts)

    step_summaries = working_memory.get("recent_step_summaries", [])[-3:]
    history = _compact_history(working_memory)
    if step_summaries:
        payload["recent_steps"] = step_summaries
    if history:
        payload["history"] = history

    return _PROMPT_TEMPLATE.format(
        battle_payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
