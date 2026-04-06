from __future__ import annotations

import json
from typing import Any

from rs.llm.agents.base_agent import AgentContext


_SYSTEM_PROMPT = """You are the battle subagent for a Slay the Spire bot.

You are controlling exactly one battle session. Decide the next best step for the current battle state.

Situational guidelines:
- If you can defeat an enemy within a few moves, prioritise finishing it off.
- If the enemy's attack would kill you or leave you critically low, prioritise blocking.
- Only use potions if extremely necessary: you are at low HP and the enemy can kill you in a few turns and you need the potion to gain an advantage; or the enemy is an elite or boss, but even then only use a potion if it can expose its full power.
- If you cannot defeat the enemy soon and it is attacking, balance damage and block based on the threat.
- In multi-enemy fights, focus fire on one enemy to reduce incoming damage sources.
- If you are unsure what to do next, consider the calculator's suggestion.

Rules:
- Legal actions and calculator recommendation are provided in each state update. Use submit_battle_commands to execute your chosen command.
- For play commands, use "play <hand_index> [target_index]".
- For hand-select or grid screens, use "choose <index>" and include "confirm" / "wait 30" when needed.
- Submit exactly ONE command at a time via submit_battle_commands. After each command, you will receive updated state. Only submit "end" when you have no more playable cards for your remaining energy.
- When calling submit_battle_commands, always include a "reasoning" argument: briefly state what you observe (enemy HP, intent, your hand), what you chose, and why.
""".strip()

# Kept for backwards compatibility with tests
_PROMPT_TEMPLATE = _SYSTEM_PROMPT + """

Current battle state:
{battle_payload}
"""


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
    room_type = game_state.get("room_type")
    if room_type:
        compact["room_type"] = room_type
    current_action = game_state.get("current_action")
    if current_action:
        compact["current_action"] = current_action
    return compact


def _compact_history(working_memory: dict[str, Any]) -> list[str]:
    batches = working_memory.get("executed_command_batches", [])[-4:]
    return [cmd for batch in batches for cmd in batch]


def _build_state_payload(context: AgentContext, working_memory: dict[str, Any]) -> dict[str, Any]:
    """Build the dynamic state payload dict (shared by all prompt builders)."""
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

    legal = working_memory.get("legal_actions", {})
    if legal.get("categories"):
        payload["legal_actions"] = legal["categories"]
    calc_rec = working_memory.get("calculator_recommendation", [])
    if calc_rec:
        payload["calculator_recommendation"] = calc_rec

    return payload


def build_battle_system_prompt(context: AgentContext, working_memory: dict[str, Any]) -> str:
    """System prompt set once at battle start. Includes rules and static context (memories)."""
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

    parts = [_SYSTEM_PROMPT]
    if context_parts:
        parts.append("\nBattle context:\n" + " | ".join(context_parts))

    relic_summaries = context.extras.get("relic_summaries", [])
    if relic_summaries:
        relic_lines = [
            f"- {r['name']}: {r['description']}" if r.get("description") else f"- {r['name']}"
            for r in relic_summaries
        ]
        parts.append("\nRelics:\n" + "\n".join(relic_lines))

    return "\n".join(parts)


def build_battle_state_update(context: AgentContext, working_memory: dict[str, Any]) -> str:
    """Compact state update appended as HumanMessage each card play."""
    payload = _build_state_payload(context, working_memory)
    return "Updated battle state:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_battle_prompt(context: AgentContext, working_memory: dict[str, Any]) -> str:
    """Legacy single-message prompt (used by tests). Combines system + state."""
    payload = _build_state_payload(context, working_memory)

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
