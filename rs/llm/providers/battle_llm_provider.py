from __future__ import annotations

import json
from typing import Any

from rs.llm.agents.base_agent import AgentContext


_PROMPT_TEMPLATE = """
You are the battle subagent for a Slay the Spire bot.

You are controlling exactly one battle session. Decide the next best step for the current battle state.

Rules:
- Use tools to analyze the state before committing to a command.
- Do not invent commands that are not in available_commands.
- For play commands, use the protocol form "play <hand_index> [target_index]".
- For hand-select or grid screens, use "choose <index>" and include "confirm" / "wait 30" when needed.
- When you have decided on the best move, call submit_battle_commands with your chosen commands.
- Prefer short, factual explanations.

Current battle state:
{battle_payload}
""".strip()


def build_battle_prompt(context: AgentContext, working_memory: dict[str, Any]) -> str:
    payload = {
        "handler_name": context.handler_name,
        "screen_type": context.screen_type,
        "available_commands": context.available_commands,
        "choice_list": context.choice_list,
        "game_state": context.game_state,
        "battle_state": {
            "hand_cards": context.extras.get("hand_cards", []),
            "selection_cards": context.extras.get("selection_cards", []),
            "monster_summaries": context.extras.get("monster_summaries", []),
            "potion_summaries": context.extras.get("potion_summaries", []),
            "player_energy": context.extras.get("player_energy"),
            "player_block": context.extras.get("player_block"),
            "player_powers": context.extras.get("player_powers", []),
        },
        "run_memory_summary": context.extras.get("run_memory_summary", ""),
        "retrieved_episodic_memories": context.extras.get("retrieved_episodic_memories", "none"),
        "retrieved_semantic_memories": context.extras.get("retrieved_semantic_memories", "none"),
        "battle_working_memory": {
            "recent_step_summaries": working_memory.get("recent_step_summaries", [])[-6:],
            "executed_command_batches": working_memory.get("executed_command_batches", [])[-4:],
        },
    }
    return _PROMPT_TEMPLATE.format(
        battle_payload=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )
