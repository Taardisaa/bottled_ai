from __future__ import annotations

from rs.llm.agents.base_agent import AgentContext
from rs.machine.state import GameState


def build_shop_purchase_agent_context(state: GameState, handler_name: str) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    game_state = state.game_state()

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "floor": state.floor(),
            "act": game_state.get("act"),
            "gold": game_state.get("gold"),
        },
        extras={
            "has_removable_curse": state.deck.contains_curses_we_can_remove(),
            "deck_size": len(state.deck.cards),
        },
    )
