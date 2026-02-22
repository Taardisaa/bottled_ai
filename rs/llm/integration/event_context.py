from __future__ import annotations

from rs.game.event import Event
from rs.llm.agents.base_agent import AgentContext
from rs.machine.state import GameState


def build_event_agent_context(state: GameState, handler_name: str) -> AgentContext:
    """Build compact event advisor context from current game state."""
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    event = state.get_event()
    event_name = event.value if isinstance(event, Event) else str(event)
    game_state = state.game_state()

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=state.get_choice_list().copy(),
        game_state={
            "event_name": event_name,
            "floor": state.floor(),
            "act": game_state.get("act"),
            "current_hp": game_state.get("current_hp"),
            "max_hp": game_state.get("max_hp"),
            "gold": game_state.get("gold"),
        },
        extras={
            "relic_names": [relic["name"] for relic in state.get_relics()],
            "deck_size": len(state.deck.cards),
        },
    )
