from __future__ import annotations

from typing import Any

from rs.game.screen_type import ScreenType
from rs.llm.agents.base_agent import AgentContext
from rs.llm.run_context import get_current_agent_identity
from rs.llm.state_summary_cache import get_cached_run_summary
from rs.machine.state import GameState


def is_battle_scope_state(state: GameState) -> bool:
    room_phase = str(state.game_state().get("room_phase", "")).strip().upper()
    screen_type = state.screen_type()
    return room_phase == "COMBAT" and screen_type in {
        ScreenType.NONE.value,
        ScreenType.HAND_SELECT.value,
        ScreenType.GRID.value,
    }


def _build_hand_cards(state: GameState) -> list[dict[str, Any]]:
    if state.combat_state() is None:
        return []

    summaries: list[dict[str, Any]] = []
    for index, card in enumerate(state.hand.cards, start=1):
        entry: dict[str, Any] = {
            "hand_index": index,
            "name": card.name,
            "cost": card.cost,
            "is_playable": card.is_playable,
        }
        if card.has_target:
            entry["has_target"] = True
        if card.upgrades:
            entry["upgrades"] = card.upgrades
        if card.ethereal:
            entry["ethereal"] = True
        if card.exhausts:
            entry["exhausts"] = True
        summaries.append(entry)
    return summaries


def _selection_card_source(state: GameState) -> list[dict[str, Any]]:
    screen_state = state.screen_state()
    if state.screen_type() == ScreenType.HAND_SELECT.value:
        raw_cards = screen_state.get("hand", [])
    elif state.screen_type() == ScreenType.GRID.value:
        raw_cards = screen_state.get("cards", [])
    else:
        raw_cards = []
    return [card for card in raw_cards if isinstance(card, dict)]


def _build_selection_cards(state: GameState) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, card in enumerate(_selection_card_source(state)):
        summaries.append({
            "choice_index": index,
            "name": str(card.get("name", "")).strip(),
            "id": str(card.get("id", "")).strip(),
            "cost": card.get("cost"),
            "type": card.get("type"),
            "upgrades": card.get("upgrades", 0),
            "has_target": bool(card.get("has_target", False)),
            "ethereal": bool(card.get("ethereal", False)),
            "exhausts": bool(card.get("exhausts", False)),
        })
    return summaries


def _build_choice_list(state: GameState, selection_cards: list[dict[str, Any]]) -> list[str]:
    if selection_cards:
        return [str(card.get("name", "")).strip().lower() for card in selection_cards]
    raw_choice_list = state.game_state().get("choice_list", [])
    if not isinstance(raw_choice_list, list):
        return []
    return [str(choice).strip().lower() for choice in raw_choice_list]


def _build_monster_summaries(state: GameState) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, monster in enumerate(state.get_monsters()):
        if bool(monster.get("is_gone", False)):
            continue
        entry: dict[str, Any] = {
            "target_index": index,
            "name": monster.get("name"),
            "current_hp": monster.get("current_hp"),
            "max_hp": monster.get("max_hp"),
            "intent": monster.get("intent"),
        }
        base_dmg = monster.get("move_base_damage", 0) or 0
        hits = monster.get("move_hits", 0) or 0
        if base_dmg > 0:
            entry["dmg"] = f"{base_dmg}x{hits}" if hits > 1 else str(base_dmg)
        block = monster.get("block", 0) or 0
        if block > 0:
            entry["block"] = block
        powers = monster.get("powers", [])
        if powers:
            entry["powers"] = [
                {"name": p.get("name"), "amount": p.get("amount")}
                for p in powers if isinstance(p, dict)
            ]
        summaries.append(entry)
    return summaries


def _build_potion_summaries(state: GameState) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, potion in enumerate(state.get_potions()):
        if str(potion.get("id", "")).strip() == "Potion Slot":
            continue
        summaries.append({
            "slot_index": index,
            "name": potion.get("name"),
        })
    return summaries


def build_battle_agent_context(
        state: GameState,
        handler_name: str,
        *,
        working_memory: dict[str, Any] | None = None,
        runtime: Any | None = None,
        retrieved_episodic_memories: str = "none",
        retrieved_semantic_memories: str = "none",
        langmem_status: str = "disabled_by_config",
) -> AgentContext:
    available_commands = state.json.get("available_commands")
    if not isinstance(available_commands, list):
        available_commands = []

    run_summary = get_cached_run_summary(state)
    game_state = state.game_state()
    combat_state = state.combat_state() or {}
    selection_cards = _build_selection_cards(state)
    hand_cards = _build_hand_cards(state)
    monster_summaries = _build_monster_summaries(state)
    potion_summaries = _build_potion_summaries(state)
    working_memory = dict(working_memory or {})

    return AgentContext(
        handler_name=handler_name,
        screen_type=state.screen_type(),
        available_commands=[str(command) for command in available_commands],
        choice_list=_build_choice_list(state, selection_cards),
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
            "turn": combat_state.get("turn"),
            "current_action": state.current_action(),
        },
        extras={
            "run_id": run_summary["run_id"],
            "agent_identity": get_current_agent_identity(),
            "deck_size": run_summary["deck_size"],
            "deck_profile": run_summary["deck_profile"],
            "deck_card_name_counts": run_summary["deck_card_name_counts"],
            "relic_names": run_summary["relic_names"],
            "held_potion_names": run_summary["held_potion_names"],
            "potions_full": run_summary["potions_full"],
            "run_memory_summary": run_summary["run_memory_summary"],
            "retrieved_episodic_memories": retrieved_episodic_memories,
            "retrieved_semantic_memories": retrieved_semantic_memories,
            "langmem_status": langmem_status,
            "battle_working_memory": working_memory,
            "hand_size": len(hand_cards),
            "hand_cards": hand_cards,
            "selection_cards": selection_cards,
            "selection_card_count": len(selection_cards),
            "selection_max_cards": state.screen_state().get("max_cards", 0),
            "selection_can_pick_zero": bool(state.screen_state().get("can_pick_zero", False)),
            "alive_monster_count": len(monster_summaries),
            "monster_summaries": monster_summaries,
            "potion_slots": len(potion_summaries),
            "potion_summaries": potion_summaries,
            "player_energy": combat_state.get("player", {}).get("energy"),
            "player_block": combat_state.get("player", {}).get("block"),
            "player_powers": [
                {"name": p.get("name"), "amount": p.get("amount")}
                for p in (combat_state.get("player", {}).get("powers", []) or [])
                if isinstance(p, dict)
            ],
            "screen_state": state.screen_state(),
            "game_state_ref": state,
            "battle_runtime": runtime,
        },
    )
