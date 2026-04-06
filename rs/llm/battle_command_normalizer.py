"""Normalize LLM-generated battle commands into strict Communication Mod syntax.

Runs before validation to auto-correct common LLM mistakes like natural language
targets, card names instead of indices, and targeted/untargeted mismatches.
"""
from __future__ import annotations

import re
from typing import Any

from rs.llm.agents.base_agent import AgentContext


_SIMPLE_COMMANDS = {"end", "confirm", "proceed", "skip", "return", "leave", "cancel"}
_QUOTE_RE = re.compile(r'["\']')


def normalize_battle_command(raw: str, context: AgentContext) -> str:
    """Attempt to normalize a raw LLM command into valid syntax.

    Returns the normalized command, or the original if no normalization applies.
    """
    command = raw.strip()
    if not command:
        return command

    # Strip quotes the LLM likes to add
    command = _QUOTE_RE.sub("", command)

    # Normalize underscores and common noise
    parts = command.split()
    if not parts:
        return command

    verb = parts[0].lower().replace("_", " ").strip()

    # Rule 1: Strip trailing noise from simple commands
    if verb in _SIMPLE_COMMANDS:
        return verb

    # Rule 1b: wait — keep only the integer duration
    if verb == "wait":
        for p in parts[1:]:
            if p.isdigit():
                return f"wait {p}"
        return command

    # Rule 2-4: play commands
    if verb == "play":
        return _normalize_play(parts, context)

    # Rule 5: potion commands
    if verb == "potion":
        return _normalize_potion(parts, context)

    # Rule 1c: handle "end_turn", "end turn" etc. as a single token
    joined = command.lower().replace("_", " ").replace("-", " ")
    for simple in _SIMPLE_COMMANDS:
        if joined.startswith(simple):
            return simple

    return command


def _normalize_play(parts: list[str], context: AgentContext) -> str:
    """Normalize a play command."""
    if len(parts) < 2:
        return " ".join(parts)

    hand_cards = context.extras.get("hand_cards", [])
    monsters = context.extras.get("monster_summaries", [])
    alive_monsters = [m for m in monsters if not m.get("is_gone", False)]

    card_token = parts[1]
    target_tokens = parts[2:]

    # Try to resolve card_token as integer index first
    card_index = _try_int(card_token)
    if card_index is None:
        # Rule 4: card name → hand index
        card_index = _resolve_card_name(" ".join(parts[1:]), hand_cards, target_tokens)
        if card_index is None:
            return " ".join(parts)
        # Re-parse: if we consumed tokens for the card name, figure out remaining target
        card_name_consumed, target_tokens = _split_card_name_and_target(parts[1:], hand_cards)
        if card_name_consumed is None:
            return " ".join(parts)

    card = _find_card_by_index(card_index, hand_cards)

    # Rule 3: resolve monster name in target
    target_index = _resolve_target(target_tokens, alive_monsters)

    # Rule 2: auto-fix targeted/untargeted mismatch
    if card is not None:
        if not card.get("has_target", False):
            # Untargeted card — strip any target
            return f"play {card_index}"
        else:
            # Targeted card
            if target_index is not None:
                return f"play {card_index} {target_index}"
            elif len(alive_monsters) == 1:
                # Only one monster — auto-target
                return f"play {card_index} {alive_monsters[0].get('target_index', 0)}"

    # Fallback: reconstruct with whatever we resolved
    if target_index is not None:
        return f"play {card_index} {target_index}"
    elif target_tokens:
        # Couldn't resolve target, pass through original
        return " ".join(parts)
    else:
        return f"play {card_index}"


def _normalize_potion(parts: list[str], context: AgentContext) -> str:
    """Normalize a potion command."""
    if len(parts) < 3:
        return " ".join(parts)

    action = parts[1].lower()
    if action not in ("use", "discard"):
        return " ".join(parts)

    potions = context.extras.get("potion_summaries", [])
    remaining = parts[2:]

    # Try integer index first
    potion_index = _try_int(remaining[0])
    target_tokens = remaining[1:] if potion_index is not None else []

    if potion_index is None:
        # Rule 5: potion name → slot index
        potion_index, target_tokens = _resolve_potion_name(remaining, potions)
        if potion_index is None:
            return " ".join(parts)

    # Resolve target if present
    monsters = context.extras.get("monster_summaries", [])
    alive_monsters = [m for m in monsters if not m.get("is_gone", False)]
    target_index = _resolve_target(target_tokens, alive_monsters)

    if target_index is not None:
        return f"potion {action} {potion_index} {target_index}"
    elif target_tokens:
        return " ".join(parts)
    else:
        return f"potion {action} {potion_index}"


def _try_int(s: str) -> int | None:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _find_card_by_index(index: int, hand_cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    for card in hand_cards:
        if card.get("hand_index") == index:
            return card
    return None


def _resolve_card_name(text: str, hand_cards: list[dict[str, Any]], target_tokens: list[str]) -> int | None:
    """Try to match a card name from the text against hand cards. Returns hand_index or None."""
    if not hand_cards:
        return None

    # Try exact match first (case-insensitive)
    text_lower = text.lower().strip()
    for card in hand_cards:
        name = str(card.get("name", "")).lower()
        if text_lower == name or text_lower.startswith(name + " ") or text_lower.startswith(name):
            # Check uniqueness
            matches = [c for c in hand_cards if str(c.get("name", "")).lower() == name]
            if len(matches) == 1:
                return matches[0].get("hand_index")
            return None  # Ambiguous
    return None


def _split_card_name_and_target(
    tokens: list[str], hand_cards: list[dict[str, Any]]
) -> tuple[int | None, list[str]]:
    """Try to split tokens into card name + target tokens."""
    for i in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:i]).lower()
        for card in hand_cards:
            name = str(card.get("name", "")).lower()
            if candidate == name:
                matches = [c for c in hand_cards if str(c.get("name", "")).lower() == name]
                if len(matches) == 1:
                    return matches[0].get("hand_index"), tokens[i:]
    return None, tokens


def _resolve_target(tokens: list[str], alive_monsters: list[dict[str, Any]]) -> int | None:
    """Resolve target tokens to a monster index."""
    if not tokens:
        return None

    # Try integer first
    if len(tokens) == 1:
        idx = _try_int(tokens[0])
        if idx is not None:
            return idx

    # Try monster name match
    target_text = " ".join(tokens).lower().strip()
    if not target_text:
        return None

    matches = []
    for m in alive_monsters:
        name = str(m.get("name", "")).lower()
        if target_text == name or name.startswith(target_text) or target_text in name:
            matches.append(m)

    if len(matches) == 1:
        return matches[0].get("target_index")

    return None  # Ambiguous or no match


def _resolve_potion_name(
    tokens: list[str], potions: list[dict[str, Any]]
) -> tuple[int | None, list[str]]:
    """Try to match potion name from tokens. Returns (slot_index, remaining_tokens)."""
    for i in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:i]).lower()
        for potion in potions:
            name = str(potion.get("name", "")).lower()
            if candidate == name:
                return potion.get("slot_index"), tokens[i:]
    return None, tokens
