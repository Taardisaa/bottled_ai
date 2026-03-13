from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

from rs.machine.state import GameState
from rs.utils.hash_utils import sha256_from_json_payload


_MAX_CACHED_SUMMARIES = 128
_RUN_SUMMARY_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _build_card_entries_with_counts(state: GameState) -> list[dict[str, int | str]]:
    counts: dict[tuple[str, int], int] = {}
    for card in state.deck.cards:
        key = (card.name.strip().lower(), int(card.upgrades))
        counts[key] = counts.get(key, 0) + 1

    entries: list[dict[str, int | str]] = []
    for (card_name, upgrade_times), count in counts.items():
        entries.append({
            "name": card_name,
            "upgrade_times": upgrade_times,
            "count": count,
        })
    return entries


def _build_deck_profile(state: GameState) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    rarity_counts: dict[str, int] = {}
    cost_buckets = {
        "zero_cost": 0,
        "one_cost": 0,
        "two_cost": 0,
        "three_plus_cost": 0,
        "x_cost": 0,
        "unplayable": 0,
    }
    upgraded_cards = 0
    exhaust_cards = 0
    targeted_cards = 0

    for card in state.deck.cards:
        type_key = card.type.value
        type_counts[type_key] = type_counts.get(type_key, 0) + 1

        rarity_key = card.rarity.value
        rarity_counts[rarity_key] = rarity_counts.get(rarity_key, 0) + 1

        if card.cost == -1:
            cost_buckets["x_cost"] += 1
        elif card.cost < 0:
            cost_buckets["unplayable"] += 1
        elif card.cost == 0:
            cost_buckets["zero_cost"] += 1
        elif card.cost == 1:
            cost_buckets["one_cost"] += 1
        elif card.cost == 2:
            cost_buckets["two_cost"] += 1
        else:
            cost_buckets["three_plus_cost"] += 1

        if card.upgrades > 0:
            upgraded_cards += 1
        if card.exhausts:
            exhaust_cards += 1
        if card.has_target:
            targeted_cards += 1

    return {
        "total_cards": len(state.deck.cards),
        "type_counts": type_counts,
        "rarity_counts": rarity_counts,
        "cost_buckets": cost_buckets,
        "upgraded_cards": upgraded_cards,
        "exhaust_cards": exhaust_cards,
        "targeted_cards": targeted_cards,
    }


def _build_run_summary(state: GameState) -> dict[str, Any]:
    game_state = state.game_state()
    deck_profile = _build_deck_profile(state)
    relic_names = [relic["name"] for relic in state.get_relics()]
    held_potion_names = state.get_held_potion_names()

    return {
        "run_id": _build_run_id(game_state),
        "deck_size": len(state.deck.cards),
        "deck_profile": deck_profile,
        "deck_card_name_counts": state.get_deck_card_list_by_name_with_upgrade_stripped(),
        "deck_card_entries": _build_card_entries_with_counts(state),
        "relic_names": relic_names,
        "held_potion_names": held_potion_names,
        "potions_full": state.are_potions_full(),
        "run_memory_summary": _build_run_memory_summary(
            game_state=game_state,
            deck_profile=deck_profile,
            relic_names=relic_names,
            held_potion_names=held_potion_names,
        ),
    }


def _build_run_id(game_state: dict[str, Any]) -> str:
    character_class = str(game_state.get("class", "unknown")).strip().lower()
    seed = str(game_state.get("seed", "unknown")).strip().lower()
    return f"{character_class}:{seed}"


def _build_run_memory_summary(
        game_state: dict[str, Any],
        deck_profile: dict[str, Any],
        relic_names: list[str],
        held_potion_names: list[str],
) -> str:
    character_class = str(game_state.get("class", "unknown"))
    floor = game_state.get("floor", "unknown")
    act = game_state.get("act", "unknown")
    current_hp = game_state.get("current_hp", "unknown")
    max_hp = game_state.get("max_hp", "unknown")
    gold = game_state.get("gold", "unknown")
    total_cards = deck_profile.get("total_cards", 0)
    type_counts = deck_profile.get("type_counts", {})
    upgraded_cards = deck_profile.get("upgraded_cards", 0)

    dominant_type = "UNKNOWN"
    if isinstance(type_counts, dict) and type_counts:
        dominant_type = str(max(type_counts.items(), key=lambda item: item[1])[0])

    relic_preview = ", ".join(relic_names[:3]) if relic_names else "none"
    filled_potions = len([name for name in held_potion_names if str(name).strip().lower() != "potion slot"])

    return (
        f"{character_class} on Act {act} Floor {floor} at HP {current_hp}/{max_hp} with {gold} gold, "
        f"{total_cards} cards ({dominant_type}-leaning, {upgraded_cards} upgraded), "
        f"relics: {relic_preview}, held potions: {filled_potions}."
    )


def _build_summary_cache_key(state: GameState) -> str:
    return sha256_from_json_payload(state.json)


def get_cached_run_summary(state: GameState) -> dict[str, Any]:
    cache_key = _build_summary_cache_key(state)
    cached = _RUN_SUMMARY_CACHE.get(cache_key)
    if cached is None:
        cached = _build_run_summary(state)
        _RUN_SUMMARY_CACHE[cache_key] = cached
        if len(_RUN_SUMMARY_CACHE) > _MAX_CACHED_SUMMARIES:
            _RUN_SUMMARY_CACHE.popitem(last=False)
    else:
        _RUN_SUMMARY_CACHE.move_to_end(cache_key)

    return deepcopy(cached)


def clear_cached_run_summaries() -> None:
    _RUN_SUMMARY_CACHE.clear()
