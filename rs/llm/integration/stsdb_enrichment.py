"""Shared stsdb description enrichment utilities for all context builders."""
from __future__ import annotations

from typing import Any


def query_card_description(name: str, upgrades: int, character_class: str = "") -> str:
    try:
        from stsdb import query_card
        result = query_card(name, upgrade_times=upgrades, character_class=character_class)
        if isinstance(result, dict) and result.get("found"):
            desc = result.get("entry", {}).get("description", "")
            if desc:
                return str(desc)
    except Exception:
        pass
    return ""


def query_relic_description(name: str) -> str:
    try:
        from stsdb import query_relic
        result = query_relic(name)
        if isinstance(result, dict) and result.get("found"):
            desc = result.get("entry", {}).get("description", "")
            if desc:
                return str(desc)
    except Exception:
        pass
    return ""


def query_potion_description(name: str) -> str:
    try:
        from stsdb import query_potion
        result = query_potion(name)
        if isinstance(result, dict) and result.get("found"):
            desc = result.get("entry", {}).get("description", "")
            if desc:
                return str(desc)
    except Exception:
        pass
    return ""


def query_power_description(name: str) -> str:
    try:
        from stsdb import query_power
        result = query_power(name)
        if isinstance(result, dict) and result.get("found"):
            desc = result.get("entry", {}).get("description", "")
            if desc:
                return str(desc)
    except Exception:
        pass
    return ""


def enrich_relic_names(relic_names: list[str]) -> list[dict[str, Any]]:
    """Convert a list of relic names into summaries with descriptions."""
    summaries: list[dict[str, Any]] = []
    for name in relic_names:
        entry: dict[str, Any] = {"name": name}
        desc = query_relic_description(name)
        if desc:
            entry["description"] = desc
        summaries.append(entry)
    return summaries


def enrich_potion_names(potion_names: list[str]) -> list[dict[str, Any]]:
    """Convert a list of potion names into summaries with descriptions."""
    summaries: list[dict[str, Any]] = []
    for name in potion_names:
        if not name:
            continue
        entry: dict[str, Any] = {"name": name}
        desc = query_potion_description(name)
        if desc:
            entry["description"] = desc
        summaries.append(entry)
    return summaries


def enrich_powers(powers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add descriptions to a list of power dicts."""
    enriched: list[dict[str, Any]] = []
    for p in powers:
        if not isinstance(p, dict):
            continue
        entry: dict[str, Any] = {"name": p.get("name"), "amount": p.get("amount")}
        desc = query_power_description(p.get("name", ""))
        if desc:
            entry["description"] = desc
        enriched.append(entry)
    return enriched


def enrich_card_entries(card_entries: list[dict[str, Any]], character_class: str = "") -> list[dict[str, Any]]:
    """Add descriptions to a list of card entry dicts (must have 'name' and optionally 'upgrade_times' or 'upgrades')."""
    enriched: list[dict[str, Any]] = []
    for card in card_entries:
        if not isinstance(card, dict):
            continue
        entry = dict(card)
        name = card.get("name", "")
        upgrades = card.get("upgrade_times", card.get("upgrades", 0)) or 0
        desc = query_card_description(name, upgrades, character_class)
        if desc:
            entry["description"] = desc
        enriched.append(entry)
    return enriched
