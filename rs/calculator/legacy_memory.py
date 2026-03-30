from __future__ import annotations

from rs.calculator.enums.card_id import CardId
from rs.calculator.interfaces.memory_items import MemoryItem, ResetSchedule, StanceType


def build_default_memory_general(*, last_known_turn: int = 0) -> dict:
    return {
        MemoryItem.KILLED_WITH_LESSON_LEARNED: 0,
        MemoryItem.CLAWS_THIS_BATTLE: 0,
        MemoryItem.FROST_THIS_BATTLE: 0,
        MemoryItem.LIGHTNING_THIS_BATTLE: 0,
        MemoryItem.MANTRA_THIS_BATTLE: 0,
        MemoryItem.PANACHE_DAMAGE: 0,
        MemoryItem.SAVE_INTERNAL_MANTRA: 0,
        MemoryItem.STANCE: StanceType.NO_STANCE,
        MemoryItem.TYPE_LAST_PLAYED: 0,
        MemoryItem.ATTACKS_THIS_TURN: 0,
        MemoryItem.CARDS_THIS_TURN: 0,
        MemoryItem.LAST_KNOWN_TURN: last_known_turn,
        MemoryItem.NECRONOMICON_READY: 1,
        MemoryItem.ORANGE_PELLETS_ATTACK: 0,
        MemoryItem.ORANGE_PELLETS_SKILL: 0,
        MemoryItem.ORANGE_PELLETS_POWER: 0,
        MemoryItem.PANACHE_COUNTER: 5,
        MemoryItem.RECYCLE: 0,
    }


def build_default_memory_by_card() -> dict[CardId, dict[ResetSchedule, dict[str, int]]]:
    schedule_by_card = {
        CardId.GENETIC_ALGORITHM: ResetSchedule.GAME,
        CardId.GLASS_KNIFE: ResetSchedule.BATTLE,
        CardId.PERSEVERANCE: ResetSchedule.BATTLE,
        CardId.RAMPAGE: ResetSchedule.BATTLE,
        CardId.RITUAL_DAGGER: ResetSchedule.GAME,
        CardId.STEAM_BARRIER: ResetSchedule.BATTLE,
        CardId.WINDMILL_STRIKE: ResetSchedule.BATTLE,
    }
    return {
        card_id: {reset_schedule: {"": 0}}
        for card_id, reset_schedule in schedule_by_card.items()
    }


def build_default_memory_snapshots(*, last_known_turn: int = 0) -> tuple[dict, dict[CardId, dict[ResetSchedule, dict[str, int]]]]:
    return (
        build_default_memory_general(last_known_turn=last_known_turn),
        build_default_memory_by_card(),
    )
