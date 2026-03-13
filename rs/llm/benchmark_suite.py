from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from definitions import ROOT_DIR
from rs.helper.seed import get_seed_string
from rs.machine.character import Character


_DEFAULT_STRATEGY_BY_CHARACTER = {
    Character.IRONCLAD: "requested_strike",
    Character.SILENT: "shivs_and_giggles",
    Character.DEFECT: "pwnder_my_orbs",
    Character.WATCHER: "peaceful_pummeling",
}

_CHARACTER_NAME_MAP = {
    "IRONCLAD": Character.IRONCLAD,
    "IRONCLAD".title(): Character.IRONCLAD,
    "THE_SILENT": Character.SILENT,
    "SILENT": Character.SILENT,
    "Silent": Character.SILENT,
    "DEFECT": Character.DEFECT,
    "Defect": Character.DEFECT,
    "WATCHER": Character.WATCHER,
    "Watcher": Character.WATCHER,
}


@dataclass(frozen=True)
class LlmBenchmarkCase:
    case_id: str
    fixture_path: str
    handler_area: str
    phase: str
    seed: str
    character: Character
    recommended_strategy: str
    act: int
    floor: int
    room_type: str
    tags: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    fixture_path: str
    handler_area: str
    phase: str
    tags: tuple[str, ...]
    notes: str = ""


_SUITE_SPECS: tuple[_CaseSpec, ...] = (
    _CaseSpec(
        case_id="silent_cleric_heal",
        fixture_path="tests/res/event/event_cleric_heal.json",
        handler_area="event",
        phase="phase_1",
        tags=("act1", "silent", "safe_event"),
        notes="Low-risk pilot event baseline for Silent event advisor comparisons.",
    ),
    _CaseSpec(
        case_id="ironclad_purifier_event",
        fixture_path="tests/res/event/event_purifier.json",
        handler_area="event",
        phase="phase_1",
        tags=("act1", "ironclad", "edge_event"),
        notes="Ironclad event case with a concrete purge-style decision.",
    ),
    _CaseSpec(
        case_id="watcher_falling_event",
        fixture_path="tests/res/event/event_falling.json",
        handler_area="event",
        phase="phase_1",
        tags=("act1", "watcher", "edge_event"),
        notes="Edge-case event with constrained card-loss options.",
    ),
    _CaseSpec(
        case_id="watcher_vampires_many_strikes",
        fixture_path="tests/res/event/vampires_with_many_strikes.json",
        handler_area="event",
        phase="phase_1",
        tags=("act2", "watcher", "edge_event"),
        notes="Event seed that stresses strike-count tradeoffs.",
    ),
    _CaseSpec(
        case_id="ironclad_shop_buy_attack",
        fixture_path="tests/res/shop/shop_buy_perfected_strike.json",
        handler_area="shop",
        phase="phase_2",
        tags=("act1", "ironclad", "shop_buy"),
        notes="Basic shop purchase decision with affordable card offers.",
    ),
    _CaseSpec(
        case_id="watcher_shop_skip_purge",
        fixture_path="tests/res/shop/shop_do_not_purge_because_no_removable_curses.json",
        handler_area="shop",
        phase="phase_2",
        tags=("act2", "watcher", "shop_purge"),
        notes="Shop regression where purge should be rejected.",
    ),
    _CaseSpec(
        case_id="watcher_shop_selective_purge",
        fixture_path="tests/res/shop/shop_purge_because_one_removable_curse_among_not_removable.json",
        handler_area="shop",
        phase="phase_2",
        tags=("act2", "watcher", "shop_purge", "curse_management"),
        notes="Shop regression with mixed removable and sticky curses.",
    ),
    _CaseSpec(
        case_id="ironclad_card_reward_take",
        fixture_path="tests/res/card_reward/card_reward_take.json",
        handler_area="card_reward",
        phase="phase_2",
        tags=("act1", "ironclad", "card_reward", "take"),
        notes="Baseline reward case where taking a card should remain attractive.",
    ),
    _CaseSpec(
        case_id="ironclad_card_reward_skip_upgrades",
        fixture_path="tests/res/card_reward/card_reward_skip_because_amount_and_some_in_deck_are_upgraded.json",
        handler_area="card_reward",
        phase="phase_2",
        tags=("act1", "ironclad", "card_reward", "skip"),
        notes="Reward case that stresses duplicate and upgrade-aware skip logic.",
    ),
    _CaseSpec(
        case_id="silent_card_reward_potion_take",
        fixture_path="tests/res/card_reward/card_reward_potion_take.json",
        handler_area="card_reward",
        phase="phase_2",
        tags=("act1", "silent", "card_reward", "potion"),
        notes="Silent reward case where potion-generated picks constrain skip behavior.",
    ),
    _CaseSpec(
        case_id="ironclad_path_act_two_start",
        fixture_path="tests/res/path/path_act_two_start.json",
        handler_area="path",
        phase="phase_3_future",
        tags=("act2", "ironclad", "pathing"),
        notes="Forward-looking path benchmark for upcoming map advisor work.",
    ),
    _CaseSpec(
        case_id="defect_echo_form_ready",
        fixture_path="tests/res/battles/powers/echo_form_ready.json",
        handler_area="battle",
        phase="phase_4_future",
        tags=("act2", "defect", "battle", "future_phase"),
        notes="Defect benchmark slot to keep future battle-meta coverage visible.",
    ),
)


@lru_cache(maxsize=None)
def _load_fixture_metadata(fixture_path: str) -> dict[str, Any]:
    path = Path(ROOT_DIR) / fixture_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    game_state = payload["game_state"]
    seed_number = int(game_state["seed"])
    raw_character = str(game_state["class"])
    return {
        "seed": get_seed_string(seed_number),
        "character": _parse_character(raw_character),
        "act": int(game_state["act"]),
        "floor": int(game_state["floor"]),
        "room_type": str(game_state["room_type"]),
    }


def _parse_character(raw_character: str) -> Character:
    if raw_character not in _CHARACTER_NAME_MAP:
        raise ValueError(f"Unsupported fixture character value: {raw_character}")
    return _CHARACTER_NAME_MAP[raw_character]


def _build_case(spec: _CaseSpec) -> LlmBenchmarkCase:
    metadata = _load_fixture_metadata(spec.fixture_path)
    character = metadata["character"]
    return LlmBenchmarkCase(
        case_id=spec.case_id,
        fixture_path=spec.fixture_path,
        handler_area=spec.handler_area,
        phase=spec.phase,
        seed=metadata["seed"],
        character=character,
        recommended_strategy=_DEFAULT_STRATEGY_BY_CHARACTER[character],
        act=metadata["act"],
        floor=metadata["floor"],
        room_type=metadata["room_type"],
        tags=spec.tags,
        notes=spec.notes,
    )


FIXED_LLM_BENCHMARK_SUITE: tuple[LlmBenchmarkCase, ...] = tuple(_build_case(spec) for spec in _SUITE_SPECS)


def get_fixed_llm_benchmark_suite(
        handler_areas: Iterable[str] | None = None,
        characters: Iterable[Character] | None = None,
        include_future_phases: bool = True,
) -> list[LlmBenchmarkCase]:
    area_filter = {area.lower() for area in handler_areas} if handler_areas is not None else None
    character_filter = set(characters) if characters is not None else None

    cases: list[LlmBenchmarkCase] = []
    for case in FIXED_LLM_BENCHMARK_SUITE:
        if area_filter is not None and case.handler_area.lower() not in area_filter:
            continue
        if character_filter is not None and case.character not in character_filter:
            continue
        if not include_future_phases and case.phase.endswith("_future"):
            continue
        cases.append(case)
    return cases


def group_suite_by_strategy_key(cases: Sequence[LlmBenchmarkCase]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for case in cases:
        grouped.setdefault(case.recommended_strategy, []).append(case.seed)
    return grouped


def summarize_benchmark_suite(cases: Sequence[LlmBenchmarkCase] | None = None) -> dict[str, Any]:
    selected_cases = list(FIXED_LLM_BENCHMARK_SUITE if cases is None else cases)

    characters_covered = sorted({case.character.value for case in selected_cases})
    handler_areas_covered = sorted({case.handler_area for case in selected_cases})
    acts_covered = sorted({case.act for case in selected_cases})
    phases_covered = sorted({case.phase for case in selected_cases})

    by_character: dict[str, int] = {}
    by_handler_area: dict[str, int] = {}
    for case in selected_cases:
        by_character[case.character.value] = by_character.get(case.character.value, 0) + 1
        by_handler_area[case.handler_area] = by_handler_area.get(case.handler_area, 0) + 1

    all_characters = {character.value for character in Character}
    missing_characters = sorted(all_characters.difference(characters_covered))

    return {
        "total_cases": len(selected_cases),
        "characters_covered": characters_covered,
        "missing_characters": missing_characters,
        "handler_areas_covered": handler_areas_covered,
        "acts_covered": acts_covered,
        "phases_covered": phases_covered,
        "by_character": by_character,
        "by_handler_area": by_handler_area,
    }
