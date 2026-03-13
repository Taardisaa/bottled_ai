from __future__ import annotations

from typing import Any, Dict, Protocol

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.memory_langgraph_agent import MemoryAugmentedLangGraphAgent
from rs.llm.providers.map_llm_provider import MapLlmProposal, MapLlmProvider


class MapProposalProvider(Protocol):
    def propose(self, context: AgentContext) -> MapLlmProposal:
        ...


class MapAdvisorAgent(MemoryAugmentedLangGraphAgent):
    """Map advisor anchored on deterministic path scoring plus conservative overrides."""

    def __init__(
            self,
            timeout_ms: int = 1500,
            llm_provider: MapProposalProvider | None = None,
    ):
        super().__init__(name="map_advisor", timeout_ms=timeout_ms)
        self._llm_provider: MapProposalProvider = MapLlmProvider() if llm_provider is None else llm_provider

    def propose_with_context(self, context: AgentContext) -> MapLlmProposal:
        return self._llm_provider.propose(context)

    def build_success_payload(self, context: AgentContext, proposal: MapLlmProposal) -> Dict[str, Any]:
        metadata = {
            "phase": "phase_3_map_advisor",
            "deterministic_best_command": context.extras.get("deterministic_best_command"),
            "graph_runtime": "langgraph",
            "graph_nodes": self.graph_node_names(),
        }
        metadata.update(proposal.metadata)
        return {
            "proposed_command": proposal.proposed_command,
            "confidence": proposal.confidence,
            "explanation": proposal.explanation,
            "required_tools_used": ["deterministic_path_scores", "llm_map_advisor", "langgraph_workflow"],
            "fallback_recommended": False,
            "metadata": metadata,
        }

    def build_fallback_payload(self, context: AgentContext, proposal: MapLlmProposal) -> Dict[str, Any]:
        proposed_command = _default_map_decision_provider(context)
        metadata = {
            "phase": "phase_3_map_advisor",
            "deterministic_best_command": context.extras.get("deterministic_best_command"),
            "fallback_reason": "llm_no_decision",
            "graph_runtime": "langgraph",
            "graph_nodes": self.graph_node_names(),
        }
        metadata.update(proposal.metadata)
        return {
            "proposed_command": proposed_command,
            "confidence": 0.7 if proposed_command is not None else 0.0,
            "explanation": "rule_based_map_policy" if proposed_command is not None else proposal.explanation,
            "required_tools_used": ["deterministic_path_scores", "langgraph_workflow"] if proposed_command is not None else [],
            "fallback_recommended": proposed_command is None,
            "metadata": metadata,
        }


def _default_map_decision_provider(context: AgentContext) -> str | None:
    deterministic_command = str(context.extras.get("deterministic_best_command", "")).strip().lower()
    choice_overviews_raw = context.extras.get("choice_path_overviews", [])
    if deterministic_command == "" or not isinstance(choice_overviews_raw, list):
        return None

    choice_overviews = [item for item in choice_overviews_raw if isinstance(item, dict)]
    deterministic_overview = next(
        (item for item in choice_overviews if str(item.get("choice_command", "")).strip().lower() == deterministic_command),
        None,
    )
    if deterministic_overview is None:
        return deterministic_command

    hp = float(context.game_state.get("current_hp", 0) or 0)
    max_hp = float(context.game_state.get("max_hp", 0) or 0)
    hp_ratio = 0.0 if max_hp <= 0 else hp / max_hp
    gold = float(context.game_state.get("gold", 0) or 0)
    act = int(context.game_state.get("act", 0) or 0)
    floor = int(context.game_state.get("floor", 0) or 0)

    safer_choice = _find_low_hp_safer_choice(choice_overviews, deterministic_overview, hp_ratio)
    if safer_choice is not None:
        return safer_choice

    shop_choice = _find_early_shop_choice(choice_overviews, deterministic_overview, hp_ratio, gold, act)
    if shop_choice is not None:
        return shop_choice

    elite_choice = _find_aggressive_elite_choice(choice_overviews, deterministic_overview, hp_ratio, act, floor)
    if elite_choice is not None:
        return elite_choice

    return deterministic_command


def _find_low_hp_safer_choice(
        choice_overviews: list[dict[str, Any]],
        deterministic_overview: dict[str, Any],
        hp_ratio: float,
) -> str | None:
    if hp_ratio > 0.55:
        return None

    deterministic_survivability = float(deterministic_overview.get("survivability", 0.0) or 0.0)
    deterministic_score = float(deterministic_overview.get("reward_survivability", 0.0) or 0.0)
    deterministic_elites = int(deterministic_overview.get("room_counts", {}).get("E", 0) or 0)

    candidates = [
        item for item in choice_overviews
        if int(item.get("room_counts", {}).get("E", 0) or 0) <= deterministic_elites
        and float(item.get("survivability", 0.0) or 0.0) >= deterministic_survivability + 0.08
        and float(item.get("reward_survivability", 0.0) or 0.0) >= deterministic_score - 1.25
    ]
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (float(item.get("survivability", 0.0) or 0.0), float(item.get("reward_survivability", 0.0) or 0.0)),
    )
    return str(best.get("choice_command", "")).strip().lower() or None


def _find_early_shop_choice(
        choice_overviews: list[dict[str, Any]],
        deterministic_overview: dict[str, Any],
        hp_ratio: float,
        gold: float,
        act: int,
) -> str | None:
    if gold < 150 or hp_ratio < 0.45 or act > 2:
        return None

    deterministic_shop_distance = deterministic_overview.get("shop_distance")
    deterministic_score = float(deterministic_overview.get("reward_survivability", 0.0) or 0.0)
    deterministic_survivability = float(deterministic_overview.get("survivability", 0.0) or 0.0)

    candidates = []
    for item in choice_overviews:
        shop_distance = item.get("shop_distance")
        if shop_distance is None or shop_distance > 3:
            continue
        if deterministic_shop_distance is not None and shop_distance >= deterministic_shop_distance:
            continue
        if float(item.get("reward_survivability", 0.0) or 0.0) < deterministic_score - 0.9:
            continue
        if float(item.get("survivability", 0.0) or 0.0) < deterministic_survivability - 0.05:
            continue
        candidates.append(item)

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (
            -int(item.get("shop_distance", 99) or 99),
            float(item.get("reward_survivability", 0.0) or 0.0),
        ),
    )
    return str(best.get("choice_command", "")).strip().lower() or None


def _find_aggressive_elite_choice(
        choice_overviews: list[dict[str, Any]],
        deterministic_overview: dict[str, Any],
        hp_ratio: float,
        act: int,
        floor: int,
) -> str | None:
    if hp_ratio < 0.8 or act != 1 or floor > 6:
        return None

    deterministic_elites = int(deterministic_overview.get("room_counts", {}).get("E", 0) or 0)
    deterministic_score = float(deterministic_overview.get("reward_survivability", 0.0) or 0.0)

    candidates = [
        item for item in choice_overviews
        if int(item.get("room_counts", {}).get("E", 0) or 0) > deterministic_elites
        and float(item.get("survivability", 0.0) or 0.0) >= 0.92
        and float(item.get("reward_survivability", 0.0) or 0.0) >= deterministic_score - 0.5
    ]
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (
            int(item.get("room_counts", {}).get("E", 0) or 0),
            float(item.get("reward_survivability", 0.0) or 0.0),
        ),
    )
    return str(best.get("choice_command", "")).strip().lower() or None
