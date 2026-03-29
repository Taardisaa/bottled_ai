from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Sequence

from rs.llm.agents.base_agent import AgentDecision
from rs.llm.agents.battle_meta_advisor_agent import BattleMetaDecision
from rs.llm.benchmark_suite import LlmBenchmarkCase, get_fixed_llm_benchmark_suite, load_benchmark_case_state
from rs.llm.integration.battle_context import build_battle_meta_agent_context
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.llm.integration.event_context import build_event_agent_context
from rs.llm.integration.map_context import build_map_agent_context
from rs.llm.integration.shop_purchase_context import build_shop_purchase_agent_context
from rs.llm.runtime import get_battle_meta_advisor, get_event_orchestrator
from rs.machine.ai_strategy import AiStrategy
from rs.machine.handlers.handler import Handler
from rs.machine.state import GameState

_HANDLER_KEYS_BY_AREA = {
    "event": "EventHandler",
    "shop": "ShopPurchaseHandler",
    "card_reward": "CardRewardHandler",
    "path": "MapHandler",
}

_HANDLER_MODULE_HINTS = {
    "event": "event_handler",
    "shop": "shop_purchase_handler",
    "card_reward": "card_reward_handler",
    "path": "map_handler",
    "battle": "battle_handler",
}


@dataclass
class OfflineBenchmarkCaseResult:
    case_id: str
    handler_area: str
    phase: str
    seed: str
    recommended_strategy: str
    baseline_commands: list[str]
    assisted_commands: list[str]
    advisor_output: str | None
    advisor_used: bool
    fallback_used: bool
    changed: bool
    confidence: float | None = None
    latency_ms: int | None = None
    token_total: int | None = None
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _NullOrchestrator:
    def decide(self, handler_name: str, context: Any) -> None:
        return None


class _FixedDecisionOrchestrator:
    def __init__(self, decision: AgentDecision | None):
        self._decision = decision

    def decide(self, handler_name: str, context: Any) -> AgentDecision | None:
        return self._decision


class _NoOpBattleMetaAdvisor:
    def decide(self, context: Any) -> BattleMetaDecision:
        deterministic_profile = str(context.extras.get("deterministic_profile", "general"))
        return BattleMetaDecision(
            comparator_profile=deterministic_profile,
            confidence=0.0,
            explanation="benchmark_noop_battle_meta_advisor",
            required_tools_used=[],
            fallback_recommended=True,
            metadata={"fallback_reason": "benchmark_noop"},
        )


class _FixedBattleMetaAdvisor:
    def __init__(self, decision: BattleMetaDecision):
        self._decision = decision

    def decide(self, context: Any) -> BattleMetaDecision:
        return self._decision


def run_offline_llm_benchmark_suite(
        cases: Sequence[LlmBenchmarkCase] | None = None,
        orchestrator: Any | None = None,
        battle_meta_advisor: Any | None = None,
        include_future_phases: bool = True,
) -> list[OfflineBenchmarkCaseResult]:
    selected_cases = list(get_fixed_llm_benchmark_suite(include_future_phases=include_future_phases) if cases is None else cases)
    resolved_orchestrator = get_event_orchestrator() if orchestrator is None else orchestrator
    resolved_battle_meta_advisor = get_battle_meta_advisor() if battle_meta_advisor is None else battle_meta_advisor

    results: list[OfflineBenchmarkCaseResult] = []
    for case in selected_cases:
        state = load_benchmark_case_state(case)
        results.append(
            _run_single_case(
                case=case,
                state=state,
                orchestrator=resolved_orchestrator,
                battle_meta_advisor=resolved_battle_meta_advisor,
            )
        )
    return results


def summarize_offline_benchmark_results(results: Sequence[OfflineBenchmarkCaseResult]) -> dict[str, Any]:
    selected_results = list(results)
    by_handler_area: dict[str, dict[str, Any]] = {}

    total_latency = 0
    latency_count = 0
    total_confidence = 0.0
    confidence_count = 0
    total_token_total = 0
    token_count = 0

    for result in selected_results:
        area_summary = by_handler_area.setdefault(
            result.handler_area,
            {
                "total_cases": 0,
                "changed_cases": 0,
                "advisor_used_cases": 0,
                "fallback_cases": 0,
            },
        )
        area_summary["total_cases"] += 1
        if result.changed:
            area_summary["changed_cases"] += 1
        if result.advisor_used:
            area_summary["advisor_used_cases"] += 1
        if result.fallback_used:
            area_summary["fallback_cases"] += 1

        if result.latency_ms is not None:
            total_latency += result.latency_ms
            latency_count += 1
        if result.confidence is not None:
            total_confidence += result.confidence
            confidence_count += 1
        if result.token_total is not None:
            total_token_total += result.token_total
            token_count += 1

    total_cases = len(selected_results)
    changed_cases = sum(1 for result in selected_results if result.changed)
    advisor_used_cases = sum(1 for result in selected_results if result.advisor_used)
    fallback_cases = sum(1 for result in selected_results if result.fallback_used)

    return {
        "total_cases": total_cases,
        "changed_cases": changed_cases,
        "advisor_used_cases": advisor_used_cases,
        "fallback_cases": fallback_cases,
        "unchanged_cases": total_cases - changed_cases,
        "changed_rate": 0.0 if total_cases == 0 else changed_cases / total_cases,
        "advisor_used_rate": 0.0 if total_cases == 0 else advisor_used_cases / total_cases,
        "fallback_rate": 0.0 if total_cases == 0 else fallback_cases / total_cases,
        "avg_latency_ms": None if latency_count == 0 else round(total_latency / latency_count, 2),
        "avg_confidence": None if confidence_count == 0 else round(total_confidence / confidence_count, 4),
        "avg_token_total": None if token_count == 0 else round(total_token_total / token_count, 2),
        "by_handler_area": by_handler_area,
    }


def serialize_offline_benchmark_results(results: Sequence[OfflineBenchmarkCaseResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _run_single_case(
        case: LlmBenchmarkCase,
        state: GameState,
        orchestrator: Any,
        battle_meta_advisor: Any,
) -> OfflineBenchmarkCaseResult:
    baseline_handler = _build_handler_for_case(case, state)
    baseline_handler = _configure_handler_without_advisor(baseline_handler)
    baseline_commands = list(baseline_handler.handle(state).commands)

    if case.handler_area == "battle":
        decision, latency_ms = _evaluate_battle_meta_decision(battle_meta_advisor, baseline_handler, state)
        assisted_handler = _configure_handler_with_battle_decision(_build_handler_for_case(case, state), decision)
        advisor_output = decision.comparator_profile
        fallback_used = bool(decision.metadata.get("fallback_reason"))
        advisor_used = not fallback_used
        confidence = decision.confidence
        token_total = _coerce_int(decision.metadata.get("token_total"))
        explanation = decision.explanation
        metadata = dict(decision.metadata)
    else:
        decision, latency_ms = _evaluate_orchestrator_decision(orchestrator, case, baseline_handler, state)
        assisted_handler = _configure_handler_with_orchestrator_decision(_build_handler_for_case(case, state), decision)
        advisor_output = None if decision is None else decision.proposed_command
        fallback_used = decision is None
        advisor_used = decision is not None
        confidence = None if decision is None else decision.confidence
        token_total = None if decision is None else _coerce_int(decision.metadata.get("token_total"))
        explanation = "" if decision is None else decision.explanation
        metadata = {} if decision is None else dict(decision.metadata)

    assisted_commands = list(assisted_handler.handle(state).commands)
    return OfflineBenchmarkCaseResult(
        case_id=case.case_id,
        handler_area=case.handler_area,
        phase=case.phase,
        seed=case.seed,
        recommended_strategy=case.recommended_strategy,
        baseline_commands=baseline_commands,
        assisted_commands=assisted_commands,
        advisor_output=advisor_output,
        advisor_used=advisor_used,
        fallback_used=fallback_used,
        changed=baseline_commands != assisted_commands,
        confidence=confidence,
        latency_ms=latency_ms,
        token_total=token_total,
        explanation=explanation,
        metadata=metadata,
    )


def _evaluate_orchestrator_decision(
        orchestrator: Any,
        case: LlmBenchmarkCase,
        handler: Handler,
        state: GameState,
) -> tuple[AgentDecision | None, int]:
    handler_key = _HANDLER_KEYS_BY_AREA[case.handler_area]
    context = _build_handler_context(case, handler, state)
    started_at = time.perf_counter()
    decision = orchestrator.decide(handler_key, context)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return decision, latency_ms


def _evaluate_battle_meta_decision(
        battle_meta_advisor: Any,
        handler: Handler,
        state: GameState,
) -> tuple[BattleMetaDecision, int]:
    deterministic_profile = handler.select_comparator_profile_key(state)
    available_profiles = handler.get_available_comparator_profile_keys(state)
    context = build_battle_meta_agent_context(
        state=state,
        handler_name=type(handler).__name__,
        deterministic_profile=deterministic_profile,
        available_profiles=available_profiles,
    )
    started_at = time.perf_counter()
    decision = battle_meta_advisor.decide(context)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return decision, latency_ms


def _build_handler_context(case: LlmBenchmarkCase, handler: Handler, state: GameState) -> Any:
    handler_name = type(handler).__name__
    if case.handler_area == "event":
        return build_event_agent_context(state, handler_name)
    if case.handler_area == "shop":
        return build_shop_purchase_agent_context(state, handler_name)
    if case.handler_area == "card_reward":
        return build_card_reward_agent_context(state, handler_name)
    if case.handler_area == "path":
        return build_map_agent_context(state, handler_name, handler.config)
    raise ValueError(f"Unsupported orchestrator benchmark handler area: {case.handler_area}")


def _build_handler_for_case(case: LlmBenchmarkCase, state: GameState) -> Handler:
    if case.handler_area == "shop":
        return _build_smart_shop_purchase_handler()
    if case.handler_area == "path":
        return _build_common_map_handler()

    strategy = _get_strategies_by_key().get(case.recommended_strategy)
    if strategy is None:
        raise ValueError(f"Unsupported recommended strategy: {case.recommended_strategy}")

    for handler in strategy.handlers:
        if not _handler_matches_area(handler, case.handler_area):
            continue
        if not handler.can_handle(state):
            continue
        return deepcopy(handler)

    raise ValueError(
        f"Could not find a matching handler for area '{case.handler_area}' in strategy '{case.recommended_strategy}'"
    )


def _handler_matches_area(handler: Handler, handler_area: str) -> bool:
    module_name = handler.__class__.__module__.lower()
    area_hint = _HANDLER_MODULE_HINTS[handler_area]
    return area_hint in module_name


def _configure_handler_without_advisor(handler: Handler) -> Handler:
    configured = deepcopy(handler)
    if hasattr(configured, "advisor_orchestrator"):
        configured.advisor_orchestrator = _NullOrchestrator()
    if hasattr(configured, "battle_meta_advisor"):
        configured.battle_meta_advisor = _NoOpBattleMetaAdvisor()
    return configured


def _configure_handler_with_orchestrator_decision(handler: Handler, decision: AgentDecision | None) -> Handler:
    configured = deepcopy(handler)
    if hasattr(configured, "advisor_orchestrator"):
        configured.advisor_orchestrator = _FixedDecisionOrchestrator(decision)
    return configured


def _configure_handler_with_battle_decision(handler: Handler, decision: BattleMetaDecision) -> Handler:
    configured = deepcopy(handler)
    if hasattr(configured, "battle_meta_advisor"):
        configured.battle_meta_advisor = _FixedBattleMetaAdvisor(decision)
    return configured


def _coerce_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _get_strategies_by_key() -> dict[str, AiStrategy]:
    # Import strategy modules lazily to avoid pulling rs.llm back in while
    # rs.llm.__init__ is still evaluating this module.
    from rs.ai.peaceful_pummeling.peaceful_pummeling import PEACEFUL_PUMMELING
    from rs.ai.pwnder_my_orbs.pwnder_my_orbs import PWNDER_MY_ORBS
    from rs.ai.requested_strike.requested_strike import REQUESTED_STRIKE
    from rs.ai.shivs_and_giggles.shivs_and_giggles import SHIVS_AND_GIGGLES

    return {
        "peaceful_pummeling": PEACEFUL_PUMMELING,
        "pwnder_my_orbs": PWNDER_MY_ORBS,
        "requested_strike": REQUESTED_STRIKE,
        "shivs_and_giggles": SHIVS_AND_GIGGLES,
    }


def _build_smart_shop_purchase_handler() -> Handler:
    from rs.ai.smart_agent.handlers.shop_purchase_handler import ShopPurchaseHandler as SmartShopPurchaseHandler

    return SmartShopPurchaseHandler()


def _build_common_map_handler() -> Handler:
    from rs.common.handlers.common_map_handler import CommonMapHandler

    return CommonMapHandler()
