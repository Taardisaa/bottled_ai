from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.providers.event_llm_provider import EventLlmProposal, EventLlmProvider


DecisionProvider = Callable[[AgentContext], str | None]


class EventProposalProvider(Protocol):
    def propose(self, context: AgentContext) -> EventLlmProposal:
        ...


class EventAdvisorAgent(BaseAgent):
    """Pilot event advisor.

    The current implementation is intentionally conservative: it returns no
    command so deterministic event handlers remain authoritative. This keeps the
    integration path active while we iterate on real model-backed policies.
    """

    def __init__(
            self,
            timeout_ms: int = 1500,
            decision_provider: DecisionProvider | None = None,
            llm_provider: EventProposalProvider | None = None,
    ):
        super().__init__(name="event_advisor", timeout_ms=timeout_ms)
        self._decision_provider: DecisionProvider = _default_event_decision_provider \
            if decision_provider is None else decision_provider
        self._llm_provider: EventProposalProvider = EventLlmProvider() if llm_provider is None else llm_provider

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        llm_proposal: EventLlmProposal = self._llm_provider.propose(context)
        if llm_proposal.proposed_command is not None:
            metadata = {
                "phase": "phase_1_event_pilot",
                "event_name": context.game_state.get("event_name", "unknown"),
            }
            metadata.update(llm_proposal.metadata)
            return {
                "proposed_command": llm_proposal.proposed_command,
                "confidence": llm_proposal.confidence,
                "explanation": llm_proposal.explanation,
                "required_tools_used": ["llm_event_advisor"],
                "fallback_recommended": False,
                "metadata": metadata,
            }

        proposed_command = self._decision_provider(context)
        metadata = {
            "phase": "phase_1_event_pilot",
            "event_name": context.game_state.get("event_name", "unknown"),
            "fallback_reason": "llm_no_decision",
        }
        metadata.update(llm_proposal.metadata)
        return {
            "proposed_command": proposed_command,
            "confidence": 0.65 if proposed_command is not None else 0.0,
            "explanation": "rule_based_event_policy" if proposed_command is not None else "no_rule_match",
            "required_tools_used": [],
            "fallback_recommended": proposed_command is None,
            "metadata": metadata,
        }


def _default_event_decision_provider(context: AgentContext) -> str | None:
    """Conservative event rules for early pilot integration.

    Restricts decisions to base CommonEventHandler only so strategy-specific
    EventHandler subclasses keep their deterministic overrides.
    """
    if context.handler_name != "CommonEventHandler":
        return None

    event_name = str(context.game_state.get("event_name", "")).strip().lower()
    if not event_name:
        return None

    choices = [str(c).lower() for c in context.choice_list]

    if event_name == "the cleric":
        hp = float(context.game_state.get("current_hp", 0))
        max_hp = float(context.game_state.get("max_hp", 1))
        hp_pct = 0.0 if max_hp <= 0 else (hp / max_hp) * 100

        if hp_pct <= 65 and "heal" in choices:
            return "choose heal"
        if "purify" in choices:
            return "choose purify"
        if hp_pct >= 90 and "leave" in choices:
            return "choose leave"
        return "choose 0"

    if event_name in {"purifier", "the divine fountain", "upgrade shrine", "lab"}:
        return "choose 0"

    return None
