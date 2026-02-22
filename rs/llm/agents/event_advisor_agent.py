from __future__ import annotations

from typing import Any, Dict

from rs.llm.agents.base_agent import AgentContext, BaseAgent


class EventAdvisorAgent(BaseAgent):
    """Pilot event advisor.

    The current implementation is intentionally conservative: it returns no
    command so deterministic event handlers remain authoritative. This keeps the
    integration path active while we iterate on real model-backed policies.
    """

    def __init__(self, timeout_ms: int = 1500):
        super().__init__(name="event_advisor", timeout_ms=timeout_ms)

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        return {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "pilot_no_model_bound",
            "required_tools_used": [],
            "fallback_recommended": True,
            "metadata": {
                "phase": "phase_1_event_pilot",
                "event_name": context.game_state.get("event_name", "unknown"),
            },
        }
