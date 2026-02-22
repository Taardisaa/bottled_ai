from __future__ import annotations

from typing import Any, Dict

from rs.llm.agents.base_agent import AgentContext, BaseAgent


class CardRewardAdvisorAgent(BaseAgent):
    """Phase-2 placeholder advisor for card reward decisions."""

    def __init__(self, timeout_ms: int = 1500):
        super().__init__(name="card_reward_advisor", timeout_ms=timeout_ms)

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        return {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "phase_2_placeholder",
            "required_tools_used": [],
            "fallback_recommended": True,
        }
