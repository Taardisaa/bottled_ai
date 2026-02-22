from __future__ import annotations

from typing import Any, Dict

from rs.llm.agents.base_agent import AgentContext, BaseAgent


class ShopPurchaseAdvisorAgent(BaseAgent):
    """Phase-2 placeholder advisor for shop purchase decisions."""

    def __init__(self, timeout_ms: int = 1500):
        super().__init__(name="shop_purchase_advisor", timeout_ms=timeout_ms)

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        return {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "phase_2_placeholder",
            "required_tools_used": [],
            "fallback_recommended": True,
        }
