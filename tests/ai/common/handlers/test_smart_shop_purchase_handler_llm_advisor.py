import unittest

from rs.ai.smart_agent.handlers.shop_purchase_handler import ShopPurchaseHandler
from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.config import LlmConfig
from rs.llm.orchestrator import AIPlayerAgent
from test_helpers.resources import load_resource_state


class StaticDecisionAgent(BaseAgent):
    def __init__(self, command: str | None):
        super().__init__("static_shop_advisor")
        self.command = command

    def _decide(self, context: AgentContext):
        return {
            "proposed_command": self.command,
            "confidence": 0.9,
            "explanation": "test_advisor",
            "required_tools_used": [],
        }


class TestSmartShopPurchaseHandlerLlmAdvisor(unittest.TestCase):
    def test_advisor_choose_command_overrides_deterministic_choice(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("ShopPurchaseHandler", StaticDecisionAgent("choose 0"))

        handler = ShopPurchaseHandler(advisor_orchestrator=orchestrator)
        state = load_resource_state("/shop/shop_buy_perfected_strike.json")

        action = handler.handle(state)

        self.assertEqual(["choose 0", "wait 30"], action.commands)

    def test_invalid_advisor_command_falls_back_to_deterministic(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("ShopPurchaseHandler", StaticDecisionAgent("choose 99"))

        handler = ShopPurchaseHandler(advisor_orchestrator=orchestrator)
        state = load_resource_state("/shop/shop_buy_perfected_strike.json")

        action = handler.handle(state)

        self.assertEqual(["choose 1", "wait 30"], action.commands)


if __name__ == "__main__":
    unittest.main()
