import unittest

from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.orchestrator import AIPlayerAgent


class StubAgent(BaseAgent):
    def __init__(self, output):
        super().__init__("stub")
        self.output = output

    def _decide(self, context):
        return self.output


class ExplodingAgent(BaseAgent):
    def __init__(self):
        super().__init__("boom")

    def _decide(self, context):
        raise RuntimeError("boom")


class TestBaseAgent(unittest.TestCase):
    def test_unavailable_command_recommends_fallback(self):
        agent = StubAgent({"proposed_command": "choose 1", "confidence": 0.9})
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["proceed", "skip"],
            choice_list=["a", "b"],
        )

        decision = agent.decide(context)

        self.assertTrue(decision.fallback_recommended)
        self.assertEqual("command_not_available", decision.metadata["validation_error"])

    def test_confidence_is_clamped(self):
        agent = StubAgent({"proposed_command": "proceed", "confidence": 3.0})
        context = AgentContext(
            handler_name="DefaultConfirmHandler",
            screen_type="COMPLETE",
            available_commands=["proceed"],
            choice_list=[],
        )

        decision = agent.decide(context)

        self.assertEqual(1.0, decision.confidence)

    def test_orchestrator_returns_safe_fallback_on_error(self):
        orchestrator = AIPlayerAgent()
        orchestrator.register_agent("EventHandler", ExplodingAgent())
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = orchestrator.decide("EventHandler", context)

        self.assertIsNotNone(decision)
        self.assertTrue(decision.fallback_recommended)
        self.assertIsNone(decision.proposed_command)


if __name__ == "__main__":
    unittest.main()
