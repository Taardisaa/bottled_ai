import unittest
import time
from typing import cast

from rs.llm.agents.base_agent import AgentContext, AgentDecision, BaseAgent
from rs.llm.config import LlmConfig
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
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("EventHandler", ExplodingAgent())
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = orchestrator.decide("EventHandler", context)

        decision = cast(AgentDecision, decision)
        self.assertTrue(decision.fallback_recommended)
        self.assertIsNone(decision.proposed_command)

    def test_orchestrator_respects_global_enable_flag(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=False, telemetry_enabled=False))
        orchestrator.register_agent("EventHandler", StubAgent({"proposed_command": "choose 0", "confidence": 1.0}))
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        self.assertIsNone(orchestrator.decide("EventHandler", context))

    def test_orchestrator_respects_enabled_handlers(self):
        orchestrator = AIPlayerAgent(
            config=LlmConfig(enabled=True, enabled_handlers=["ShopPurchaseHandler"], telemetry_enabled=False)
        )
        orchestrator.register_agent("EventHandler", StubAgent({"proposed_command": "choose 0", "confidence": 1.0}))
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        self.assertIsNone(orchestrator.decide("EventHandler", context))

    def test_orchestrator_falls_back_on_low_confidence(self):
        orchestrator = AIPlayerAgent(
            config=LlmConfig(enabled=True, confidence_threshold=0.8, telemetry_enabled=False)
        )
        orchestrator.register_agent("EventHandler", StubAgent({"proposed_command": "choose 0", "confidence": 0.2}))
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = cast(AgentDecision, orchestrator.decide("EventHandler", context))
        self.assertTrue(decision.fallback_recommended)
        self.assertEqual("low_confidence", decision.metadata["validation_error"])

    def test_orchestrator_falls_back_on_token_budget(self):
        orchestrator = AIPlayerAgent(
            config=LlmConfig(enabled=True, max_tokens_per_decision=10, telemetry_enabled=False)
        )
        orchestrator.register_agent(
            "EventHandler",
            StubAgent({
                "proposed_command": "choose 0",
                "confidence": 0.95,
                "metadata": {"token_total": 999},
            }),
        )
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = cast(AgentDecision, orchestrator.decide("EventHandler", context))
        self.assertTrue(decision.fallback_recommended)
        self.assertEqual("token_budget_exceeded", decision.metadata["validation_error"])

    def test_orchestrator_falls_back_on_timeout(self):
        class SlowAgent(BaseAgent):
            def __init__(self):
                super().__init__("slow")

            def _decide(self, context):
                time.sleep(0.05)
                return {"proposed_command": "choose 0", "confidence": 1.0}

        orchestrator = AIPlayerAgent(
            config=LlmConfig(enabled=True, timeout_ms=5, telemetry_enabled=False)
        )
        orchestrator.register_agent("EventHandler", SlowAgent())
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = cast(AgentDecision, orchestrator.decide("EventHandler", context))
        self.assertTrue(decision.fallback_recommended)
        self.assertEqual("timeout", decision.metadata["validation_error"])

    def test_orchestrator_ignores_timeout_when_set_to_negative_one(self):
        class SlowButValidAgent(BaseAgent):
            def __init__(self):
                super().__init__("slow_valid")

            def _decide(self, context):
                time.sleep(0.03)
                return {"proposed_command": "choose 0", "confidence": 1.0}

        orchestrator = AIPlayerAgent(
            config=LlmConfig(enabled=True, timeout_ms=-1, telemetry_enabled=False)
        )
        orchestrator.register_agent("EventHandler", SlowButValidAgent())
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = cast(AgentDecision, orchestrator.decide("EventHandler", context))
        self.assertFalse(decision.fallback_recommended)
        self.assertEqual("choose 0", decision.proposed_command)

    def test_orchestrator_retry_count_is_configurable(self):
        class FlakyAgent(BaseAgent):
            def __init__(self):
                super().__init__("flaky")
                self.calls = 0

            def _decide(self, context):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first call fails")
                return {"proposed_command": "choose 0", "confidence": 1.0}

        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        no_retry_agent = FlakyAgent()
        orchestrator_no_retry = AIPlayerAgent(
            config=LlmConfig(enabled=True, max_retries=0, telemetry_enabled=False)
        )
        orchestrator_no_retry.register_agent("EventHandler", no_retry_agent)
        decision_no_retry = cast(AgentDecision, orchestrator_no_retry.decide("EventHandler", context))
        self.assertTrue(decision_no_retry.fallback_recommended)

        one_retry_agent = FlakyAgent()
        orchestrator_one_retry = AIPlayerAgent(
            config=LlmConfig(enabled=True, max_retries=1, telemetry_enabled=False)
        )
        orchestrator_one_retry.register_agent("EventHandler", one_retry_agent)
        decision_one_retry = cast(AgentDecision, orchestrator_one_retry.decide("EventHandler", context))
        self.assertFalse(decision_one_retry.fallback_recommended)
        self.assertEqual("choose 0", decision_one_retry.proposed_command)


if __name__ == "__main__":
    unittest.main()
