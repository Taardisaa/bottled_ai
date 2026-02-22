import unittest

from rs.common.handlers.common_event_handler import CommonEventHandler
from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.config import LlmConfig
from rs.llm.orchestrator import AIPlayerAgent
from test_helpers.resources import load_resource_state


class StaticDecisionAgent(BaseAgent):
    def __init__(self, command: str | None):
        super().__init__("static_event_advisor")
        self.command = command

    def _decide(self, context: AgentContext):
        return {
            "proposed_command": self.command,
            "confidence": 0.9,
            "explanation": "test_advisor",
            "required_tools_used": [],
        }


class TestEventHandlerLlmAdvisor(unittest.TestCase):
    def test_advisor_command_overrides_deterministic_choice(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(telemetry_enabled=False))
        orchestrator.register_agent("EventHandler", StaticDecisionAgent("choose 1"))

        handler = CommonEventHandler(
            removal_priority_list=[],
            cards_desired_for_deck={},
            advisor_orchestrator=orchestrator,
        )
        state = load_resource_state("/event/divine_fountain.json")

        action = handler.handle(state)

        self.assertEqual(["choose 1", "wait 30"], action.commands)

    def test_invalid_advisor_command_falls_back_to_deterministic(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(telemetry_enabled=False))
        orchestrator.register_agent("EventHandler", StaticDecisionAgent("choose 99"))

        handler = CommonEventHandler(
            removal_priority_list=[],
            cards_desired_for_deck={},
            advisor_orchestrator=orchestrator,
        )
        state = load_resource_state("/event/divine_fountain.json")

        action = handler.handle(state)

        self.assertEqual(["choose 0", "wait 30"], action.commands)


if __name__ == "__main__":
    unittest.main()
