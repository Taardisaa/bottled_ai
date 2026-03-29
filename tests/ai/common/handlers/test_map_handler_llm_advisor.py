import unittest
from unittest.mock import patch

from rs.common.handlers.common_map_handler import CommonMapHandler
from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.config import LlmConfig
from rs.llm.orchestrator import AIPlayerAgent
from test_helpers.resources import load_resource_state


class StaticDecisionAgent(BaseAgent):
    def __init__(self, command: str | None):
        super().__init__("static_map_advisor")
        self.command = command

    def _decide(self, context: AgentContext):
        return {
            "proposed_command": self.command,
            "confidence": 0.9,
            "explanation": "test_advisor",
            "required_tools_used": [],
        }


class TestMapHandlerLlmAdvisor(unittest.TestCase):
    def test_advisor_choose_command_overrides_deterministic_choice(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("MapHandler", StaticDecisionAgent("choose 0"))

        handler = CommonMapHandler(advisor_orchestrator=orchestrator)
        state = load_resource_state("path/path_basic.json")

        action = handler.handle(state)

        self.assertEqual(["choose 0"], action.commands)

    def test_invalid_advisor_command_falls_back_to_deterministic(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("MapHandler", StaticDecisionAgent("choose 99"))

        handler = CommonMapHandler(advisor_orchestrator=orchestrator)
        state = load_resource_state("path/path_basic.json")

        action = handler.handle(state)

        self.assertEqual(["choose 0"], action.commands)

    def test_unified_graph_mode_skips_handler_advisor_path(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("MapHandler", StaticDecisionAgent("choose 1"))

        handler = CommonMapHandler(advisor_orchestrator=orchestrator)
        state = load_resource_state("path/path_basic.json")

        with patch("rs.common.handlers.common_map_handler.is_ai_player_graph_enabled", return_value=True):
            action = handler.handle(state)

        self.assertEqual(["choose 0"], action.commands)


if __name__ == "__main__":
    unittest.main()
