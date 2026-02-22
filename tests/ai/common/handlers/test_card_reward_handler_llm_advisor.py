import unittest

from rs.common.handlers.card_reward.common_card_reward_handler import CommonCardRewardHandler
from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.config import LlmConfig
from rs.llm.orchestrator import AIPlayerAgent
from test_helpers.resources import load_resource_state


class StaticDecisionAgent(BaseAgent):
    def __init__(self, command: str | None):
        super().__init__("static_card_reward_advisor")
        self.command = command

    def _decide(self, context: AgentContext):
        return {
            "proposed_command": self.command,
            "confidence": 0.9,
            "explanation": "test_advisor",
            "required_tools_used": [],
        }


class TestCardRewardHandlerLlmAdvisor(unittest.TestCase):
    def test_advisor_choose_command_overrides_deterministic_choice(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("CardRewardHandler", StaticDecisionAgent("choose 1"))

        handler = CommonCardRewardHandler(
            cards_desired_for_deck={"perfected strike": 1},
            advisor_orchestrator=orchestrator,
        )
        state = load_resource_state("/card_reward/card_reward_take.json")

        action = handler.handle(state)

        self.assertEqual(["choose 1", "wait 30"], action.commands)

    def test_invalid_advisor_command_falls_back_to_deterministic(self):
        orchestrator = AIPlayerAgent(config=LlmConfig(enabled=True, telemetry_enabled=False))
        orchestrator.register_agent("CardRewardHandler", StaticDecisionAgent("choose 99"))

        handler = CommonCardRewardHandler(
            cards_desired_for_deck={"perfected strike": 1},
            advisor_orchestrator=orchestrator,
        )
        state = load_resource_state("/card_reward/card_reward_take.json")

        action = handler.handle(state)

        self.assertEqual(["skip", "proceed"], action.commands)


if __name__ == "__main__":
    unittest.main()
