import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.event_advisor_agent import EventAdvisorAgent
from rs.llm.providers.event_llm_provider import EventLlmProposal


class StubLlmProvider:
    def __init__(self, command: str | None = None):
        self.command = command

    def propose(self, context: AgentContext) -> EventLlmProposal:
        return EventLlmProposal(
            proposed_command=self.command,
            confidence=0.8 if self.command is not None else 0.0,
            explanation="stub",
            metadata={"provider": "stub"},
        )


class TestEventAdvisorAgent(unittest.TestCase):
    def test_common_event_handler_proposes_cleric_decision(self):
        agent = EventAdvisorAgent(llm_provider=StubLlmProvider())
        context = AgentContext(
            handler_name="CommonEventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["heal", "purify", "leave"],
            game_state={
                "event_name": "The Cleric",
                "current_hp": 20,
                "max_hp": 100,
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose heal", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)

    def test_strategy_event_handler_keeps_fallback(self):
        agent = EventAdvisorAgent(llm_provider=StubLlmProvider())
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["0", "1"],
            game_state={"event_name": "The Divine Fountain"},
        )

        decision = agent.decide(context)

        self.assertIsNone(decision.proposed_command)
        self.assertTrue(decision.fallback_recommended)

    def test_llm_provider_command_takes_precedence(self):
        agent = EventAdvisorAgent(llm_provider=StubLlmProvider(command="choose 0"))
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["leave", "take"],
            game_state={"event_name": "Any Event"},
        )

        decision = agent.decide(context)

        self.assertEqual("choose 0", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)
        self.assertEqual(["llm_event_advisor"], decision.required_tools_used)


if __name__ == "__main__":
    unittest.main()
