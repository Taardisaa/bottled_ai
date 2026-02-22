import types
import unittest
from unittest.mock import patch

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.event_llm_provider import EventLlmProvider


class FakeStructuredResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class TestEventLlmProvider(unittest.TestCase):
    def test_propose_reads_pydantic_llm_response(self):
        fake_module = types.SimpleNamespace(
            ask_llm_once=lambda **kwargs: (
                FakeStructuredResponse({
                    "proposed_command": "choose 0",
                    "confidence": 0.61,
                    "explanation": "safe path",
                }),
                77,
            )
        )
        context = AgentContext(
            handler_name="CommonEventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a", "b"],
            game_state={"event_name": "Dummy"},
        )

        with patch.dict("sys.modules", {"rs.utils.llm_utils": fake_module}):
            provider = EventLlmProvider(model="gpt-5-mini")
            proposal = provider.propose(context)

        self.assertEqual("choose 0", proposal.proposed_command)
        self.assertEqual(0.61, proposal.confidence)
        self.assertEqual("safe path", proposal.explanation)
        self.assertEqual(77, proposal.metadata["token_total"])

    def test_propose_reads_structured_llm_response(self):
        fake_module = types.SimpleNamespace(
            ask_llm_once=lambda **kwargs: (
                {
                    "proposed_command": "choose 1",
                    "confidence": 0.72,
                    "explanation": "pick reward",
                },
                123,
            )
        )
        context = AgentContext(
            handler_name="CommonEventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a", "b"],
            game_state={"event_name": "Dummy"},
        )

        with patch.dict("sys.modules", {"rs.utils.llm_utils": fake_module}):
            provider = EventLlmProvider(model="gpt-5-mini")
            proposal = provider.propose(context)

        self.assertEqual("choose 1", proposal.proposed_command)
        self.assertEqual(0.72, proposal.confidence)
        self.assertEqual("pick reward", proposal.explanation)
        self.assertEqual(123, proposal.metadata["token_total"])


if __name__ == "__main__":
    unittest.main()
