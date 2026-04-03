import types
import unittest
from unittest.mock import patch

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.event_llm_provider import EventDecisionSchema, EventLlmProvider
from test_helpers.resources import load_resource_state
from rs.llm.integration.event_context import build_event_agent_context


class TestEventLlmProvider(unittest.TestCase):
    def test_propose_reads_pydantic_llm_response(self):
        captured = {}

        def fake_ask_llm_once(**kwargs):
            captured["struct"] = kwargs.get("struct")
            return (
                EventDecisionSchema(
                    proposed_command="choose 0",
                    confidence=0.61,
                    explanation="safe path",
                ),
                77,
            )

        fake_module = types.SimpleNamespace(
            ask_llm_once=fake_ask_llm_once,
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
        self.assertIs(EventDecisionSchema, captured["struct"])

    def test_propose_accepts_dict_by_validating_into_schema(self):
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

    def test_propose_coerces_bare_index_into_choose_command(self):
        fake_module = types.SimpleNamespace(
            ask_llm_once=lambda **kwargs: (
                {
                    "proposed_command": "0",
                    "confidence": 0.95,
                    "explanation": "pick first event option",
                },
                55,
            )
        )
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose", "wait", "state"],
            choice_list=["gold", "leave"],
            game_state={"event_name": "Dummy"},
        )

        with patch.dict("sys.modules", {"rs.utils.llm_utils": fake_module}):
            provider = EventLlmProvider(model="gpt-5-mini")
            proposal = provider.propose(context)

        self.assertEqual("choose 0", proposal.proposed_command)
        self.assertEqual(0.95, proposal.confidence)

    def test_build_prompt_renders_event_options_and_prompt_safe_status(self):
        state = load_resource_state("event/event_neow.json")
        context = build_event_agent_context(state, "EventHandler")
        context.extras.update(
            {
                "recent_llm_decisions": "choose 0 at prior event",
                "retrieved_episodic_memories": "none",
                "retrieved_semantic_memories": "none",
            }
        )

        prompt = EventLlmProvider(model="gpt-5-mini")._build_prompt(context)

        self.assertIn('choose <index>', prompt)
        self.assertIn('answer in short plain text using these fields', prompt)
        self.assertIn('- 0 | enabled | label="Obtain a random rare Card"', prompt)
        self.assertIn('text="[ Obtain a random rare Card ]"', prompt)
        self.assertNotIn('LangMem status:', prompt)
        self.assertNotIn('Handler:', prompt)
        self.assertNotIn('Screen:', prompt)
        self.assertNotIn('Extras:', prompt)
        self.assertNotIn('Return ONLY a JSON object', prompt)

    def test_build_prompt_falls_back_to_choice_list_when_event_options_missing(self):
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose", "wait", "state"],
            choice_list=["heal", "leave"],
            game_state={"event_name": "Fallback Event", "event_options": []},
            extras={},
        )

        prompt = EventLlmProvider(model="gpt-5-mini")._build_prompt(context)

        self.assertIn('- 0 | enabled | choice="heal"', prompt)
        self.assertIn('- 1 | enabled | choice="leave"', prompt)

    def test_build_prompt_omits_empty_memory_lines(self):
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose", "wait", "state"],
            choice_list=["heal"],
            game_state={"event_name": "Test Event", "event_options": []},
            extras={
                "retrieved_episodic_memories": "none",
                "retrieved_semantic_memories": "",
            },
        )

        prompt = EventLlmProvider(model="gpt-5-mini")._build_prompt(context)

        self.assertNotIn("Retrieved episodic memories:", prompt)
        self.assertNotIn("Retrieved semantic memories:", prompt)


if __name__ == "__main__":
    unittest.main()
