import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.generic_llm_provider import GenericLlmProvider


class TestGenericLlmProviderPrompt(unittest.TestCase):
    def test_prompt_includes_sectioned_schema_and_field_dictionary(self):
        context = AgentContext(
            handler_name="GenericHandler",
            screen_type="NONE",
            available_commands=["confirm", "cancel", "wait", "state"],
            choice_list=[],
            game_state={
                "floor": 12,
                "act": 2,
                "room_phase": "COMPLETE",
                "room_type": "EventRoom",
                "character_class": "WATCHER",
            },
            extras={
                "generic_payload": {
                    "screen": {"screen_type": "NONE"},
                    "commands": {"available_commands": ["confirm", "cancel", "wait", "state"]},
                    "game_state": {"floor": 12, "act": 2},
                    "resources": {"deck_size": 20},
                    "screen_state": {},
                },
                "sectioned_schema_explanations": {
                    "screen": "Screen explanation.",
                    "commands": "Commands explanation.",
                },
                "field_dictionary": {
                    "available_commands": {
                        "type": "list[str]",
                        "meaning": "Legal command verbs.",
                        "constraints": "Must pick from this list.",
                    }
                },
                "raw_game_state_keys": ["floor", "act", "screen_type"],
            },
        )

        prompt = GenericLlmProvider(model="gpt-5-mini")._build_prompt(context)

        self.assertIn("## Sectioned Schema Payload", prompt)
        self.assertIn("## Sectioned Explanation Blocks", prompt)
        self.assertIn("## Field Dictionary", prompt)
        self.assertIn('"screen_type": "NONE"', prompt)
        self.assertIn('"available_commands"', prompt)
        self.assertIn('"meaning": "Legal command verbs."', prompt)
        self.assertIn("## Raw Game State Key Index", prompt)
        self.assertNotIn("## Validation Feedback From Previous Rejected Proposal", prompt)
        self.assertIn('"floor"', prompt)

    def test_prompt_avoids_heuristic_policy_phrases(self):
        context = AgentContext(
            handler_name="GenericHandler",
            screen_type="NONE",
            available_commands=["wait"],
            choice_list=[],
            game_state={},
            extras={
                "generic_payload": {},
                "sectioned_schema_explanations": {},
                "field_dictionary": {},
                "raw_game_state_keys": [],
            },
        )

        prompt_lower = GenericLlmProvider(model="gpt-5-mini")._build_prompt(context).lower()
        self.assertNotIn("heuristic", prompt_lower)
        self.assertNotIn("best action", prompt_lower)
        self.assertNotIn("priority list", prompt_lower)

    def test_prompt_includes_validation_feedback_block_only_when_present(self):
        context = AgentContext(
            handler_name="GenericHandler",
            screen_type="GRID",
            available_commands=["choose"],
            choice_list=["a", "b"],
            game_state={},
            extras={
                "generic_payload": {},
                "sectioned_schema_explanations": {},
                "field_dictionary": {},
                "raw_game_state_keys": [],
            },
        )

        prompt = GenericLlmProvider(model="gpt-5-mini")._build_prompt(
            context,
            validation_feedback={
                "code": "choose_requires_index",
                "message": "choose must be in format choose <index>",
                "valid_example": "choose 0",
            },
        )
        self.assertIn("## Validation Feedback From Previous Rejected Proposal", prompt)
        self.assertIn('"code": "choose_requires_index"', prompt)
        self.assertIn('"valid_example": "choose 0"', prompt)


if __name__ == "__main__":
    unittest.main()
