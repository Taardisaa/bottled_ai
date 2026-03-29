import unittest

from rs.llm.integration.event_context import build_event_agent_context
from test_helpers.resources import load_resource_state


class TestEventContextBuilder(unittest.TestCase):
    def test_build_event_context_extracts_expected_fields(self):
        state = load_resource_state("event/event_cleric_heal.json")

        context = build_event_agent_context(state, "CommonEventHandler")

        self.assertEqual("CommonEventHandler", context.handler_name)
        self.assertEqual("EVENT", context.screen_type)
        self.assertEqual("The Cleric", context.game_state["event_name"])
        self.assertIn("choose", context.available_commands)
        self.assertIn("heal", [c.lower() for c in context.choice_list])
        self.assertEqual(3, len(context.game_state["event_options"]))
        self.assertEqual(0, context.game_state["event_options"][0]["choice_index"])
        self.assertEqual("Heal", context.game_state["event_options"][0]["label"])
        self.assertFalse(context.game_state["event_options"][0]["disabled"])


if __name__ == "__main__":
    unittest.main()
