import unittest

from rs.llm.integration.campfire_context import build_campfire_agent_context
from test_helpers.resources import load_resource_state


class TestCampfireContext(unittest.TestCase):
    def test_context_includes_option_flags(self):
        state = load_resource_state("/campfire/campfire_dig.json")

        context = build_campfire_agent_context(state, "CampfireHandler")

        self.assertTrue(context.extras["campfire_option_flags"]["rest"])
        self.assertTrue(context.extras["campfire_option_flags"]["smith"])
        self.assertTrue(context.extras["campfire_option_flags"]["dig"])
        self.assertTrue(context.extras["campfire_option_flags"]["toke"])
        self.assertTrue(context.extras["campfire_option_flags"]["recall"])

    def test_context_includes_relic_counters(self):
        state = load_resource_state("/campfire/campfire_girya_smith_because_counter.json")

        context = build_campfire_agent_context(state, "CampfireHandler")

        self.assertIn("girya", context.extras["relic_counters"])
        self.assertIn("relic_names", context.extras)
        self.assertIn("deck_profile", context.extras)

    def test_single_option_recall_state_is_preserved(self):
        state = load_resource_state("/campfire/campfire_default_because_options_blocked.json")

        context = build_campfire_agent_context(state, "CampfireHandler")

        self.assertEqual(["recall"], context.choice_list)
        self.assertTrue(context.extras["campfire_option_flags"]["recall"])


if __name__ == "__main__":
    unittest.main()
