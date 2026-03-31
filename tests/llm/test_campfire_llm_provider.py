import unittest

from rs.llm.integration.campfire_context import build_campfire_agent_context
from rs.llm.providers.campfire_llm_provider import CampfireLlmProvider
from test_helpers.resources import load_resource_state


class TestCampfireLlmProvider(unittest.TestCase):
    def test_prompt_contains_campfire_payload(self):
        state = load_resource_state("/campfire/campfire_dig.json")
        context = build_campfire_agent_context(state, "CampfireHandler")

        prompt = CampfireLlmProvider(model="gpt-5-mini")._build_prompt(context, {"recent_step_summaries": []})

        self.assertIn('"campfire_options"', prompt)
        self.assertIn('"campfire_option_flags"', prompt)
        self.assertIn('"relic_counters"', prompt)
        self.assertIn("Do not use fixed hp thresholds", prompt)


if __name__ == "__main__":
    unittest.main()
