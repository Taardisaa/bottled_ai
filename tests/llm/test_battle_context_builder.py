import unittest

from rs.llm.integration.battle_context import build_battle_meta_agent_context
from test_helpers.resources import load_resource_state


class TestBattleContextBuilder(unittest.TestCase):
    def test_builds_battle_meta_context_from_fixture(self):
        state = load_resource_state(
            "battles/specific_comparator_cases/big_fight/big_fight_prioritize_power_over_damage.json"
        )

        context = build_battle_meta_agent_context(
            state=state,
            handler_name="CommonBattleHandler",
            deterministic_profile="big_fight",
            available_profiles=["big_fight", "general"],
        )

        self.assertEqual("CommonBattleHandler", context.handler_name)
        self.assertEqual(state.screen_type(), context.screen_type)
        self.assertEqual("big_fight", context.extras["deterministic_profile"])
        self.assertEqual(["big_fight", "general"], context.extras["available_profiles"])
        self.assertGreater(len(context.extras["monster_summaries"]), 0)
        self.assertIn("deck_profile", context.extras)
        self.assertIn("held_potion_names", context.extras)


if __name__ == "__main__":
    unittest.main()
