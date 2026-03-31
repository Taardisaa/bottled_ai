import copy
import unittest

from rs.llm.integration.astrolabe_transform_context import build_astrolabe_transform_agent_context
from rs.llm.integration.boss_reward_context import build_boss_reward_agent_context
from rs.llm.integration.combat_reward_context import build_combat_reward_agent_context
from test_helpers.resources import load_resource_state


class TestRewardContexts(unittest.TestCase):
    def test_combat_reward_context_preserves_reward_order(self):
        state = load_resource_state("/combat_reward/combat_reward_several_rewards.json")

        context = build_combat_reward_agent_context(state, "CombatRewardHandler")

        reward_summaries = context.extras["reward_summaries"]
        self.assertEqual(["GOLD", "RELIC", "POTION", "CARD"], [row["reward_type"] for row in reward_summaries])
        self.assertEqual([0, 1, 2, 3], [row["choice_index"] for row in reward_summaries])

    def test_boss_reward_context_marks_choice_metadata_mismatch(self):
        state = load_resource_state("/relics/boss_reward_first_is_best.json")
        mismatched = copy.deepcopy(state.json)
        mismatched["game_state"]["choice_list"] = ["wrong one", "wrong two", "wrong three"]
        state = load_resource_state("/relics/boss_reward_first_is_best.json")
        state.json = mismatched

        context = build_boss_reward_agent_context(state, "BossRewardHandler")

        self.assertTrue(context.extras["choice_metadata_mismatch"])
        self.assertEqual("wrong one", context.extras["boss_relic_options"][0]["choice_name"])

    def test_astrolabe_context_tracks_remaining_picks(self):
        state = load_resource_state("/relics/boss_reward_astrolabe.json")

        context = build_astrolabe_transform_agent_context(state, "AstrolabeTransformHandler")

        self.assertEqual(3, context.extras["num_cards"])
        self.assertEqual(1, len(context.extras["selected_cards"]))
        self.assertEqual(2, context.extras["picks_remaining"])


if __name__ == "__main__":
    unittest.main()
