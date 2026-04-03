import copy
import unittest

from rs.llm.integration.astrolabe_transform_context import build_astrolabe_transform_agent_context
from rs.llm.integration.boss_reward_context import build_boss_reward_agent_context
from rs.llm.integration.combat_reward_context import build_combat_reward_agent_context
from rs.llm.integration.grid_select_context import build_grid_select_agent_context
from rs.machine.state import GameState
from test_helpers.resources import load_resource_state


class TestRewardContexts(unittest.TestCase):
    def test_combat_reward_context_preserves_reward_order(self):
        state = load_resource_state("/combat_reward/combat_reward_several_rewards.json")

        context = build_combat_reward_agent_context(state, "CombatRewardHandler")

        reward_summaries = context.extras["reward_summaries"]
        self.assertEqual(["GOLD", "RELIC", "POTION"], [row["reward_type"] for row in reward_summaries])
        self.assertEqual([0, 1, 2], [row["choice_index"] for row in reward_summaries])
        self.assertEqual(["gold", "relic", "potion"], [row["choice_token"] for row in reward_summaries])
        self.assertTrue(all("reward_summary_line" in row for row in reward_summaries))
        self.assertEqual(["gold", "relic", "potion"], context.extras["llm_choice_list"])
        self.assertTrue(context.extras["has_card_reward_row"])
        self.assertEqual(3, context.extras["non_card_reward_count"])
        self.assertEqual(3, context.extras["card_reward_choice_index"])
        self.assertEqual([3], context.extras["card_reward_choice_indexes"])
        self.assertEqual(["card"], context.extras["card_reward_choice_tokens"])

        all_reward_summaries = context.extras["all_reward_summaries"]
        self.assertEqual(["GOLD", "RELIC", "POTION", "CARD"], [row["reward_type"] for row in all_reward_summaries])

        gold_row = reward_summaries[0]
        self.assertEqual(35, gold_row["gold"])
        self.assertIn("type=GOLD", gold_row["reward_summary_line"])

        relic_row = reward_summaries[1]
        self.assertEqual("frozen egg", relic_row["relic_name"])
        self.assertIn("relic_id='Frozen Egg 2'", relic_row["reward_summary_line"])

        potion_row = reward_summaries[2]
        self.assertEqual("weak potion", potion_row["potion_name"])
        self.assertIn("can_discard=True", potion_row["reward_summary_line"])

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

    def test_grid_select_context_treats_confirm_up_as_zero_remaining_picks(self):
        state = load_resource_state("/other/upgrade_bash.json")
        payload = copy.deepcopy(state.json)
        payload["available_commands"] = ["confirm", "cancel", "wait", "state"]
        payload["game_state"]["screen_state"]["confirm_up"] = True
        payload["game_state"]["screen_state"]["selected_cards"] = []

        context = build_grid_select_agent_context(GameState(payload), "GridSelectHandler")

        self.assertTrue(context.extras["confirm_up"])
        self.assertEqual(0, context.extras["picks_remaining"])


if __name__ == "__main__":
    unittest.main()
