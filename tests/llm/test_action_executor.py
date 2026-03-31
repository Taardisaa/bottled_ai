import unittest

from rs.llm.action_policy_registry import ActionPolicyRegistry
from rs.llm.agents.base_agent import AgentContext


class TestActionPolicyRegistry(unittest.TestCase):
    def test_boss_skip_maps_to_skip_proceed(self):
        registry = ActionPolicyRegistry()
        context = AgentContext(
            handler_name="BossRewardHandler",
            screen_type="BOSS_REWARD",
            available_commands=["skip", "proceed"],
            choice_list=["a", "b", "c"],
            game_state={},
            extras={},
        )
        resolution = registry.resolve(context, "skip")
        self.assertTrue(resolution.validation.is_valid)
        self.assertEqual(["skip", "proceed"], resolution.commands)

    def test_duplicate_choose_token_is_rejected(self):
        registry = ActionPolicyRegistry()
        context = AgentContext(
            handler_name="CombatRewardHandler",
            screen_type="COMBAT_REWARD",
            available_commands=["choose"],
            choice_list=["relic", "relic"],
            game_state={},
            extras={},
        )
        resolution = registry.resolve(context, "choose relic")
        self.assertIsNone(resolution.commands)
        self.assertEqual("ambiguous_choice_token", resolution.validation.code)


if __name__ == "__main__":
    unittest.main()
