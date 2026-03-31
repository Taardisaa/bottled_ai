import unittest

from rs.llm.integration.astrolabe_transform_context import build_astrolabe_transform_agent_context
from rs.llm.integration.boss_reward_context import build_boss_reward_agent_context
from rs.llm.integration.combat_reward_context import build_combat_reward_agent_context
from rs.llm.providers.reward_llm_provider import (
    AstrolabeTransformLlmProvider,
    BossRewardLlmProvider,
    CombatRewardLlmProvider,
)
from test_helpers.resources import load_resource_state


class TestRewardLlmProviders(unittest.TestCase):
    def test_combat_reward_prompt_includes_duplicate_token_counts(self):
        state = load_resource_state("/combat_reward/combat_reward_relics_2_take_second.json")
        context = build_combat_reward_agent_context(state, "CombatRewardHandler")

        prompt = CombatRewardLlmProvider(model="gpt-5-mini")._build_prompt(context, {"recent_step_summaries": []})

        self.assertIn('"choice_token_counts"', prompt)
        self.assertIn('"relic": 2', prompt)
        self.assertIn("CombatReward structured context:", prompt)
        self.assertIn("Reward rows:", prompt)
        self.assertIn("Legal Commands and Exact Semantics", prompt)
        self.assertIn("Command availability:", prompt)
        self.assertIn("choice_token_counts=", prompt)
        self.assertIn("Never use `choose <token>` when `choice_token_counts[token] > 1`.", prompt)
        self.assertIn("if `choice_token_counts.potion=3`, do not return `choose potion`", prompt)

    def test_combat_reward_prompt_includes_card_delegation_note(self):
        state = load_resource_state("/combat_reward/combat_reward_several_rewards.json")
        context = build_combat_reward_agent_context(state, "CombatRewardHandler")

        prompt = CombatRewardLlmProvider(model="gpt-5-mini")._build_prompt(context, {"recent_step_summaries": []})
        self.assertIn("Deterministic handoff:", prompt)
        self.assertIn("has_card_reward_row=True", prompt)
        self.assertIn("non_card_reward_count=3", prompt)
        self.assertNotIn("type=CARD", prompt)

    def test_boss_reward_prompt_includes_metadata_mismatch_flag(self):
        state = load_resource_state("/relics/boss_reward_first_is_best.json")
        context = build_boss_reward_agent_context(state, "BossRewardHandler")

        prompt = BossRewardLlmProvider(model="gpt-5-mini")._build_prompt(context, {"recent_step_summaries": []})

        self.assertIn('"boss_relic_options"', prompt)
        self.assertIn('"choice_metadata_mismatch"', prompt)
        self.assertIn("Treat `choice_list` as the source of truth", prompt)

    def test_astrolabe_prompt_includes_remaining_picks(self):
        state = load_resource_state("/relics/boss_reward_astrolabe.json")
        context = build_astrolabe_transform_agent_context(state, "AstrolabeTransformHandler")

        prompt = AstrolabeTransformLlmProvider(model="gpt-5-mini")._build_prompt(context, {"recent_step_summaries": []})

        self.assertIn('"picks_remaining": 2', prompt)
        self.assertIn('"selected_cards"', prompt)
        self.assertIn("Do not use any fixed card-removal priority list", prompt)


if __name__ == "__main__":
    unittest.main()
