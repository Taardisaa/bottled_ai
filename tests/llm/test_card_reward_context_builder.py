import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.machine.state import GameState


class TestCardRewardContextBuilder(unittest.TestCase):
    def test_build_card_reward_context_includes_upgrade_aware_deck_entries(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "card_reward" / "card_reward_take.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload)

        context = build_card_reward_agent_context(state, "CardRewardHandler")

        deck_entries = context.extras["deck_card_entries"]
        self.assertGreater(len(deck_entries), 0)
        self.assertIn(
            {"name": "bash+", "upgrade_times": 1, "count": 1},
            deck_entries,
        )
        self.assertEqual("IRONCLAD", context.game_state["character_class"])
        self.assertEqual("MonsterRoom", context.game_state["room_type"])
        self.assertEqual("Slime Boss", context.game_state["act_boss"])
        self.assertFalse(context.extras["potions_full"])

        deck_profile = context.extras["deck_profile"]
        self.assertEqual(10, deck_profile["total_cards"])
        self.assertEqual(6, deck_profile["type_counts"]["ATTACK"])
        self.assertEqual(4, deck_profile["type_counts"]["SKILL"])
        self.assertEqual(9, deck_profile["cost_buckets"]["one_cost"])
        self.assertEqual(1, deck_profile["cost_buckets"]["two_cost"])
        self.assertEqual(1, deck_profile["upgraded_cards"])

        choice_card_summaries = context.extras["choice_card_summaries"]
        self.assertEqual(3, len(choice_card_summaries))
        self.assertEqual("pommel strike", choice_card_summaries[0]["name"])
        self.assertEqual("ATTACK", choice_card_summaries[0]["type"])
        self.assertTrue(choice_card_summaries[0]["has_target"])

        self.assertEqual(
            {"bowl_available": False, "skip_available": True},
            context.extras["reward_screen_flags"],
        )


if __name__ == "__main__":
    unittest.main()
