import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.shop_purchase_context import build_shop_purchase_agent_context
from rs.machine.state import GameState
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class TestShopPurchaseContextBuilder(unittest.TestCase):
    def test_build_shop_context_includes_offer_summaries_and_deck_profile(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "shop" / "shop_buy_perfected_strike.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload, TheBotsMemoryBook.new_default())

        context = build_shop_purchase_agent_context(state, "ShopPurchaseHandler")

        self.assertEqual("IRONCLAD", context.game_state["character_class"])
        self.assertEqual("ShopRoom", context.game_state["room_type"])
        self.assertEqual("Slime Boss", context.game_state["act_boss"])
        self.assertEqual(75, context.extras["purge_cost"])
        self.assertTrue(context.extras["purge_available"])
        self.assertFalse(context.extras["potions_full"])
        self.assertEqual(["potion slot", "potion slot", "potion slot"], context.extras["held_potion_names"])

        deck_profile = context.extras["deck_profile"]
        self.assertEqual(11, deck_profile["total_cards"])
        self.assertEqual(6, deck_profile["type_counts"]["ATTACK"])
        self.assertEqual(5, deck_profile["type_counts"]["SKILL"])
        self.assertEqual(9, deck_profile["cost_buckets"]["one_cost"])
        self.assertEqual(2, deck_profile["cost_buckets"]["two_cost"])
        self.assertEqual(1, deck_profile["upgraded_cards"])

        offer_summaries = context.extras["offer_summaries"]
        self.assertEqual(7, len(offer_summaries["cards"]))
        self.assertEqual("perfected strike", offer_summaries["cards"][0]["name"])
        self.assertEqual(49, offer_summaries["cards"][0]["price"])
        self.assertEqual(3, len(offer_summaries["relics"]))
        self.assertEqual("Juzu Bracelet", offer_summaries["relics"][0]["name"])
        self.assertEqual(3, len(offer_summaries["potions"]))
        self.assertEqual("Gambler's Brew", offer_summaries["potions"][0]["name"])


if __name__ == "__main__":
    unittest.main()
