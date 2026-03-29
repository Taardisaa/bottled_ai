import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.shop_purchase_llm_provider import ShopPurchaseLlmProvider


class TestShopPurchaseLlmProviderPrompt(unittest.TestCase):
    def test_prompt_includes_shop_offer_and_deck_context(self):
        context = AgentContext(
            handler_name="ShopPurchaseHandler",
            screen_type="SHOP_SCREEN",
            available_commands=["choose", "leave"],
            choice_list=["purge", "perfected strike+", "vajra", "strength potion"],
            game_state={
                "floor": 10,
                "act": 1,
                "gold": 223,
                "current_hp": 57,
                "max_hp": 85,
                "room_type": "ShopRoom",
                "character_class": "IRONCLAD",
                "ascension_level": 3,
                "act_boss": "The Guardian",
            },
            extras={
                "has_removable_curse": False,
                "deck_size": 12,
                "deck_profile": {
                    "total_cards": 12,
                    "type_counts": {"ATTACK": 7, "SKILL": 5},
                    "cost_buckets": {"one_cost": 11, "two_cost": 1},
                    "upgraded_cards": 0,
                },
                "run_memory_summary": "IRONCLAD on Act 1 Floor 10 at HP 57/85 with 223 gold.",
                "recent_llm_decisions": "A1 F9 CardRewardHandler -> choose 1 (0.76, take scaling)",
                "langmem_status": "embeddings_unavailable:request failed",
                "current_priorities": ["remove strikes", "preserve gold"],
                "risk_flags": [],
                "deck_direction": "strength",
                "run_hypotheses": ["maintain ShopPurchaseHandler consistency"],
                "relic_names": ["Burning Blood", "Mummified Hand"],
                "held_potion_names": ["gambler's brew", "elixir"],
                "potions_full": False,
                "purge_cost": 75,
                "purge_available": True,
                "offer_summaries": {
                    "cards": [
                        {"name": "perfected strike+", "type": "ATTACK", "rarity": "COMMON", "cost": 2, "price": 52},
                    ],
                    "relics": [
                        {"name": "Vajra", "price": 154},
                    ],
                    "potions": [
                        {"name": "Strength Potion", "price": 52, "requires_target": False},
                    ],
                },
            },
        )

        provider = ShopPurchaseLlmProvider(model="gpt-5-mini")
        prompt = provider._build_prompt(context)

        self.assertIn("answer in short plain text using these fields", prompt)
        self.assertIn('choose <index>', prompt)
        self.assertIn('Available protocol commands:', prompt)
        self.assertIn('- 0 | option="purge"', prompt)
        self.assertIn("Class: IRONCLAD, Ascension: 3, Act boss: The Guardian", prompt)
        self.assertIn("Run memory summary: IRONCLAD on Act 1 Floor 10 at HP 57/85 with 223 gold.", prompt)
        self.assertIn("Recent LLM decisions: A1 F9 CardRewardHandler -> choose 1 (0.76, take scaling)", prompt)
        self.assertIn("LangMem status: unavailable", prompt)
        self.assertNotIn("request failed", prompt)
        self.assertIn("Deck profile:", prompt)
        self.assertIn('"ATTACK": 7', prompt)
        self.assertIn('Current relics: ["Burning Blood", "Mummified Hand"]', prompt)
        self.assertIn('Held potions: ["gambler\'s brew", "elixir"]', prompt)
        self.assertIn("Purge available: True", prompt)
        self.assertIn("Purge cost: 75", prompt)
        self.assertIn("Shop card offers:", prompt)
        self.assertIn('"name": "perfected strike+"', prompt)
        self.assertIn("Shop relic offers:", prompt)
        self.assertIn('"name": "Vajra"', prompt)
        self.assertIn("Shop potion offers:", prompt)
        self.assertIn('"name": "Strength Potion"', prompt)
        self.assertNotIn("Return ONLY a JSON object", prompt)


if __name__ == "__main__":
    unittest.main()
