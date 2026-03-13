import unittest
from unittest.mock import patch

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProvider


class TestCardRewardLlmProviderPrompt(unittest.TestCase):
    def test_prompt_includes_stsdb_details_when_available(self):
        context = AgentContext(
            handler_name="CardRewardHandler",
            screen_type="CARD_REWARD",
            available_commands=["choose", "skip"],
            choice_list=["Pommel Strike", "Cleave", "Bowl"],
            game_state={
                "floor": 10,
                "act": 1,
                "room_phase": "COMPLETE",
                "room_type": "MonsterRoom",
                "current_hp": 50,
                "max_hp": 80,
                "character_class": "IRONCLAD",
                "ascension_level": 5,
                "act_boss": "Hexaghost",
            },
            extras={
                "deck_size": 15,
                "relic_names": ["Vajra"],
                "held_potion_names": ["strength potion"],
                "potions_full": False,
                "deck_card_name_counts": {"strike": 4, "defend": 4, "pommel strike": 1},
                "deck_card_entries": [
                    {"name": "pommel strike+1", "upgrade_times": 1, "count": 1},
                    {"name": "strike", "upgrade_times": 0, "count": 4},
                ],
                "deck_profile": {
                    "total_cards": 15,
                    "type_counts": {"ATTACK": 7, "SKILL": 7, "POWER": 1},
                    "cost_buckets": {"one_cost": 10, "two_cost": 3, "zero_cost": 2},
                    "upgraded_cards": 2,
                },
                "run_memory_summary": "IRONCLAD on Act 1 Floor 10 at HP 50/80 with 99 gold.",
                "recent_llm_decisions": "A1 F9 EventHandler -> choose 0 (0.88, safe heal)",
                "choice_card_summaries": [
                    {"index": 0, "name": "pommel strike", "type": "ATTACK", "rarity": "COMMON", "cost": 1},
                    {"index": 1, "name": "cleave", "type": "ATTACK", "rarity": "COMMON", "cost": 1},
                ],
                "reward_screen_flags": {"bowl_available": True, "skip_available": True},
            },
        )

        with patch(
            "rs.llm.providers.card_reward_llm_provider.query_card",
            side_effect=lambda name, upgrade_times=0: {
                "name": name,
                "type": "ATTACK",
                "cost": 1,
                "description": "Deal damage",
            },
        ):
            provider = CardRewardLlmProvider(model="gpt-5-mini")
            prompt = provider._build_prompt(context)

        self.assertIn("Card DB status: available", prompt)
        self.assertIn("Choice card details (stsdb):", prompt)
        self.assertIn("Class: IRONCLAD, Ascension: 5, Act boss: Hexaghost", prompt)
        self.assertIn("Run memory summary: IRONCLAD on Act 1 Floor 10 at HP 50/80 with 99 gold.", prompt)
        self.assertIn("Recent LLM decisions: A1 F9 EventHandler -> choose 0 (0.88, safe heal)", prompt)
        self.assertIn("Deck profile:", prompt)
        self.assertIn('"upgraded_cards": 2', prompt)
        self.assertIn("Choice card summaries:", prompt)
        self.assertIn('"name": "pommel strike"', prompt)
        self.assertIn('Reward screen flags: {"bowl_available": true, "skip_available": true}', prompt)
        self.assertIn('"name": "pommel strike"', prompt)
        self.assertIn("Deck card counts:", prompt)

    def test_query_card_receives_upgrade_times(self):
        calls = []

        def fake_query_card(name, upgrade_times=0):
            calls.append((name, upgrade_times))
            return {
                "name": name,
                "type": "ATTACK",
                "cost": 1,
            }

        context = AgentContext(
            handler_name="CardRewardHandler",
            screen_type="CARD_REWARD",
            available_commands=["choose", "skip"],
            choice_list=["Bash+", "Searing Blow+4", "Cleave"],
            game_state={
                "floor": 10,
                "act": 1,
                "room_phase": "COMPLETE",
                "current_hp": 50,
                "max_hp": 80,
            },
            extras={
                "deck_size": 15,
                "relic_names": [],
                "deck_card_name_counts": {},
                "deck_card_entries": [
                    {"name": "searing blow+7", "upgrade_times": 7, "count": 1},
                ],
                "run_memory_summary": "IRONCLAD on Act 1 Floor 10 at HP 50/80 with 99 gold.",
                "recent_llm_decisions": "none",
            },
        )

        with patch("rs.llm.providers.card_reward_llm_provider.query_card", side_effect=fake_query_card):
            provider = CardRewardLlmProvider(model="gpt-5-mini")
            provider._build_prompt(context)

        self.assertIn(("searing blow", 4), calls)
        self.assertIn(("searing blow", 7), calls)
        self.assertIn(("cleave", 0), calls)

if __name__ == "__main__":
    unittest.main()
