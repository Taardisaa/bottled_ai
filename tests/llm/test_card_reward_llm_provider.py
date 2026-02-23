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
                "current_hp": 50,
                "max_hp": 80,
            },
            extras={
                "deck_size": 15,
                "relic_names": ["Vajra"],
                "deck_card_name_counts": {"strike": 4, "defend": 4, "pommel strike": 1},
                "deck_card_entries": [
                    {"name": "pommel strike+1", "upgrade_times": 1, "count": 1},
                    {"name": "strike", "upgrade_times": 0, "count": 4},
                ],
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
