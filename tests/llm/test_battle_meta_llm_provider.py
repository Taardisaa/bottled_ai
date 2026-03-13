import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.battle_meta_llm_provider import BattleMetaLlmProvider


class TestBattleMetaLlmProviderPrompt(unittest.TestCase):
    def test_prompt_includes_profile_choices_and_battle_summary(self):
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT_REWARD",
            available_commands=["play", "end"],
            choice_list=[],
            game_state={
                "floor": 33,
                "act": 3,
                "room_type": "MonsterRoomBoss",
                "character_class": "WATCHER",
                "ascension_level": 20,
                "current_hp": 54,
                "max_hp": 72,
                "gold": 123,
                "turn": 5,
                "player_block": 18,
                "player_energy": 2,
            },
            extras={
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
                "monster_summaries": [{"name": "Time Eater", "intent": "ATTACK_BUFF"}],
                "player_power_summaries": [{"id": "MentalFortress", "amount": 2}],
                "relic_names": ["Violet Lotus"],
                "held_potion_names": ["Dexterity Potion"],
                "deck_profile": {"total_cards": 23, "type_counts": {"ATTACK": 7, "POWER": 4}},
            },
        )

        provider = BattleMetaLlmProvider(model="gpt-5-mini")
        prompt = provider._build_prompt(context)

        self.assertIn("Deterministic profile: big_fight", prompt)
        self.assertIn('Available profiles: ["big_fight", "general"]', prompt)
        self.assertIn('"name": "Time Eater"', prompt)
        self.assertIn('Relics: ["Violet Lotus"]', prompt)
        self.assertIn('Held potions: ["Dexterity Potion"]', prompt)
        self.assertIn("Player energy: 2", prompt)


if __name__ == "__main__":
    unittest.main()
