import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.providers.map_llm_provider import MapLlmProvider


class TestMapLlmProviderPrompt(unittest.TestCase):
    def test_prompt_includes_deterministic_path_summaries(self):
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose", "return"],
            choice_list=["x=0", "x=3", "x=4", "x=6"],
            game_state={
                "floor": 0,
                "act": 1,
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "room_type": "NeowRoom",
                "character_class": "IRONCLAD",
                "ascension_level": 0,
                "act_boss": "The Guardian",
                "current_position": "0_-1",
            },
            extras={
                "relic_names": ["Burning Blood"],
                "held_potion_names": [],
                "potions_full": False,
                "run_memory_summary": "IRONCLAD on Act 1 Floor 0 at HP 80/80 with 99 gold.",
                "recent_llm_decisions": "A1 F0 EventHandler -> choose 0 (0.91, free value)",
                "deck_profile": {"total_cards": 11, "type_counts": {"ATTACK": 6, "SKILL": 5}},
                "next_nodes": [{"symbol": "M", "x": 0, "y": 0}, {"symbol": "M", "x": 6, "y": 0}],
                "boss_available": False,
                "first_node_chosen": False,
                "deterministic_best_command": "choose 3",
                "choice_path_overviews": [
                    {"choice_label": "x=0", "choice_command": "choose 0", "shop_distance": None, "reward_survivability": 2.3},
                    {"choice_label": "x=6", "choice_command": "choose 3", "shop_distance": 1, "reward_survivability": 4.1},
                ],
                "sorted_path_summaries": [
                    {
                        "choice_label": "x=0",
                        "rooms": ["MONSTER", "QUESTION", "CAMPFIRE"],
                        "reward_survivability": 2.3,
                    },
                    {
                        "choice_label": "x=6",
                        "rooms": ["MONSTER", "SHOP", "ELITE"],
                        "reward_survivability": 4.1,
                    },
                ],
            },
        )

        provider = MapLlmProvider(model="gpt-5-mini")
        prompt = provider._build_prompt(context)

        self.assertIn("Deterministic best command: choose 3", prompt)
        self.assertIn("Choice path overviews (one per choice, worst to best):", prompt)
        self.assertIn("Sorted path summaries (worst to best):", prompt)
        self.assertIn('"choice_label": "x=6"', prompt)
        self.assertIn("Run memory summary: IRONCLAD on Act 1 Floor 0 at HP 80/80 with 99 gold.", prompt)
        self.assertIn("Recent LLM decisions: A1 F0 EventHandler -> choose 0 (0.91, free value)", prompt)
        self.assertIn('Current relics: ["Burning Blood"]', prompt)
        self.assertIn("Current position: 0_-1", prompt)


if __name__ == "__main__":
    unittest.main()
