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
                "langmem_status": "embeddings_unavailable:missing model",
                "current_priorities": ["avoid elites"],
                "risk_flags": ["low_max_hp"],
                "deck_direction": "frontload",
                "run_hypotheses": ["maintain CommonMapHandler consistency"],
                "deck_profile": {"total_cards": 11, "type_counts": {"ATTACK": 6, "SKILL": 5}},
                "boss_available": False,
                "first_node_chosen": False,
                "deterministic_best_command": "choose 3",
                "choice_branch_summaries": [
                    {
                        "choice_index": 0,
                        "choice_label": "x=0",
                        "choice_command": "choose 0",
                        "path_count": 3,
                        "monster_count_range": {"min": 3, "max": 4},
                        "event_count_range": {"min": 1, "max": 2},
                        "shop_count_range": {"min": 0, "max": 1},
                        "campfire_count_range": {"min": 1, "max": 1},
                        "elite_count_range": {"min": 0, "max": 1},
                        "treasure_count_range": {"min": 1, "max": 1},
                        "first_shop_distance_range": {"min": 2, "max": 5},
                        "first_campfire_distance_range": {"min": 4, "max": 4},
                        "first_elite_distance_range": {"min": 5, "max": 6},
                        "branch_shape_summary": "3 descendant paths; shared prefix MONSTER > QUESTION",
                    },
                    {
                        "choice_index": 3,
                        "choice_label": "x=6",
                        "choice_command": "choose 3",
                        "path_count": 2,
                        "monster_count_range": {"min": 3, "max": 3},
                        "event_count_range": {"min": 1, "max": 2},
                        "shop_count_range": {"min": 1, "max": 1},
                        "campfire_count_range": {"min": 1, "max": 1},
                        "elite_count_range": {"min": 1, "max": 2},
                        "treasure_count_range": {"min": 1, "max": 1},
                        "first_shop_distance_range": {"min": 1, "max": 1},
                        "first_campfire_distance_range": {"min": 4, "max": 4},
                        "first_elite_distance_range": {"min": 4, "max": 5},
                        "branch_shape_summary": "2 descendant paths; shared prefix MONSTER > QUESTION",
                    },
                ],
                "choice_representative_paths": [
                    {
                        "choice_index": 0,
                        "choice_label": "x=0",
                        "choice_command": "choose 0",
                        "representative_paths": [
                            {
                                "choice_command": "choose 0",
                                "rooms": ["MONSTER", "QUESTION", "QUESTION", "MONSTER", "QUESTION", "CAMPFIRE"],
                                "room_counts": {"MONSTER": 2, "QUESTION": 3, "ELITE": 0, "CAMPFIRE": 1, "TREASURE": 0, "SHOP": 0, "BOSS": 1},
                                "path_length": 6,
                                "first_shop_distance": None,
                                "first_campfire_distance": 5,
                                "first_elite_distance": None,
                            }
                        ],
                    },
                    {
                        "choice_index": 3,
                        "choice_label": "x=6",
                        "choice_command": "choose 3",
                        "representative_paths": [
                            {
                                "choice_command": "choose 3",
                                "rooms": ["MONSTER", "QUESTION", "SHOP", "QUESTION", "SHOP", "ELITE"],
                                "room_counts": {"MONSTER": 1, "QUESTION": 2, "ELITE": 1, "CAMPFIRE": 0, "TREASURE": 0, "SHOP": 2, "BOSS": 1},
                                "path_length": 6,
                                "first_shop_distance": 2,
                                "first_campfire_distance": None,
                                "first_elite_distance": 5,
                            }
                        ],
                    },
                ],
            },
        )

        provider = MapLlmProvider(model="gpt-5-mini")
        prompt = provider._build_prompt(context)

        self.assertIn("answer in short plain text using these fields", prompt)
        self.assertIn('choose <index>', prompt)
        self.assertIn('Available protocol commands:', prompt)
        self.assertIn('- 0 | route="x=0"', prompt)
        self.assertIn("Deterministic best command: choose 3", prompt)
        self.assertIn("Choice branch summaries:", prompt)
        self.assertIn("Representative descendant paths:", prompt)
        self.assertIn("paths=2", prompt)
        self.assertIn("shape=2 descendant paths; shared prefix MONSTER > QUESTION", prompt)
        self.assertIn("rooms=MONSTER > QUESTION > SHOP > QUESTION > SHOP > ELITE", prompt)
        self.assertIn("Run memory summary: IRONCLAD on Act 1 Floor 0 at HP 80/80 with 99 gold.", prompt)
        self.assertIn("Recent LLM decisions: A1 F0 EventHandler -> choose 0 (0.91, free value)", prompt)
        self.assertIn("LangMem status: unavailable", prompt)
        self.assertNotIn("missing model", prompt)
        self.assertIn('Current relics: ["Burning Blood"]', prompt)
        self.assertIn("Current position: 0_-1", prompt)
        self.assertNotIn("Return ONLY a JSON object", prompt)
        self.assertNotIn("Sorted path summaries (worst to best):", prompt)


if __name__ == "__main__":
    unittest.main()
