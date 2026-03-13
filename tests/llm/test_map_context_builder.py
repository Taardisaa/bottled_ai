import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.map_context import build_map_agent_context
from rs.machine.state import GameState
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class TestMapContextBuilder(unittest.TestCase):
    def test_build_map_context_includes_sorted_path_summaries(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "path" / "path_many_options.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload, TheBotsMemoryBook.new_default())

        context = build_map_agent_context(state, "CommonMapHandler")

        self.assertEqual("MAP", context.screen_type)
        self.assertEqual(0, context.game_state["floor"])
        self.assertEqual(1, context.game_state["act"])
        self.assertEqual("IRONCLAD", context.game_state["character_class"])
        self.assertEqual("0_-1", context.game_state["current_position"])
        self.assertEqual("choose 2", context.extras["deterministic_best_command"])

        path_summaries = context.extras["sorted_path_summaries"]
        self.assertGreater(len(path_summaries), 0)
        self.assertEqual("x=0", path_summaries[0]["choice_label"])
        self.assertEqual("choose 0", path_summaries[0]["choice_command"])
        self.assertIn("reward_survivability", path_summaries[0])
        self.assertIn("room_counts", path_summaries[0])
        self.assertIn("rooms", path_summaries[0])
        self.assertEqual(sorted(
            summary["reward_survivability"] for summary in path_summaries
        ), [summary["reward_survivability"] for summary in path_summaries])

        choice_overviews = context.extras["choice_path_overviews"]
        self.assertGreater(len(choice_overviews), 0)
        self.assertEqual(sorted(
            overview["reward_survivability"] for overview in choice_overviews
        ), [overview["reward_survivability"] for overview in choice_overviews])
        self.assertIn("shop_distance", choice_overviews[0])
        self.assertIn("elite_distance", choice_overviews[0])


if __name__ == "__main__":
    unittest.main()
