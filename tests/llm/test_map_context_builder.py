import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.map_context import build_map_agent_context
from rs.machine.state import GameState


class TestMapContextBuilder(unittest.TestCase):
    def test_build_map_context_includes_branch_summaries_and_representative_paths(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "path" / "path_many_options.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload)

        context = build_map_agent_context(state, "CommonMapHandler")

        self.assertEqual("MAP", context.screen_type)
        self.assertEqual(0, context.game_state["floor"])
        self.assertEqual(1, context.game_state["act"])
        self.assertEqual("IRONCLAD", context.game_state["character_class"])
        self.assertEqual("choose 2", context.extras["deterministic_best_command"])
        self.assertEqual(64, context.extras["map_graph_metadata"]["node_count"])
        self.assertGreater(context.extras["map_graph_metadata"]["edge_count"], 0)
        self.assertEqual(11, context.extras["deck_profile"]["total_cards"])
        self.assertIn("type_counts", context.extras["deck_profile"])
        self.assertIn("upgraded_cards", context.extras["deck_profile"])
        self.assertNotIn("cost_buckets", context.extras["deck_profile"])

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

        branch_summaries = context.extras["choice_branch_summaries"]
        self.assertEqual(4, len(branch_summaries))
        self.assertEqual("x=0", branch_summaries[0]["choice_label"])
        self.assertEqual("choose 0", branch_summaries[0]["choice_command"])
        self.assertEqual(8, branch_summaries[0]["path_count"])
        self.assertEqual({"min": 5, "max": 8}, branch_summaries[0]["monster_count_range"])
        self.assertEqual({"min": 3, "max": 5}, branch_summaries[0]["event_count_range"])
        self.assertEqual({"min": 0, "max": 0}, branch_summaries[0]["shop_count_range"])
        self.assertEqual({"min": 3, "max": 4}, branch_summaries[0]["campfire_count_range"])
        self.assertEqual({"min": 0, "max": 1}, branch_summaries[0]["elite_count_range"])
        self.assertIsNone(branch_summaries[0]["first_shop_distance_range"])
        self.assertEqual({"min": 5, "max": 5}, branch_summaries[0]["first_campfire_distance_range"])
        self.assertEqual({"min": 10, "max": 10}, branch_summaries[0]["first_elite_distance_range"])
        self.assertIn("descendant paths", branch_summaries[0]["branch_shape_summary"])
        self.assertIn("shared prefix MONSTER > MONSTER > QUESTION", branch_summaries[0]["branch_shape_summary"])

        representative_groups = context.extras["choice_representative_paths"]
        self.assertEqual(4, len(representative_groups))
        self.assertEqual("choose 0", representative_groups[0]["choice_command"])
        self.assertLessEqual(len(representative_groups[0]["representative_paths"]), 1)
        first_path = representative_groups[0]["representative_paths"][0]
        self.assertIn("rooms", first_path)
        self.assertIn("path_length", first_path)
        self.assertIn("first_shop_distance", first_path)
        self.assertIn("first_campfire_distance", first_path)
        self.assertIn("first_elite_distance", first_path)
        self.assertNotIn("room_counts", first_path)


if __name__ == "__main__":
    unittest.main()
