import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.state_summary_cache import clear_cached_run_summaries, get_cached_run_summary
from rs.machine.state import GameState
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class TestStateSummaryCache(unittest.TestCase):
    def setUp(self):
        clear_cached_run_summaries()

    def test_returns_deep_copied_cached_summary(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "card_reward" / "card_reward_take.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload, TheBotsMemoryBook.new_default())

        first_summary = get_cached_run_summary(state)
        first_summary["relic_names"].append("fake relic")
        first_summary["deck_profile"]["type_counts"]["ATTACK"] = -1

        second_summary = get_cached_run_summary(state)

        self.assertNotIn("fake relic", second_summary["relic_names"])
        self.assertEqual(6, second_summary["deck_profile"]["type_counts"]["ATTACK"])


if __name__ == "__main__":
    unittest.main()
