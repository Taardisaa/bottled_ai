import json
import unittest
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.machine.state import GameState
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class TestCardRewardContextBuilder(unittest.TestCase):
    def test_build_card_reward_context_includes_upgrade_aware_deck_entries(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "card_reward" / "card_reward_take.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload, TheBotsMemoryBook.new_default())

        context = build_card_reward_agent_context(state, "CardRewardHandler")

        deck_entries = context.extras["deck_card_entries"]
        self.assertGreater(len(deck_entries), 0)
        self.assertIn(
            {"name": "bash+", "upgrade_times": 1, "count": 1},
            deck_entries,
        )


if __name__ == "__main__":
    unittest.main()
