import unittest
import json
from pathlib import Path

from definitions import ROOT_DIR
from rs.llm.integration.event_context import build_event_agent_context
from rs.machine.state import GameState
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class TestEventContextBuilder(unittest.TestCase):
    def test_build_event_context_extracts_expected_fields(self):
        state_path = Path(ROOT_DIR) / "tests" / "res" / "event" / "event_cleric_heal.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = GameState(payload, TheBotsMemoryBook.new_default())

        context = build_event_agent_context(state, "CommonEventHandler")

        self.assertEqual("CommonEventHandler", context.handler_name)
        self.assertEqual("EVENT", context.screen_type)
        self.assertEqual("The Cleric", context.game_state["event_name"])
        self.assertIn("choose", context.available_commands)
        self.assertIn("heal", [c.lower() for c in context.choice_list])


if __name__ == "__main__":
    unittest.main()
