import unittest
from unittest.mock import MagicMock, patch

from rs.machine.character import Character
from rs.machine.game import Game
from rs.machine.state import GameState


class _FakeClient:
    pass


class TestGameLangmemFinalize(unittest.TestCase):
    def test_finalize_uses_last_game_state_when_final_json_omits_game_state(self):
        service = MagicMock()
        game = Game(_FakeClient(), Character.IRONCLAD)
        game.run_elites = []
        game.run_bosses = []
        game._langmem_last_game_state = {
            "floor": 3,
            "act": 2,
            "class": "IRONCLAD",
            "seed": 999888777,
            "screen_type": "GAME_OVER",
            "screen_state": {"score": 1234},
            "current_hp": 0,
            "max_hp": 70,
            "gold": 50,
        }
        game.last_state = GameState(
            {
                "in_game": False,
                "ready_for_command": True,
            }
        )

        with patch("rs.machine.game.get_langmem_service", return_value=service):
            game._Game__finalize_langmem_run()

        service.finalize_run.assert_called_once()
        call_context = service.finalize_run.call_args[0][0]
        self.assertEqual("RunFinalizer", call_context.handler_name)
        self.assertEqual("GAME_OVER", call_context.screen_type)
        self.assertEqual(3, call_context.game_state.get("floor"))
        self.assertEqual(2, call_context.game_state.get("act"))
        self.assertEqual("IRONCLAD", call_context.game_state.get("character_class"))
        self.assertEqual("ironclad:999888777", call_context.extras.get("run_id"))
        payload = service.finalize_run.call_args[0][1]
        self.assertEqual(1234, payload.get("score"))
        self.assertEqual(3, payload.get("floor"))
