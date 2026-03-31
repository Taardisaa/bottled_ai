import unittest
from unittest.mock import patch

from rs.llm.ai_player_graph import GraphExecutionResult
from rs.machine.character import Character
from rs.machine.game import Game


class _FakeClient:
    pass


class _FakeState:
    def __init__(self):
        self._running_calls = 0

    def is_game_running(self):
        self._running_calls += 1
        return self._running_calls <= 1

    def game_state(self):
        return {"room_type": "", "floor": 1, "act": 1, "class": "IRONCLAD"}

    def get_monsters(self):
        return []

    def screen_type(self):
        return "COMBAT_REWARD"

    def screen_state(self):
        return {"score": 0}

    def floor(self):
        return 1


class _FakeGraph:
    def __init__(self, result):
        self._result = result

    def is_enabled(self):
        return True

    def can_handle(self, _state):
        return True

    def execute(self, _state, runtime=None):
        return self._result


class _FakeLangMemService:
    def finalize_run(self, *_args, **_kwargs):
        return None


class TestGameAiGraphRuntimeHandoff(unittest.TestCase):
    def test_run_does_not_fail_when_graph_handles_with_no_commands(self):
        game = Game(_FakeClient(), Character.WATCHER)
        game.last_state = _FakeState()
        game.run_elites = []
        game.run_bosses = []
        game.last_elite = ""
        game.last_boss = ""

        handled_result = GraphExecutionResult(
            handled=True,
            commands=None,
            final_state=game.last_state,
        )
        graph = _FakeGraph(handled_result)

        with patch("rs.machine.game.await_controller", return_value=None), \
                patch("rs.machine.game.get_ai_player_graph", return_value=graph), \
                patch("rs.machine.game.get_langmem_service", return_value=_FakeLangMemService()), \
                patch.object(game, "_Game__send_command") as send_command_mock:
            game.run()

        send_command_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
