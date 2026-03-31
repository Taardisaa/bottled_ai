import unittest

from rs.machine.state import GameState


class TestGameStateChoiceList(unittest.TestCase):
    def test_get_choice_list_returns_empty_when_missing(self):
        state = GameState({
            "in_game": True,
            "available_commands": ["confirm", "cancel"],
            "game_state": {
                "screen_type": "GRID",
                "screen_state": {"cards": []},
                "deck": [],
                "potions": [],
                "relics": [],
            },
        })

        self.assertEqual([], state.get_choice_list())

    def test_get_choice_list_returns_strings_for_list_values(self):
        state = GameState({
            "in_game": True,
            "available_commands": ["choose"],
            "game_state": {
                "choice_list": ["a", 1, True],
                "screen_type": "EVENT",
                "screen_state": {},
                "deck": [],
                "potions": [],
                "relics": [],
            },
        })

        self.assertEqual(["a", "1", "True"], state.get_choice_list())


if __name__ == "__main__":
    unittest.main()
