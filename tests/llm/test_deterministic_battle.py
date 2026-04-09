import unittest

from rs.llm.battle_subagent import BattleSubagentConfig
from rs.llm.deterministic_battle import DeterministicBattleSubagent
from rs.common.handlers.common_battle_handler import CommonBattleHandler, BattleHandlerConfig


class TestCommonBattleHandlerRestored(unittest.TestCase):
    def test_handler_instantiates(self):
        handler = CommonBattleHandler()
        self.assertIsNotNone(handler)
        self.assertEqual(11_000, handler.max_path_count)

    def test_comparator_profiles(self):
        handler = CommonBattleHandler()
        config = handler.config
        self.assertIsNotNone(config.general_comparator)
        self.assertIsNotNone(config.big_fight_comparator)
        self.assertIsNotNone(config.gremlin_nob_comparator)


class TestDeterministicSubagentConstruction(unittest.TestCase):
    def test_subagent_instantiates(self):
        subagent = DeterministicBattleSubagent(
            config=BattleSubagentConfig(),
            langmem_service=None,
        )
        self.assertIsNotNone(subagent._battle_handler)
        self.assertEqual(11_000, subagent._battle_handler.max_path_count)
        self.assertEqual(2, len(subagent._potion_handlers))


class TestSelectionScreen(unittest.TestCase):
    def test_choose_when_cards_available(self):
        class FakeState:
            def game_state(self):
                return {"screen_state": {"cards": [{"name": "Strike"}], "selected_count": 0, "max_select_count": 1}}
            def screen_type(self):
                return "HAND_SELECT"
        cmds = DeterministicBattleSubagent._handle_selection_screen(FakeState())
        self.assertEqual(["choose 0"], cmds)

    def test_confirm_when_selection_done(self):
        class FakeState:
            def game_state(self):
                return {"screen_state": {"cards": [{"name": "Strike"}], "selected_count": 1, "max_select_count": 1}}
            def screen_type(self):
                return "HAND_SELECT"
        cmds = DeterministicBattleSubagent._handle_selection_screen(FakeState())
        self.assertEqual(["confirm", "wait 30"], cmds)


class TestStateSignature(unittest.TestCase):
    def test_different_monster_hp_gives_different_signature(self):
        class FakeState:
            def __init__(self, monster_hp):
                self._hp = monster_hp
            def game_state(self):
                return {
                    "available_commands": ["play", "end"],
                    "current_action": "",
                    "turn": 1,
                    "current_hp": 50,
                    "player": {"block": 0, "energy": 3},
                    "combat_state": {"monsters": [{"current_hp": self._hp, "is_gone": False}]},
                }
            def screen_type(self):
                return "NONE"

        sig_a = DeterministicBattleSubagent._state_signature(FakeState(40))
        sig_b = DeterministicBattleSubagent._state_signature(FakeState(30))
        self.assertNotEqual(sig_a, sig_b)


if __name__ == "__main__":
    unittest.main()
