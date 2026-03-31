import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.validator import validate_command, validate_command_batch
from test_helpers.resources import load_resource_state


def _battle_context():
    state = load_resource_state("battles/general/another_simple.json")
    return AgentContext(
        handler_name="BattleHandler",
        screen_type=state.screen_type(),
        available_commands=["play", "potion", "choose", "end", "wait"],
        choice_list=list(state.game_state().get("choice_list", [])),
        extras={
            "game_state_ref": state,
            "selection_card_count": 0,
        },
    )


class TestBattleValidator(unittest.TestCase):
    def test_validate_command_battle_accepts_valid_targeted_play(self):
        context = _battle_context()
        result = validate_command(context, "play 2 0", mode="battle")
        self.assertTrue(result.is_valid)

    def test_validate_command_battle_rejects_target_on_untargeted_card(self):
        context = _battle_context()
        result = validate_command(context, "play 1 0", mode="battle")
        self.assertFalse(result.is_valid)
        self.assertEqual("bad_syntax", result.code)

    def test_validate_command_battle_rejects_invalid_target_index(self):
        context = _battle_context()
        result = validate_command(context, "play 2 9", mode="battle")
        self.assertFalse(result.is_valid)
        self.assertEqual("index_out_of_range", result.code)

    def test_validate_command_battle_rejects_empty_potion_slot(self):
        context = _battle_context()
        result = validate_command(context, "potion use 1", mode="battle")
        self.assertFalse(result.is_valid)
        self.assertEqual("command_not_available", result.code)

    def test_validate_command_battle_accepts_valid_potion_command(self):
        context = _battle_context()
        result = validate_command(context, "potion use 0", mode="battle")
        self.assertTrue(result.is_valid)

    def test_validate_command_batch_returns_per_command_errors(self):
        context = _battle_context()
        result = validate_command_batch(
            context,
            ["play 2 0", "play 1 0", "potion use 1"],
            mode="battle",
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(2, len(result["errors"]))
        error_codes = {error["code"] for error in result["errors"]}
        self.assertIn("bad_syntax", error_codes)
        self.assertIn("command_not_available", error_codes)


if __name__ == "__main__":
    unittest.main()
