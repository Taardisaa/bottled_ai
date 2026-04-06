import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.battle_command_normalizer import normalize_battle_command


def _ctx(
    hand_cards=None,
    monster_summaries=None,
    potion_summaries=None,
) -> AgentContext:
    return AgentContext(
        handler_name="BattleHandler",
        screen_type="NONE",
        available_commands=["play", "end", "potion", "wait"],
        choice_list=[],
        extras={
            "hand_cards": hand_cards or [],
            "monster_summaries": monster_summaries or [],
            "potion_summaries": potion_summaries or [],
        },
    )


_HAND = [
    {"hand_index": 1, "name": "Strike", "cost": 1, "has_target": True, "is_playable": True},
    {"hand_index": 2, "name": "Defend", "cost": 1, "is_playable": True},
    {"hand_index": 3, "name": "Eruption", "cost": 2, "has_target": True, "is_playable": True},
    {"hand_index": 4, "name": "Vigilance", "cost": 2, "is_playable": True},
]

_MONSTERS = [
    {"name": "Jaw Worm", "target_index": 0, "current_hp": 40, "is_gone": False},
]

_TWO_MONSTERS = [
    {"name": "Acid Slime", "target_index": 0, "current_hp": 30, "is_gone": False},
    {"name": "Spike Slime", "target_index": 1, "current_hp": 25, "is_gone": False},
]

_DUPLICATE_MONSTERS = [
    {"name": "Acid Slime", "target_index": 0, "current_hp": 30, "is_gone": False},
    {"name": "Acid Slime", "target_index": 1, "current_hp": 20, "is_gone": False},
]

_POTIONS = [
    {"slot_index": 0, "name": "Fire Potion"},
    {"slot_index": 1, "name": "Swift Potion"},
]


class TestSimpleCommandNormalization(unittest.TestCase):

    def test_end_turn(self):
        ctx = _ctx()
        self.assertEqual("end", normalize_battle_command("end turn", ctx))

    def test_end_underscore(self):
        ctx = _ctx()
        self.assertEqual("end", normalize_battle_command("end_turn", ctx))

    def test_confirm_action(self):
        ctx = _ctx()
        self.assertEqual("confirm", normalize_battle_command("confirm action", ctx))

    def test_wait_strips_noise(self):
        ctx = _ctx()
        self.assertEqual("wait 30", normalize_battle_command("wait 30 seconds", ctx))

    def test_plain_end(self):
        ctx = _ctx()
        self.assertEqual("end", normalize_battle_command("end", ctx))

    def test_plain_wait(self):
        ctx = _ctx()
        self.assertEqual("wait 30", normalize_battle_command("wait 30", ctx))


class TestTargetedUntargetedFix(unittest.TestCase):

    def test_strip_target_from_untargeted_card(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        # Defend (index 2) is untargeted, LLM adds target 0
        self.assertEqual("play 2", normalize_battle_command("play 2 0", ctx))

    def test_strip_target_from_vigilance(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        # Vigilance (index 4) is untargeted
        self.assertEqual("play 4", normalize_battle_command("play 4 0", ctx))

    def test_auto_add_target_single_monster(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        # Strike (index 1) is targeted, only 1 monster
        self.assertEqual("play 1 0", normalize_battle_command("play 1", ctx))

    def test_no_auto_target_multiple_monsters(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_TWO_MONSTERS)
        # Strike (index 1) is targeted, 2 monsters — can't auto-target
        self.assertEqual("play 1", normalize_battle_command("play 1", ctx))

    def test_keep_valid_targeted_play(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_TWO_MONSTERS)
        # Already correct
        self.assertEqual("play 1 0", normalize_battle_command("play 1 0", ctx))


class TestMonsterNameResolution(unittest.TestCase):

    def test_resolve_monster_name(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command('play 1 "Jaw Worm"', ctx))

    def test_resolve_monster_name_no_quotes(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_TWO_MONSTERS)
        self.assertEqual("play 1 1", normalize_battle_command("play 1 Spike Slime", ctx))

    def test_ambiguous_monster_name_unchanged(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_DUPLICATE_MONSTERS)
        # Two "Acid Slime" — ambiguous, leave unchanged
        self.assertEqual("play 1 Acid Slime", normalize_battle_command("play 1 Acid Slime", ctx))

    def test_partial_monster_name(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command("play 1 jaw", ctx))


class TestCardNameResolution(unittest.TestCase):

    def test_resolve_card_name_with_target(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command("play Strike 0", ctx))

    def test_resolve_card_name_untargeted(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 2", normalize_battle_command("play Defend", ctx))

    def test_resolve_card_name_with_monster_name(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 3 0", normalize_battle_command("play Eruption Jaw Worm", ctx))

    def test_ambiguous_card_name(self):
        hand_with_dupes = _HAND + [
            {"hand_index": 5, "name": "Strike", "cost": 1, "has_target": True, "is_playable": True},
        ]
        ctx = _ctx(hand_cards=hand_with_dupes, monster_summaries=_MONSTERS)
        # Two Strikes — ambiguous
        self.assertEqual("play Strike 0", normalize_battle_command("play Strike 0", ctx))


class TestPotionNameResolution(unittest.TestCase):

    def test_resolve_potion_name(self):
        ctx = _ctx(potion_summaries=_POTIONS, monster_summaries=_MONSTERS)
        self.assertEqual("potion use 0", normalize_battle_command("potion use Fire Potion", ctx))

    def test_resolve_potion_name_with_target(self):
        ctx = _ctx(potion_summaries=_POTIONS, monster_summaries=_MONSTERS)
        self.assertEqual("potion use 0 0", normalize_battle_command("potion use Fire Potion 0", ctx))

    def test_resolve_potion_discard(self):
        ctx = _ctx(potion_summaries=_POTIONS)
        self.assertEqual("potion discard 1", normalize_battle_command("potion discard Swift Potion", ctx))

    def test_potion_integer_passthrough(self):
        ctx = _ctx(potion_summaries=_POTIONS)
        self.assertEqual("potion use 0", normalize_battle_command("potion use 0", ctx))

    def test_potion_with_monster_name_target(self):
        ctx = _ctx(potion_summaries=_POTIONS, monster_summaries=_MONSTERS)
        self.assertEqual(
            "potion use 0 0",
            normalize_battle_command('potion use Fire Potion Jaw Worm', ctx),
        )


class TestQuoteStripping(unittest.TestCase):

    def test_strip_double_quotes(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command('play 1 "Jaw Worm"', ctx))

    def test_strip_single_quotes(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command("play 1 'Jaw Worm'", ctx))


class TestPassthrough(unittest.TestCase):

    def test_already_valid_play(self):
        ctx = _ctx(hand_cards=_HAND, monster_summaries=_MONSTERS)
        self.assertEqual("play 1 0", normalize_battle_command("play 1 0", ctx))

    def test_already_valid_end(self):
        ctx = _ctx()
        self.assertEqual("end", normalize_battle_command("end", ctx))

    def test_empty_string(self):
        ctx = _ctx()
        self.assertEqual("", normalize_battle_command("", ctx))


if __name__ == "__main__":
    unittest.main()
