import unittest

from rs.common.handlers.common_battle_handler import CommonBattleHandler
from rs.llm.agents.battle_meta_advisor_agent import BattleMetaDecision
from test_helpers.resources import load_resource_state


class StubBattleMetaAdvisor:
    def __init__(self, profile_key: str):
        self._profile_key = profile_key

    def decide(self, context):
        return BattleMetaDecision(
            comparator_profile=self._profile_key,
            confidence=0.9,
            explanation="stubbed battle profile override",
            required_tools_used=["stub"],
        )


class TestBattleHandlerMetaAdvisor(unittest.TestCase):
    def test_meta_advisor_can_override_big_fight_with_general_profile(self):
        state = load_resource_state(
            "battles/specific_comparator_cases/big_fight/big_fight_prioritize_power_over_damage.json"
        )

        default_commands = CommonBattleHandler().handle(state).commands
        overridden_commands = CommonBattleHandler(
            battle_meta_advisor=StubBattleMetaAdvisor("general")
        ).handle(state).commands

        self.assertEqual(["play 1"], default_commands)
        self.assertEqual(["play 2 0"], overridden_commands)

    def test_invalid_profile_override_is_ignored(self):
        state = load_resource_state(
            "battles/specific_comparator_cases/big_fight/big_fight_prioritize_artifact_removal_over_damage.json"
        )

        commands = CommonBattleHandler(
            battle_meta_advisor=StubBattleMetaAdvisor("transient")
        ).handle(state).commands

        self.assertEqual(["play 1 0"], commands)


if __name__ == "__main__":
    unittest.main()
