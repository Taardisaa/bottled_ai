import unittest

from rs.llm.agents.base_agent import AgentDecision
from rs.llm.agents.battle_meta_advisor_agent import BattleMetaDecision
from rs.llm.benchmark_suite import LlmBenchmarkCase
from rs.llm.offline_benchmark import run_offline_llm_benchmark_suite, summarize_offline_benchmark_results
from rs.machine.character import Character


class StubOrchestrator:
    def __init__(self, decisions_by_handler: dict[str, AgentDecision | None]):
        self._decisions_by_handler = decisions_by_handler

    def decide(self, handler_name: str, context):
        return self._decisions_by_handler.get(handler_name)


class StubBattleMetaAdvisor:
    def __init__(self, decision: BattleMetaDecision):
        self._decision = decision

    def decide(self, context):
        return self._decision


class TestOfflineBenchmark(unittest.TestCase):
    def test_runner_reports_command_changes_for_event_and_battle_cases(self):
        cases = [
            LlmBenchmarkCase(
                case_id="event_divine_fountain",
                fixture_path="tests/res/event/divine_fountain.json",
                handler_area="event",
                phase="phase_1",
                seed="unused",
                character=Character.WATCHER,
                recommended_strategy="peaceful_pummeling",
                act=1,
                floor=5,
                room_type="EventRoom",
                tags=("event",),
            ),
            LlmBenchmarkCase(
                case_id="battle_big_fight_override",
                fixture_path="tests/res/battles/specific_comparator_cases/big_fight/big_fight_prioritize_power_over_damage.json",
                handler_area="battle",
                phase="phase_4",
                seed="unused",
                character=Character.IRONCLAD,
                recommended_strategy="requested_strike",
                act=3,
                floor=33,
                room_type="MonsterRoomBoss",
                tags=("battle",),
            ),
        ]

        orchestrator = StubOrchestrator(
            {
                "EventHandler": AgentDecision(
                    proposed_command="choose 1",
                    confidence=0.9,
                    explanation="benchmark event override",
                ),
            }
        )
        battle_meta_advisor = StubBattleMetaAdvisor(
            BattleMetaDecision(
                comparator_profile="general",
                confidence=0.91,
                explanation="benchmark battle override",
            )
        )

        results = run_offline_llm_benchmark_suite(
            cases=cases,
            orchestrator=orchestrator,
            battle_meta_advisor=battle_meta_advisor,
        )

        self.assertEqual(2, len(results))

        event_result = next(result for result in results if result.handler_area == "event")
        self.assertTrue(event_result.changed)
        self.assertEqual(["choose 0", "wait 30"], event_result.baseline_commands)
        self.assertEqual(["choose 1", "wait 30"], event_result.assisted_commands)
        self.assertEqual("choose 1", event_result.advisor_output)
        self.assertTrue(event_result.advisor_used)
        self.assertFalse(event_result.fallback_used)

        battle_result = next(result for result in results if result.handler_area == "battle")
        self.assertTrue(battle_result.changed)
        self.assertEqual(["play 1"], battle_result.baseline_commands)
        self.assertEqual(["play 2 0"], battle_result.assisted_commands)
        self.assertEqual("general", battle_result.advisor_output)
        self.assertTrue(battle_result.advisor_used)
        self.assertFalse(battle_result.fallback_used)

        summary = summarize_offline_benchmark_results(results)
        self.assertEqual(2, summary["total_cases"])
        self.assertEqual(2, summary["changed_cases"])
        self.assertEqual(2, summary["advisor_used_cases"])
        self.assertEqual(0, summary["fallback_cases"])
        self.assertEqual(1, summary["by_handler_area"]["event"]["changed_cases"])
        self.assertEqual(1, summary["by_handler_area"]["battle"]["changed_cases"])


if __name__ == "__main__":
    unittest.main()
