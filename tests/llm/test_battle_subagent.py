import copy
import unittest

from rs.llm.battle_subagent import BattleSubagent, BattleSubagentConfig
from rs.llm.battle_tools import (
    AnalyzeWithCalculatorTool,
    EnumerateLegalActionsTool,
    ExecuteBattleCommandTool,
    RetrieveBattleExperienceTool,
    ValidateBattleCommandTool,
)
from rs.llm.providers.battle_llm_provider import BattleDirective
from rs.machine.state import GameState
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []
        self.custom_memories = []

    def build_context_memory(self, context):
        return {
            "retrieved_episodic_memories": "EP battle memory",
            "retrieved_semantic_memories": "SEM battle memory",
            "langmem_status": "ready",
        }

    def record_accepted_decision(self, context, decision):
        self.recorded.append((context, decision))

    def record_custom_memory(self, context, content, tags=(), reflect=False):
        self.custom_memories.append((context, content, tags, reflect))

    def status(self):
        return "ready"


class FakeBattleRuntime:
    def __init__(self, initial_state, next_states):
        self._current_state = initial_state
        self._next_states = list(next_states)
        self.command_batches = []

    def current_state(self):
        return self._current_state

    def execute(self, commands):
        self.command_batches.append(list(commands))
        if self._next_states:
            self._current_state = self._next_states.pop(0)
        return self._current_state


class ScriptedBattleProvider:
    def __init__(self, directives):
        self._directives = list(directives)
        self.session_ids = []
        self.validation_feedbacks = []

    def propose(self, context, working_memory, tool_descriptions, validation_feedback=None):
        self.session_ids.append(context.extras["battle_working_memory"]["session_id"])
        self.validation_feedbacks.append(dict(validation_feedback or {}))
        if not self._directives:
            return BattleDirective(mode="action", explanation="default_end", confidence=0.5, commands=["end"])
        return self._directives.pop(0)


def build_battle_subagent(provider, langmem_service):
    subagent = BattleSubagent(
        provider=provider,
        langmem_service=langmem_service,
        config=BattleSubagentConfig(max_decision_loops=16, max_tool_calls=16, fallback_max_path_count=100),
    )
    subagent.register_tool(EnumerateLegalActionsTool())
    subagent.register_tool(AnalyzeWithCalculatorTool())
    subagent.register_tool(ValidateBattleCommandTool())
    subagent.register_tool(ExecuteBattleCommandTool())
    subagent.register_tool(RetrieveBattleExperienceTool(langmem_service))
    return subagent


class TestBattleSubagent(unittest.TestCase):
    def test_subagent_can_use_tool_then_commit_action_until_battle_ends(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="tool", explanation="list legal actions", confidence=0.7, tool_name="enumerate_legal_actions"),
            BattleDirective(mode="action", explanation="end turn", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["end"]], runtime.command_batches)
        self.assertEqual(1, len(langmem_service.recorded))
        self.assertEqual(1, len(langmem_service.custom_memories))

    def test_subagent_can_execute_directly_via_execute_tool(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(
                mode="tool",
                explanation="execute immediately",
                confidence=0.9,
                tool_name="execute_battle_command",
                tool_payload={"commands": ["end"]},
            ),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["end"]], runtime.command_batches)
        self.assertEqual(1, len(langmem_service.recorded))

    def test_session_id_is_stable_within_battle_and_rotates_between_battles(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="tool", explanation="inspect state", confidence=0.6, tool_name="enumerate_legal_actions"),
            BattleDirective(mode="action", explanation="end first battle", confidence=0.8, commands=["end"]),
            BattleDirective(mode="tool", explanation="inspect second state", confidence=0.6, tool_name="enumerate_legal_actions"),
            BattleDirective(mode="action", explanation="end second battle", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        battle_state = load_resource_state("battles/general/battle_simple_state.json")
        reward_state = load_resource_state("/card_reward/card_reward_take.json")

        first_runtime = FakeBattleRuntime(battle_state, [reward_state])
        second_runtime = FakeBattleRuntime(battle_state, [reward_state])

        subagent.run(battle_state, first_runtime)
        subagent.run(battle_state, second_runtime)

        first_session_ids = provider.session_ids[:2]
        second_session_ids = provider.session_ids[2:4]
        self.assertEqual(1, len(set(first_session_ids)))
        self.assertEqual(1, len(set(second_session_ids)))
        self.assertNotEqual(first_session_ids[0], second_session_ids[0])

    def test_subagent_can_use_multiple_tools_before_action(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="tool", explanation="refresh memories", confidence=0.7, tool_name="retrieve_battle_experience"),
            BattleDirective(mode="tool", explanation="legacy analysis", confidence=0.7, tool_name="analyze_with_calculator"),
            BattleDirective(mode="action", explanation="end after tools", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual(1, len(langmem_service.recorded))
        self.assertIn("retrieve_battle_experience", langmem_service.recorded[0][1].required_tools_used)
        self.assertIn("analyze_with_calculator", langmem_service.recorded[0][1].required_tools_used)

    def test_subagent_integration_can_run_multiple_battle_steps_before_exit(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="action", explanation="play opening strike", confidence=0.9, commands=["play 1 0"]),
            BattleDirective(mode="action", explanation="end the turn", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        mid_state = load_resource_state("battles/general/another_simple.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [mid_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["play 1 0"], ["end"]], runtime.command_batches)
        self.assertEqual(2, result.steps)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())

    def test_subagent_refines_invalid_choose_action_to_valid_indexed_action(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="action", explanation="bad choose", confidence=0.7, commands=["choose rest"]),
            BattleDirective(mode="action", explanation="fixed", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["end"]], runtime.command_batches)
        self.assertEqual("choose_requires_index", provider.validation_feedbacks[1].get("code"))

    def test_subagent_stops_after_validation_retry_limit(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="action", explanation="bad choose", confidence=0.7, commands=["choose rest"]),
            BattleDirective(mode="action", explanation="bad choose again", confidence=0.7, commands=["choose rest"]),
            BattleDirective(mode="action", explanation="should not run", confidence=0.8, commands=["end"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual(1, len(runtime.command_batches))
        self.assertEqual(2, len(provider.validation_feedbacks))

    def test_subagent_breaks_repeated_choose_loop_with_progression_fallback(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedBattleProvider([
            BattleDirective(mode="action", explanation="choose card", confidence=0.8, commands=["choose 1"]),
            BattleDirective(mode="action", explanation="choose card again", confidence=0.8, commands=["choose 1"]),
            BattleDirective(mode="action", explanation="should be bypassed", confidence=0.8, commands=["choose 1"]),
        ])
        subagent = build_battle_subagent(provider, langmem_service)

        stalled_source = load_resource_state("battles/general/discard.json")
        stalled_payload = copy.deepcopy(stalled_source.json)
        stalled_payload["available_commands"] = ["choose", "confirm", "wait", "state"]
        stalled_state = GameState(copy.deepcopy(stalled_payload))

        runtime = FakeBattleRuntime(
            stalled_state,
            [
                GameState(copy.deepcopy(stalled_payload)),
                GameState(copy.deepcopy(stalled_payload)),
                load_resource_state("/card_reward/card_reward_take.json"),
            ],
        )

        result = subagent.run(stalled_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual(["choose 1"], runtime.command_batches[0])
        self.assertEqual(["choose 1"], runtime.command_batches[1])
        self.assertEqual(["confirm"], runtime.command_batches[2])
        self.assertLessEqual(len(runtime.command_batches), 3)


if __name__ == "__main__":
    unittest.main()
