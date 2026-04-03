import copy
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from rs.llm.battle_subagent import BattleSubagent, BattleSubagentConfig
from rs.llm.langmem_service import _patch_bind_tools_method
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


class ScriptedChatModel:
    """Mock ChatModel that returns pre-scripted AIMessage responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invoke_calls.append(list(messages))
        if not self._responses:
            return _submit(["end"])
        return self._responses.pop(0)


class PatchedScriptedChatModel(ScriptedChatModel):
    pass


_patch_bind_tools_method(PatchedScriptedChatModel)


def _tc(name, args=None, tc_id=None):
    return {"name": name, "args": args or {}, "id": tc_id or f"tc_{name}", "type": "tool_call"}


def _submit(commands, content="", tc_id=None):
    return AIMessage(
        content=content,
        tool_calls=[_tc("submit_battle_commands", {"commands": commands}, tc_id=tc_id or "tc_submit")],
    )


def _tool_call(tool_name, args=None, content="", tc_id=None):
    return AIMessage(
        content=content,
        tool_calls=[_tc(tool_name, args or {}, tc_id=tc_id or f"tc_{tool_name}")],
    )


def _markup_tool_call(name, args=None):
    return AIMessage(
        content=(
            "<tool_call>"
            + str({
                "name": name,
                "arguments": args or {},
            }).replace("'", '"')
            + "</tool_call>"
        )
    )


def build_battle_subagent(chat_model, langmem_service):
    return BattleSubagent(
        chat_model=chat_model,
        langmem_service=langmem_service,
        config=BattleSubagentConfig(max_decision_loops=16, max_tool_calls=16, fallback_max_path_count=100),
    )


class TestBattleSubagent(unittest.TestCase):
    def test_subagent_can_use_tool_then_commit_action_until_battle_ends(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _tool_call("enumerate_legal_actions"),
            _submit(["end"]),
        ])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["end"]], runtime.command_batches)
        self.assertEqual(1, len(langmem.recorded))
        self.assertEqual(1, len(langmem.custom_memories))
        custom_context, custom_content, custom_tags, custom_reflect = langmem.custom_memories[0]
        self.assertEqual("BattleHandler", custom_context.handler_name)
        self.assertTrue(custom_content.startswith("Battle session ended on floor"))
        self.assertEqual(("battle_summary", "BattleHandler"), custom_tags)
        self.assertTrue(custom_reflect)

    def test_subagent_can_submit_commands_directly(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([_submit(["end"])])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["end"]], runtime.command_batches)
        self.assertEqual(1, len(langmem.recorded))

    def test_subagent_repairs_markup_tool_calls_before_validation(self):
        langmem = FakeLangMemService()
        model = PatchedScriptedChatModel([_markup_tool_call("submit_battle_commands", {"commands": ["end"]})])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        with patch("rs.llm.battle_subagent.log_to_run") as log_mock:
            result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["end"]], runtime.command_batches)
        messages = [call.args[0] for call in log_mock.call_args_list]
        self.assertTrue(any("tool_calls=['submit_battle_commands']" in message for message in messages))
        self.assertFalse(any("validation produced no commands" in message for message in messages))

    def test_subagent_logs_per_step_decision_flow(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([_submit(["end"], content="End the turn after reviewing the hand.")])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        with patch("rs.llm.battle_subagent.log_to_run") as log_mock:
            result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        messages = [call.args[0] for call in log_mock.call_args_list]
        self.assertTrue(any("BattleSubagent memory loaded:" in message for message in messages))
        self.assertTrue(any("BattleSubagent state ingest:" in message for message in messages))
        self.assertTrue(any("BattleSubagent model response:" in message for message in messages))
        self.assertTrue(any("BattleSubagent accepted submission:" in message for message in messages))
        self.assertTrue(any("BattleSubagent executing commands:" in message for message in messages))
        self.assertTrue(any("BattleSubagent post-execution state:" in message for message in messages))

    def test_session_id_is_stable_within_battle_and_rotates_between_battles(self):
        langmem = FakeLangMemService()
        battle_state = load_resource_state("battles/general/battle_simple_state.json")
        reward_state = load_resource_state("/card_reward/card_reward_take.json")

        model = ScriptedChatModel([
            _tool_call("enumerate_legal_actions"),
            _submit(["end"], tc_id="tc_first"),
            _tool_call("enumerate_legal_actions"),
            _submit(["end"], tc_id="tc_second"),
        ])
        subagent = build_battle_subagent(model, langmem)

        first_runtime = FakeBattleRuntime(battle_state, [reward_state])
        second_runtime = FakeBattleRuntime(battle_state, [reward_state])

        result1 = subagent.run(battle_state, first_runtime)
        result2 = subagent.run(battle_state, second_runtime)

        self.assertNotEqual("", result1.session_id)
        self.assertNotEqual("", result2.session_id)
        self.assertNotEqual(result1.session_id, result2.session_id)

    def test_subagent_can_use_multiple_tools_before_action(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _tool_call("retrieve_battle_experience"),
            _tool_call("analyze_with_calculator"),
            _submit(["end"]),
        ])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual(1, len(langmem.recorded))
        self.assertIn("retrieve_battle_experience", langmem.recorded[0][1].required_tools_used)
        self.assertIn("analyze_with_calculator", langmem.recorded[0][1].required_tools_used)

    def test_subagent_integration_can_run_multiple_battle_steps_before_exit(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _submit(["play 1 0"]),
            _submit(["end"]),
        ])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        mid_state = load_resource_state("battles/general/another_simple.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [mid_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["play 1 0"], ["end"]], runtime.command_batches)
        self.assertEqual(2, result.steps)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())

    def test_subagent_refines_invalid_submission_via_corrective_message(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _submit(["choose rest"]),   # invalid — corrective message injected
            _submit(["end"]),           # valid on retry
        ])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["end"]], runtime.command_batches)
        # Second invoke call must include the corrective HumanMessage
        self.assertGreaterEqual(len(model.invoke_calls), 2)
        second_call_messages = model.invoke_calls[1]
        self.assertTrue(
            any("validation_error" in str(m.content) for m in second_call_messages),
            "Expected corrective validation_error message in second invoke",
        )

    def test_subagent_uses_guardrail_after_validation_retry_limit(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _submit(["choose rest"]),   # invalid attempt 1
            _submit(["choose rest"]),   # invalid attempt 2 — exhausted, guardrail fires
        ])
        subagent = build_battle_subagent(model, langmem)
        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual(1, len(runtime.command_batches))
        self.assertEqual(2, len(model.invoke_calls))

    def test_subagent_breaks_repeated_choose_loop_with_progression_fallback(self):
        langmem = FakeLangMemService()
        model = ScriptedChatModel([
            _submit(["choose 1"]),
            _submit(["choose 1"]),
            _submit(["choose 1"]),   # should be bypassed by no-progress guardrail
        ])
        subagent = build_battle_subagent(model, langmem)

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
