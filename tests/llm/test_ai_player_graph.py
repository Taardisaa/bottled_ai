import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rs.llm.ai_player_graph import AIPlayerGraph
from rs.llm.battle_runtime import BattleSessionResult
from rs.llm.campfire_subagent import CampfireSessionResult
from rs.llm.config import LlmConfig
from rs.llm.reward_subagent import RewardSessionResult
from rs.machine.state import GameState
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []
        self.custom_memories = []

    def build_context_memory(self, context):
        return {
            "retrieved_episodic_memories": "none",
            "retrieved_semantic_memories": "none",
            "langmem_status": "ready",
        }

    def record_accepted_decision(self, context, decision):
        self.recorded.append((context, decision))

    def status(self):
        return "ready"

    def record_custom_memory(self, context, content, tags=(), reflect=False):
        self.custom_memories.append((context, content, tags, reflect))


class FakeBattleRuntime:
    def __init__(self, initial_state, next_state):
        self._current_state = initial_state
        self._next_state = next_state
        self.commands = []

    def current_state(self):
        return self._current_state

    def execute(self, commands):
        self.commands.append(list(commands))
        self._current_state = self._next_state
        return self._current_state


class FakeBattleSubagent:
    def __init__(self):
        self.calls = []

    def run(self, state, runtime):
        self.calls.append(state)
        final_state = runtime.execute(["end"])
        return BattleSessionResult(
            handled=True,
            final_state=final_state,
            session_id="fake-session",
            executed_commands=[["end"]],
            steps=1,
            summary="fake battle",
        )


class FakeRewardSubagent:
    def __init__(self, commands=None):
        self.calls = []
        self.commands = ["choose 0"] if commands is None else list(commands)

    def run(self, state, runtime):
        self.calls.append(state)
        final_state = runtime.execute(self.commands)
        return RewardSessionResult(
            handled=True,
            final_state=final_state,
            session_id="fake-reward-session",
            executed_commands=[list(self.commands)],
            steps=1,
            summary="fake reward",
        )


class FakeCampfireSubagent:
    def __init__(self, commands=None):
        self.calls = []
        self.commands = ["choose rest"] if commands is None else list(commands)

    def run(self, state, runtime):
        self.calls.append(state)
        final_state = runtime.execute(self.commands)
        return CampfireSessionResult(
            handled=True,
            final_state=final_state,
            session_id="fake-campfire-session",
            executed_commands=[list(self.commands)],
            steps=1,
            summary="fake campfire",
        )


class StaticProposalProvider:
    def __init__(self, command: str | None, confidence: float = 0.9, explanation: str = "test_policy"):
        self.command = command
        self.confidence = confidence
        self.explanation = explanation

    def propose(self, context):
        return SimpleNamespace(
            proposed_command=self.command,
            confidence=self.confidence,
            explanation=self.explanation,
            metadata={"provider": "static"},
        )


class ScriptedGenericProposalProvider:
    def __init__(self, commands: list[str | None]):
        self.commands = list(commands)
        self.feedbacks = []
        self.calls = 0

    def propose(self, context, validation_feedback=None):
        self.feedbacks.append(validation_feedback)
        command = self.commands[min(self.calls, len(self.commands) - 1)]
        self.calls += 1
        return SimpleNamespace(
            proposed_command=command,
            confidence=0.95,
            explanation="scripted",
            metadata={"provider": "scripted-generic", "calls": self.calls},
        )


class ScriptedProposalProvider:
    def __init__(self, commands: list[str | None]):
        self.commands = list(commands)
        self.feedbacks = []
        self.calls = 0

    def propose(self, context, validation_feedback=None):
        self.feedbacks.append(validation_feedback)
        command = self.commands[min(self.calls, len(self.commands) - 1)]
        self.calls += 1
        return SimpleNamespace(
            proposed_command=command,
            confidence=0.95,
            explanation="scripted",
            metadata={"provider": "scripted", "calls": self.calls},
        )


def _build_unhandled_state() -> GameState:
    return GameState(
        {
            "available_commands": ["confirm", "cancel", "wait", "state"],
            "ready_for_command": True,
            "in_game": True,
            "game_state": {
                "screen_type": "NONE",
                "screen_name": "NONE",
                "screen_state": {},
                "choice_list": [],
                "floor": 5,
                "act": 1,
                "room_phase": "COMPLETE",
                "room_type": "EventRoom",
                "class": "WATCHER",
                "ascension_level": 0,
                "current_hp": 60,
                "max_hp": 72,
                "gold": 99,
                "deck": [],
                "relics": [],
                "potions": [],
                "map": [],
                "keys": {"emerald": False, "ruby": False, "sapphire": False},
                "action_phase": "WAITING_ON_USER",
                "is_screen_up": False,
            },
        }
    )


class TestAIPlayerGraph(unittest.TestCase):
    def test_event_decision_returns_commands_and_records_langmem(self):
        langmem_service = FakeLangMemService()
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=langmem_service,
        )
        graph._event_provider = StaticProposalProvider("choose 1")

        state = load_resource_state("/event/divine_fountain.json")
        commands = graph.decide(state)

        self.assertEqual(["choose 1", "wait 30"], commands)
        self.assertEqual(1, len(langmem_service.recorded))

    def test_checkpointed_short_term_memory_persists_within_run(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._event_provider = StaticProposalProvider("choose 1")
        state = load_resource_state("/event/divine_fountain.json")
        context = graph._build_context(state)
        self.assertIsNotNone(context)
        thread_id = graph._resolve_thread_id(context)
        graph_input = {"context_payload": graph._serialize_context(context)}

        first_output = graph._invoke_with_timeout(graph_input, thread_id)
        second_output = graph._invoke_with_timeout(graph_input, thread_id)

        self.assertEqual(["choose 1", "wait 30"], first_output["commands"])
        self.assertEqual(2, len(second_output["recent_key_decisions"]))
        self.assertIn("EventHandler -> choose 1", second_output["distilled_run_summary"])

    def test_invalid_graph_command_returns_none(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._event_provider = StaticProposalProvider("choose 99")

        state = load_resource_state("/event/divine_fountain.json")

        self.assertIsNone(graph.decide(state))

    def test_battle_state_is_routed_to_battle_subagent(self):
        battle_subagent = FakeBattleSubagent()
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            battle_subagent=battle_subagent,
        )

        state = load_resource_state("battles/general/battle_simple_state.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual(next_state.screen_type(), result.final_state.screen_type())
        self.assertEqual([["end"]], runtime.commands)
        self.assertEqual(1, len(battle_subagent.calls))

    def test_combat_reward_state_is_routed_to_reward_subagent(self):
        reward_subagent = FakeRewardSubagent(["choose 0"])
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            combat_reward_subagent=reward_subagent,
        )

        state = load_resource_state("/combat_reward/combat_reward_gold.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["choose 0"]], runtime.commands)
        self.assertEqual(1, len(reward_subagent.calls))

    def test_card_reward_state_uses_card_reward_provider_not_combat_subagent(self):
        reward_subagent = FakeRewardSubagent(["choose 0"])
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            combat_reward_subagent=reward_subagent,
        )
        graph._card_reward_provider = StaticProposalProvider("choose 0")

        state = load_resource_state("/card_reward/card_reward_take.json")
        commands = graph.decide(state)

        self.assertEqual(["choose 0", "wait 30"], commands)
        self.assertEqual(0, len(reward_subagent.calls))

    def test_campfire_state_is_routed_to_campfire_subagent(self):
        campfire_subagent = FakeCampfireSubagent(["choose rest"])
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            campfire_subagent=campfire_subagent,
        )

        state = load_resource_state("/campfire/campfire_rest.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["choose rest"]], runtime.commands)
        self.assertEqual(1, len(campfire_subagent.calls))

    def test_boss_reward_state_is_routed_to_reward_subagent(self):
        reward_subagent = FakeRewardSubagent(["choose 1"])
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            boss_reward_subagent=reward_subagent,
        )

        state = load_resource_state("/relics/boss_reward_first_is_best.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["choose 1"]], runtime.commands)
        self.assertEqual(1, len(reward_subagent.calls))

    def test_astrolabe_transform_state_is_routed_to_reward_subagent(self):
        reward_subagent = FakeRewardSubagent(["choose 2"])
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
            astrolabe_transform_subagent=reward_subagent,
        )

        state = load_resource_state("/relics/boss_reward_astrolabe.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual([["choose 2"]], runtime.commands)
        self.assertEqual(1, len(reward_subagent.calls))

    def test_chest_state_is_handled_by_unified_executor(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._generic_provider = StaticProposalProvider("choose 0")
        state = load_resource_state("/other/chest_medium_reward.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.commands)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())

    def test_hand_select_state_is_handled_by_unified_executor(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._generic_provider = StaticProposalProvider("choose 0")
        state = load_resource_state("/other/discard_relic.json")
        next_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, next_state)

        self.assertTrue(graph.can_handle(state))
        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.commands)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())

    def test_single_choice_choose_state_is_handled_deterministically(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )

        state = load_resource_state("/event/divine_fountain.json")
        state.json["available_commands"] = ["choose"]
        state.json["game_state"]["choice_list"] = ["leave"]

        self.assertTrue(graph.can_handle(state))
        self.assertEqual(["choose 0"], graph.decide(state))

    def test_single_choice_choose_state_skips_subagent_runtime_path(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )

        state = load_resource_state("/event/divine_fountain.json")
        state.json["available_commands"] = ["choose"]
        state.json["game_state"]["choice_list"] = ["leave"]
        next_state = load_resource_state("/event/divine_fountain.json")
        runtime = FakeBattleRuntime(state, next_state)

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.commands)

    def test_multiple_available_commands_skips_deterministic_path(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/event/divine_fountain.json")
        state.json["game_state"]["choice_list"] = ["leave"]
        self.assertIsNone(graph._deterministic_single_command(state))

    def test_single_non_choose_available_command_is_deterministic(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/event/divine_fountain.json")
        state.json["available_commands"] = ["leave"]
        self.assertEqual(["leave"], graph._deterministic_single_command(state))

    def test_protocol_only_command_is_not_deterministic(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/event/divine_fountain.json")
        state.json["available_commands"] = ["wait"]
        self.assertIsNone(graph._deterministic_single_command(state))

    def test_card_reward_choose_only_is_not_deterministic(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/card_reward/card_reward_take.json")
        state.json["available_commands"] = ["choose"]
        state.json["game_state"]["choice_list"] = ["pommel strike"]
        self.assertIsNone(graph._deterministic_single_command(state))

    def test_combat_reward_scope_is_not_deterministic(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/combat_reward/combat_reward_card.json")
        state.json["available_commands"] = ["choose"]
        state.json["game_state"]["choice_list"] = ["card"]
        self.assertIsNone(graph._deterministic_single_command(state))

    def test_forced_followup_replays_skip_once_on_card_reward(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(state, state)
        run_id = f"{str(state.game_state().get('class', 'unknown')).strip().lower()}:{str(state.game_state().get('seed', 'unknown')).strip().lower()}"
        graph._last_card_reward_skip = {"run_id": run_id, "floor": state.floor()}
        graph._last_card_reward_skip_consumed = False

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual([["skip", "wait 30"]], runtime.commands)
        self.assertTrue(graph._last_card_reward_skip_consumed)

    def test_forced_followup_proceeds_once_on_card_only_combat_reward(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        state = load_resource_state("/combat_reward/combat_reward_card.json")
        runtime = FakeBattleRuntime(state, state)
        run_id = f"{str(state.game_state().get('class', 'unknown')).strip().lower()}:{str(state.game_state().get('seed', 'unknown')).strip().lower()}"
        graph._last_card_reward_skip = {"run_id": run_id, "floor": state.floor()}
        graph._last_card_reward_skip_consumed = False

        result = graph.execute(state, runtime=runtime)

        self.assertIsNotNone(result)
        self.assertTrue(result.handled)
        self.assertEqual([["proceed"]], runtime.commands)
        self.assertTrue(graph._last_card_reward_skip_consumed)

    def test_previously_unhandled_state_routes_to_generic_handler(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._generic_provider = StaticProposalProvider("confirm")

        state = _build_unhandled_state()
        commands = graph.decide(state)

        self.assertTrue(graph.can_handle(state))
        self.assertEqual(["confirm"], commands)

    def test_generic_handler_invalid_proposal_still_blocked_by_validator(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        graph._generic_provider = StaticProposalProvider("choose 99")

        state = _build_unhandled_state()

        self.assertTrue(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))

    def test_generic_handler_retries_with_validation_feedback(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        scripted_provider = ScriptedGenericProposalProvider(["choose", "choose 0"])
        graph._generic_provider = scripted_provider

        state = load_resource_state("/other/discard_relic.json")
        commands = graph.decide(state)

        self.assertEqual(["choose 0"], commands)
        self.assertEqual(2, scripted_provider.calls)
        self.assertIsNone(scripted_provider.feedbacks[0])
        self.assertIsInstance(scripted_provider.feedbacks[1], dict)
        self.assertEqual("choose_requires_index", scripted_provider.feedbacks[1].get("code"))

    def test_event_handler_retries_with_validation_feedback(self):
        graph = AIPlayerGraph(
            config=LlmConfig(
                enabled=True,
                ai_player_graph_enabled=True,
                telemetry_enabled=False,
                graph_trace_enabled=False,
            ),
            langmem_service=FakeLangMemService(),
        )
        scripted_provider = ScriptedProposalProvider(["choose", "choose 0"])
        graph._event_provider = scripted_provider

        state = load_resource_state("/event/divine_fountain.json")
        commands = graph.decide(state)

        self.assertEqual(["choose 0", "wait 30"], commands)
        self.assertEqual(2, scripted_provider.calls)
        self.assertIsNone(scripted_provider.feedbacks[0])
        self.assertIsInstance(scripted_provider.feedbacks[1], dict)
        self.assertEqual("choose_requires_index", scripted_provider.feedbacks[1].get("code"))

    def test_graph_trace_logs_successful_decision_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "graph_trace.jsonl")
            graph = AIPlayerGraph(
                config=LlmConfig(
                    enabled=True,
                    ai_player_graph_enabled=True,
                    telemetry_enabled=False,
                    graph_trace_enabled=True,
                    graph_trace_path=trace_path,
                ),
                langmem_service=FakeLangMemService(),
            )
            graph._event_provider = StaticProposalProvider("choose 1")

            state = load_resource_state("/event/divine_fountain.json")

            commands = graph.decide(state)

            self.assertEqual(["choose 1", "wait 30"], commands)
            payloads = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines()]
            event_types = [payload["event_type"] for payload in payloads]
            node_names = [payload["node_name"] for payload in payloads]
            self.assertIn("graph_decide_start", event_types)
            self.assertIn("graph_decide_success", event_types)
            self.assertIn("ingest_game_state", node_names)
            self.assertIn("validate_decision", node_names)
            self.assertIn("emit_commands", node_names)

    def test_graph_trace_logs_invalid_decision_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "graph_trace.jsonl")
            graph = AIPlayerGraph(
                config=LlmConfig(
                    enabled=True,
                    ai_player_graph_enabled=True,
                    telemetry_enabled=False,
                    graph_trace_enabled=True,
                    graph_trace_path=trace_path,
                ),
                langmem_service=FakeLangMemService(),
            )
            graph._event_provider = StaticProposalProvider("choose 99")

            state = load_resource_state("/event/divine_fountain.json")

            self.assertIsNone(graph.decide(state))

            payloads = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines()]
            event_types = [payload["event_type"] for payload in payloads]
            validation_payloads = [payload for payload in payloads if payload["node_name"] == "validate_decision"]
            self.assertIn("graph_decide_invalid_output", event_types)
            self.assertTrue(
                any(
                    payload["validation_code"] in {"index_out_of_range", "empty_command"}
                    for payload in validation_payloads
                )
            )


if __name__ == "__main__":
    unittest.main()
