import unittest
from types import SimpleNamespace

from rs.llm.ai_player_graph import AIPlayerGraph
from rs.llm.config import LlmConfig
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []

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


class TestAIPlayerGraph(unittest.TestCase):
    def test_event_decision_returns_commands_and_records_langmem(self):
        langmem_service = FakeLangMemService()
        graph = AIPlayerGraph(
            config=LlmConfig(enabled=True, ai_player_graph_enabled=True, telemetry_enabled=False),
            langmem_service=langmem_service,
        )
        graph._event_provider = StaticProposalProvider("choose 1")

        state = load_resource_state("/event/divine_fountain.json")
        commands = graph.decide(state)

        self.assertEqual(["choose 1", "wait 30"], commands)
        self.assertEqual(1, len(langmem_service.recorded))

    def test_checkpointed_short_term_memory_persists_within_run(self):
        graph = AIPlayerGraph(
            config=LlmConfig(enabled=True, ai_player_graph_enabled=True, telemetry_enabled=False),
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
            config=LlmConfig(enabled=True, ai_player_graph_enabled=True, telemetry_enabled=False),
            langmem_service=FakeLangMemService(),
        )
        graph._event_provider = StaticProposalProvider("choose 99")

        state = load_resource_state("/event/divine_fountain.json")

        self.assertIsNone(graph.decide(state))

    def test_battle_state_is_not_handled_by_unified_graph(self):
        graph = AIPlayerGraph(
            config=LlmConfig(enabled=True, ai_player_graph_enabled=True, telemetry_enabled=False),
            langmem_service=FakeLangMemService(),
        )

        state = load_resource_state("battles/general/battle_simple_state.json")

        self.assertFalse(graph.can_handle(state))
        self.assertIsNone(graph.decide(state))


if __name__ == "__main__":
    unittest.main()
