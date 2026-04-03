import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig
from rs.llm.langmem_service import (
    CompatibilityToolCallChatModel,
    LangMemRepository,
    LangMemService,
    _patch_bind_tools_method,
    _repair_ai_message_tool_calls,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        lowered = str(text).lower()
        return [
            1.0 if "choose" in lowered else 0.0,
            1.0 if "event" in lowered else 0.0,
            float(len(lowered) % 17),
        ]


class FakeReflectionManager:
    def __init__(self, namespace, store):
        self.namespace = namespace
        self.store = store

    def invoke(self, payload):
        return [{
            "namespace": self.namespace,
            "key": "semantic-memory-1",
            "value": {
                "kind": "Memory",
                "content": {"content": "Prefer safe event choices when HP is low."},
            },
        }]


class FakeEmptyReflectionManager:
    def __init__(self, namespace, store):
        self.namespace = namespace
        self.store = store

    def invoke(self, payload):
        return []


class FakeRetryingReflectionManager:
    def __init__(self, namespace, store):
        self.namespace = namespace
        self.store = store
        self.attempt_count = 0

    def invoke(self, payload):
        self.attempt_count += 1
        if self.attempt_count < 3:
            return []
        return [{
            "namespace": self.namespace,
            "key": "semantic-memory-after-retry",
            "value": {
                "kind": "Memory",
                "content": {"content": "Prioritize early blocking in this kind of fight."},
            },
        }]


class FakeMarkupToolCallChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake_markup_tool_call"

    @property
    def _identifying_params(self) -> dict:
        return {}

    def _generate(self, messages, stop=None, **kwargs):
        message = AIMessage(
            content=(
                '<tool_call>'
                '{"name":"Memory","arguments":{"content":"Block early to preserve HP."}}'
                '</tool_call>'
            )
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        def _invoke(input_value, config=None):
            return AIMessage(
                content=(
                    '<tool_call>'
                    '{"name":"Memory","arguments":{"content":"Block early to preserve HP."}}'
                    '</tool_call>'
                )
            )

        async def _ainvoke(input_value, config=None):
            return _invoke(input_value, config=config)

        return RunnableLambda(_invoke, afunc=_ainvoke)


class FakeBoundToolRunnable:
    def invoke(self, input_value, config=None):
        return AIMessage(
            content=(
                '<tool_call>'
                '{"name":"Memory","arguments":{"content":"Block early to preserve HP."}}'
                '</tool_call>'
            )
        )

    async def ainvoke(self, input_value, config=None):
        return self.invoke(input_value, config=config)


class FakePatchedBindToolsChatModel:
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return FakeBoundToolRunnable()


class LocalEmbeddingsLangMemService(LangMemService):
    def _build_local_embeddings_client(self, model_name):
        return FakeEmbeddings()


class BrokenEmbeddingsLangMemService(LangMemService):
    def _build_embeddings_client(self):
        raise ValueError("simulated_embedding_init_failure")


class TestLangMemService(unittest.TestCase):
    def test_compatibility_tool_call_model_parses_markup_into_tool_calls(self):
        wrapped = CompatibilityToolCallChatModel(model=FakeMarkupToolCallChatModel())

        response = wrapped.invoke([HumanMessage(content="extract a memory")])

        self.assertEqual(1, len(response.tool_calls))
        self.assertEqual("Memory", response.tool_calls[0]["name"])
        self.assertEqual({"content": "Block early to preserve HP."}, response.tool_calls[0]["args"])

    def test_repair_ai_message_tool_calls_keeps_existing_tool_calls(self):
        message = AIMessage(
            content="already parsed",
            tool_calls=[{"name": "Memory", "args": {"content": "keep this"}, "id": "x", "type": "tool_call"}],
        )

        repaired = _repair_ai_message_tool_calls(message)

        self.assertIs(message, repaired)

    def test_compatibility_tool_call_model_repairs_bound_tool_calls(self):
        wrapped = CompatibilityToolCallChatModel(model=FakeMarkupToolCallChatModel())

        response = wrapped.bind_tools([{"type": "function", "function": {"name": "Memory"}}]).invoke(
            [HumanMessage(content="extract a memory")]
        )

        self.assertEqual(1, len(response.tool_calls))
        self.assertEqual("Memory", response.tool_calls[0]["name"])
        self.assertEqual({"content": "Block early to preserve HP."}, response.tool_calls[0]["args"])

    def test_bind_tools_patch_repairs_markup_tool_calls(self):
        _patch_bind_tools_method(FakePatchedBindToolsChatModel)

        response = FakePatchedBindToolsChatModel().bind_tools(
            [{"type": "function", "function": {"name": "Memory"}}]
        ).invoke([HumanMessage(content="extract a memory")])

        self.assertEqual(1, len(response.tool_calls))
        self.assertEqual("Memory", response.tool_calls[0]["name"])
        self.assertEqual({"content": "Block early to preserve HP."}, response.tool_calls[0]["args"])

    def test_repository_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = LangMemRepository(str(Path(tmp) / "memory.sqlite3"))
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose"],
                choice_list=["a"],
                game_state={"floor": 5, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )
            service.record_accepted_decision(
                context,
                AgentDecision(
                    proposed_command="choose 0",
                    confidence=0.9,
                    explanation="safe relic line",
                ),
            )
            service.shutdown(wait=True)

            records = repo.load_all()
            self.assertEqual(1, len(records))
            self.assertEqual("episodic", records[0].memory_type)
            self.assertIn("choose 0", records[0].content)

    def test_service_retrieves_episodic_memory_for_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                    langmem_top_k=3,
                ),
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose"],
                choice_list=["a"],
                game_state={"floor": 5, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )
            service.record_accepted_decision(
                context,
                AgentDecision(
                    proposed_command="choose 0",
                    confidence=0.9,
                    explanation="safe event choice",
                ),
            )

            payload = service.build_context_memory(context)

            self.assertEqual("ready", payload["langmem_status"])
            self.assertIn("choose 0", payload["retrieved_episodic_memories"])
            self.assertEqual("none", payload["retrieved_semantic_memories"])
            service.shutdown(wait=True)

    def test_service_uses_local_embeddings_when_no_remote_endpoint_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LocalEmbeddingsLangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                    langmem_embeddings_base_url="",
                    langmem_embeddings_model="BAAI/bge-small-en-v1.5",
                ),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )

            self.assertEqual("ready", service.status())
            service.shutdown(wait=True)

    def test_finalize_run_creates_semantic_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "memory.sqlite3"
            repo = LangMemRepository(str(repo_path))
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                    langmem_reflection_batch_size=1,
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose"],
                choice_list=["a"],
                game_state={"floor": 9, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )

            service.finalize_run(
                context,
                {
                    "floor": 9,
                    "score": 123,
                    "victory": False,
                    "bosses": ["Slime Boss"],
                    "elites": ["Gremlin Nob"],
                    "run_memory_summary": "IRONCLAD act 1 floor 9 hp 33/80 gold 120",
                },
            )
            service.shutdown(wait=True)

            records = repo.load_all()
            semantic_records = [record for record in records if record.memory_type == "semantic"]
            self.assertEqual(1, len(semantic_records))
            self.assertIn("Prefer safe event choices", semantic_records[0].content)
            semantic_items = service._store.search(("semantic", "neo_primates", "ironclad", "EventHandler"), limit=5)
            self.assertEqual({"content": "Prefer safe event choices when HP is low."}, semantic_items[0].value["content"])

    def test_hydrated_semantic_memory_restores_structured_content_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "memory.sqlite3"
            repo = LangMemRepository(str(repo_path))
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                    langmem_reflection_batch_size=1,
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="BattleHandler",
                screen_type="BATTLE",
                available_commands=["play 1"],
                choice_list=[],
                game_state={"floor": 3, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )
            service.record_custom_memory(
                context,
                "Battle session ended on floor 3. Executed 2 command batches.",
                tags=("battle_summary", "BattleHandler"),
                reflect=True,
            )
            service.shutdown(wait=True)

            rehydrated = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeReflectionManager(namespace, store),
            )
            semantic_items = rehydrated._store.search(("semantic", "neo_primates", "ironclad", "BattleHandler"), limit=5)
            self.assertEqual(1, len(semantic_items))
            self.assertEqual({"content": "Prefer safe event choices when HP is low."}, semantic_items[0].value["content"])
            rehydrated.shutdown(wait=True)

    def test_reflect_batch_retries_up_to_three_attempts_before_persisting_semantic_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "memory.sqlite3"
            repo = LangMemRepository(str(repo_path))
            retrying_manager = FakeRetryingReflectionManager(("semantic", "neo_primates", "ironclad", "BattleHandler"), None)
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: retrying_manager,
            )
            context = AgentContext(
                handler_name="BattleHandler",
                screen_type="BATTLE",
                available_commands=["play 1"],
                choice_list=[],
                game_state={"floor": 3, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )

            service.record_custom_memory(
                context,
                "Battle session ended on floor 3. Executed 2 command batches. Recent battle notes: defended then attacked.",
                tags=("battle_summary", "BattleHandler"),
                reflect=True,
            )
            service.shutdown(wait=True)

            records = repo.load_all()
            semantic_records = [record for record in records if record.memory_type == "semantic"]
            self.assertEqual(3, retrying_manager.attempt_count)
            self.assertEqual(1, len(semantic_records))
            self.assertEqual(("semantic", "neo_primates", "ironclad", "BattleHandler"), semantic_records[0].namespace)
            self.assertIn("Prioritize early blocking", semantic_records[0].content)

    def test_record_custom_memory_with_reflection_does_not_persist_semantic_when_manager_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "memory.sqlite3"
            repo = LangMemRepository(str(repo_path))
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeEmptyReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="BattleHandler",
                screen_type="BATTLE",
                available_commands=["play 1"],
                choice_list=[],
                game_state={"floor": 3, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )

            service.record_custom_memory(
                context,
                "Battle session ended on floor 3. Executed 2 command batches. Recent battle notes: defended then attacked.",
                tags=("battle_summary", "BattleHandler"),
                reflect=True,
            )
            service.shutdown(wait=True)

            records = repo.load_all()
            episodic_records = [record for record in records if record.memory_type == "episodic"]
            semantic_records = [record for record in records if record.memory_type == "semantic"]
            self.assertEqual(1, len(episodic_records))
            self.assertEqual(0, len(semantic_records))
            self.assertEqual(("run", "neo_primates", "ironclad", "seed1"), episodic_records[0].namespace)
            self.assertEqual(
                ("custom_memory", "BattleHandler", "battle_summary", "BattleHandler"),
                episodic_records[0].tags,
            )

    def test_finalize_run_does_not_persist_semantic_when_manager_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "memory.sqlite3"
            repo = LangMemRepository(str(repo_path))
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(repo_path),
                    langmem_reflection_batch_size=1,
                ),
                repository=repo,
                embeddings_client=FakeEmbeddings(),
                reflection_manager_factory=lambda namespace, store: FakeEmptyReflectionManager(namespace, store),
            )
            context = AgentContext(
                handler_name="RunFinalizer",
                screen_type="GAME_OVER",
                available_commands=[],
                choice_list=[],
                game_state={"floor": 9, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:seed1", "agent_identity": "neo_primates"},
            )

            service.finalize_run(
                context,
                {
                    "floor": 9,
                    "score": 123,
                    "victory": False,
                    "bosses": ["Slime Boss"],
                    "elites": ["Gremlin Nob"],
                    "run_memory_summary": "IRONCLAD act 1 floor 9 hp 33/80 gold 120",
                },
            )
            service.shutdown(wait=True)

            records = repo.load_all()
            semantic_records = [record for record in records if record.memory_type == "semantic"]
            self.assertEqual([], semantic_records)

    def test_service_disables_when_embeddings_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = BrokenEmbeddingsLangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                ),
            )

            self.assertTrue(service.status().startswith("embeddings_unavailable"))
            service.shutdown(wait=True)

    def test_service_does_not_create_sqlite_when_embeddings_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            service = BrokenEmbeddingsLangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(db_path),
                ),
            )
            self.assertTrue(service.status().startswith("embeddings_unavailable"))
            self.assertFalse(service.is_ready())

            service.shutdown(wait=True)
            self.assertFalse(db_path.exists())

    def test_service_raises_when_embeddings_unavailable_and_fail_fast_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as raised:
                BrokenEmbeddingsLangMemService(
                    config=LlmConfig(
                        enabled=True,
                        langmem_enabled=True,
                        langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                        langmem_fail_fast_init=True,
                    ),
                )

            self.assertIn("LangMem initialization failed", str(raised.exception))

if __name__ == "__main__":
    unittest.main()
