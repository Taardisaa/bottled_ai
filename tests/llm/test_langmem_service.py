import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig
from rs.llm.langmem_service import LangMemRepository, LangMemService


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


class LocalEmbeddingsLangMemService(LangMemService):
    def _build_local_embeddings_client(self, model_name):
        return FakeEmbeddings()


class TestLangMemService(unittest.TestCase):
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
                extras={"run_id": "ironclad:seed1", "strategy_name": "peaceful_pummeling"},
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
                extras={"run_id": "ironclad:seed1", "strategy_name": "peaceful_pummeling"},
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
                extras={"run_id": "ironclad:seed1", "strategy_name": "peaceful_pummeling"},
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

    def test_service_disables_when_embeddings_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LangMemService(
                config=LlmConfig(
                    enabled=True,
                    langmem_enabled=True,
                    langmem_sqlite_path=str(Path(tmp) / "memory.sqlite3"),
                ),
            )

            self.assertTrue(service.status().startswith("embeddings_unavailable"))
            service.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
