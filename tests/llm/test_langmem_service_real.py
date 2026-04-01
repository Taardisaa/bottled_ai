import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from langchain_openai import ChatOpenAI

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import load_llm_config
from rs.llm.langmem_service import LangMemRepository, LangMemService
from rs.utils.config import config as llm_runtime_config
from rs.utils.llm_utils import run_llm_preflight_check


class TestLangMemServiceReal(unittest.TestCase):
    @staticmethod
    def _extract_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    maybe_text = part.get("text")
                    if isinstance(maybe_text, str):
                        text_parts.append(maybe_text)
            return " ".join(text_parts)
        return str(content)

    @staticmethod
    def _build_chat_model() -> ChatOpenAI:
        base_url = llm_runtime_config.llm_base_url or llm_runtime_config.openai_base_url or None
        api_key = llm_runtime_config.llm_api_key or llm_runtime_config.openai_key or "langmem-real-test"
        return ChatOpenAI(
            model=llm_runtime_config.fast_llm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
        )

    def test_real_service_ready_and_persists_episodic_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            cfg = replace(load_llm_config(), langmem_enabled=True, langmem_sqlite_path=str(db_path))
            service = LangMemService(config=cfg)
            self.addCleanup(service.shutdown, True)

            if not service.is_ready():
                self.skipTest(f"LangMem not ready in this runtime: {service.status()}")

            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose 0"],
                choice_list=["option a", "option b"],
                game_state={"floor": 5, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": "ironclad:realtestseed", "agent_identity": "real_test_agent"},
            )
            decision = AgentDecision(
                proposed_command="choose 0",
                confidence=0.95,
                explanation="real integration persistence check",
                fallback_recommended=False,
            )
            service.record_accepted_decision(context, decision)

            payload = service.build_context_memory(context)
            self.assertTrue(payload["langmem_status"].startswith("ready"))
            self.assertIn("choose 0", payload["retrieved_episodic_memories"])

            service.shutdown(wait=True)
            repo = LangMemRepository(str(db_path))
            records = repo.load_all()
            episodic = [record for record in records if record.memory_type == "episodic"]
            self.assertGreaterEqual(len(episodic), 1)

    def test_custom_pipeline_can_store_and_recall(self):
        preflight = run_llm_preflight_check(model=llm_runtime_config.fast_llm_model)
        if not preflight.available:
            self.skipTest(f"LLM preflight unavailable for real E2E test: {preflight.error}")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            cfg = replace(load_llm_config(), langmem_enabled=True, langmem_sqlite_path=str(db_path))
            service = LangMemService(config=cfg)
            self.addCleanup(service.shutdown, True)
            if not service.is_ready():
                self.skipTest(f"LangMem not ready in this runtime: {service.status()}")

            token = f"DARKMODE_TOKEN_{uuid.uuid4().hex[:12]}"
            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose 0"],
                choice_list=["option a", "option b"],
                game_state={"floor": 8, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": f"ironclad:{uuid.uuid4().hex}", "agent_identity": "real_test_agent"},
            )
            service.record_custom_memory(
                context,
                f"User lighting preference token is {token}.",
                tags=("real_e2e", "preference"),
                reflect=False,
            )
            memory_payload = service.build_context_memory(context)
            episodic_text = str(memory_payload.get("retrieved_episodic_memories", ""))
            self.assertIn(token, episodic_text)

            model = self._build_chat_model()
            response = model.invoke(
                [{
                    "role": "user",
                    "content": (
                        "Using this retrieved memory context, answer with the exact token only.\n"
                        f"Memory: {episodic_text}\n"
                        "Question: What token did I ask you to remember?"
                    ),
                }]
            )
            final_text = self._extract_text(getattr(response, "content", ""))
            self.assertIn(token, final_text)


if __name__ == "__main__":
    unittest.main()
