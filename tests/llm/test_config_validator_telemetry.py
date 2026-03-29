import json
import os
import tempfile
import unittest
from pathlib import Path

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import load_llm_config
from rs.llm.telemetry import build_decision_telemetry, write_decision_telemetry
from rs.llm.validator import validate_command


class TestConfigValidatorTelemetry(unittest.TestCase):
    def test_load_llm_config_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_config.yaml"
            config_path.write_text(
                "\n".join([
                    "enabled: true",
                    "enabled_handlers:",
                    "  - EventHandler",
                    "  - ShopPurchaseHandler",
                    "timeout_ms: 2100",
                    "max_retries: 3",
                    "confidence_threshold: 0.55",
                    "telemetry_enabled: false",
                    "telemetry_path: logs/custom.jsonl",
                    "ai_player_graph_enabled: true",
                ]),
                encoding="utf-8",
            )

            previous = {key: os.environ.get(key) for key in [
                "LLM_ENABLED",
                "LLM_ENABLED_HANDLERS",
                "LLM_TIMEOUT_MS",
                "LLM_MAX_RETRIES",
                "LLM_CONFIDENCE_THRESHOLD",
                "LLM_TELEMETRY_ENABLED",
                "LLM_TELEMETRY_PATH",
                "AI_PLAYER_GRAPH_ENABLED",
                "LANGMEM_ENABLED",
                "LANGMEM_SQLITE_PATH",
                "LANGMEM_EMBEDDINGS_BASE_URL",
                "LANGMEM_EMBEDDINGS_API_KEY",
                "LANGMEM_EMBEDDINGS_MODEL",
                "LANGMEM_TOP_K",
                "LANGMEM_REFLECTION_BATCH_SIZE",
                "LANGMEM_MAX_SEMANTIC_MEMORIES_PER_NAMESPACE",
            ]}
            try:
                for key in previous:
                    os.environ.pop(key, None)
                config = load_llm_config(str(config_path))
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertTrue(config.enabled)
            self.assertEqual(["EventHandler", "ShopPurchaseHandler"], config.enabled_handlers)
            self.assertEqual(2100, config.timeout_ms)
            self.assertEqual(3, config.max_retries)
            self.assertEqual(0.55, config.confidence_threshold)
            self.assertFalse(config.telemetry_enabled)
            self.assertEqual("logs/custom.jsonl", config.telemetry_path)
            self.assertTrue(config.ai_player_graph_enabled)
            self.assertFalse(config.langmem_enabled)
            self.assertEqual("dataset/langmem/memory.sqlite3", config.langmem_sqlite_path)
            self.assertEqual("bge-small-en-v1.5", config.langmem_embeddings_model)

    def test_load_llm_config_allows_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_config.yaml"
            config_path.write_text(
                "\n".join([
                    "enabled: false",
                    "enabled_handlers: []",
                    "timeout_ms: 1500",
                    "confidence_threshold: 0.4",
                    "telemetry_enabled: true",
                    "telemetry_path: logs/base.jsonl",
                    "ai_player_graph_enabled: false",
                ]),
                encoding="utf-8",
            )

            previous = {key: os.environ.get(key) for key in [
                "LLM_ENABLED",
                "LLM_ENABLED_HANDLERS",
                "LLM_TIMEOUT_MS",
                "LLM_MAX_RETRIES",
                "LLM_CONFIDENCE_THRESHOLD",
                "AI_PLAYER_GRAPH_ENABLED",
                "LANGMEM_ENABLED",
                "LANGMEM_SQLITE_PATH",
                "LANGMEM_EMBEDDINGS_BASE_URL",
                "LANGMEM_EMBEDDINGS_API_KEY",
                "LANGMEM_EMBEDDINGS_MODEL",
                "LANGMEM_TOP_K",
                "LANGMEM_REFLECTION_BATCH_SIZE",
                "LANGMEM_MAX_SEMANTIC_MEMORIES_PER_NAMESPACE",
            ]}

            try:
                os.environ["LLM_ENABLED"] = "true"
                os.environ["LLM_ENABLED_HANDLERS"] = "EventHandler,ShopPurchaseHandler"
                os.environ["LLM_TIMEOUT_MS"] = "2200"
                os.environ["LLM_MAX_RETRIES"] = "4"
                os.environ["LLM_CONFIDENCE_THRESHOLD"] = "0.77"
                os.environ["AI_PLAYER_GRAPH_ENABLED"] = "true"
                os.environ["LANGMEM_ENABLED"] = "true"
                os.environ["LANGMEM_SQLITE_PATH"] = "dataset/test_langmem.sqlite3"
                os.environ["LANGMEM_EMBEDDINGS_BASE_URL"] = "http://127.0.0.1:9000/v1"
                os.environ["LANGMEM_EMBEDDINGS_API_KEY"] = "local-key"
                os.environ["LANGMEM_EMBEDDINGS_MODEL"] = "bge-test"
                os.environ["LANGMEM_TOP_K"] = "4"
                os.environ["LANGMEM_REFLECTION_BATCH_SIZE"] = "6"
                os.environ["LANGMEM_MAX_SEMANTIC_MEMORIES_PER_NAMESPACE"] = "12"

                config = load_llm_config(str(config_path))
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertTrue(config.enabled)
            self.assertEqual(["EventHandler", "ShopPurchaseHandler"], config.enabled_handlers)
            self.assertEqual(2200, config.timeout_ms)
            self.assertEqual(4, config.max_retries)
            self.assertEqual(0.77, config.confidence_threshold)
            self.assertTrue(config.ai_player_graph_enabled)
            self.assertTrue(config.langmem_enabled)
            self.assertEqual("dataset/test_langmem.sqlite3", config.langmem_sqlite_path)
            self.assertEqual("http://127.0.0.1:9000/v1", config.langmem_embeddings_base_url)
            self.assertEqual("local-key", config.langmem_embeddings_api_key)
            self.assertEqual("bge-test", config.langmem_embeddings_model)
            self.assertEqual(4, config.langmem_top_k)
            self.assertEqual(6, config.langmem_reflection_batch_size)
            self.assertEqual(12, config.langmem_max_semantic_memories_per_namespace)

    def test_validator_rejects_out_of_range_choose_index(self):
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a", "b"],
        )

        result = validate_command(context, "choose 2")

        self.assertFalse(result.is_valid)
        self.assertEqual("index_out_of_range", result.code)

    def test_validator_accepts_named_choice(self):
        context = AgentContext(
            handler_name="Campfire",
            screen_type="REST",
            available_commands=["choose"],
            choice_list=["rest", "smith"],
        )

        result = validate_command(context, "choose smith")

        self.assertTrue(result.is_valid)
        self.assertEqual("ok", result.code)

    def test_telemetry_writes_jsonl_record(self):
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
            game_state={"floor": 1},
        )
        decision = AgentDecision(
            proposed_command="choose 0",
            confidence=0.8,
            explanation="test",
            required_tools_used=["event_db"],
            metadata={"validation_error": "ok"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "llm_decisions.jsonl"
            telemetry = build_decision_telemetry(context, decision, latency_ms=12)
            write_decision_telemetry(telemetry, str(out_path))

            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            payload = json.loads(lines[0])
            self.assertEqual("EventHandler", payload["handler_name"])
            self.assertEqual("choose 0", payload["proposed_command"])
            self.assertEqual("ok", payload["validation_result"])


if __name__ == "__main__":
    unittest.main()
