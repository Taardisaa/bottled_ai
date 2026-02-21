import json
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
                    "max_tokens_per_decision: 333",
                    "confidence_threshold: 0.55",
                    "telemetry_enabled: false",
                    "telemetry_path: logs/custom.jsonl",
                ]),
                encoding="utf-8",
            )

            config = load_llm_config(str(config_path))

            self.assertTrue(config.enabled)
            self.assertEqual(["EventHandler", "ShopPurchaseHandler"], config.enabled_handlers)
            self.assertEqual(2100, config.timeout_ms)
            self.assertEqual(333, config.max_tokens_per_decision)
            self.assertEqual(0.55, config.confidence_threshold)
            self.assertFalse(config.telemetry_enabled)
            self.assertEqual("logs/custom.jsonl", config.telemetry_path)

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
