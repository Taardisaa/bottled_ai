import unittest
from unittest.mock import patch

from pydantic import BaseModel

from rs.utils import llm_utils


class DecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class TestLiteLlmUtils(unittest.TestCase):
    def test_ask_llm_once_openrouter_uses_litellm_kwargs(self):
        original_key = llm_utils.config.openrouter_key
        original_url = llm_utils.config.openrouter_base_url
        llm_utils.config.openrouter_key = "test-openrouter"
        llm_utils.config.openrouter_base_url = "https://openrouter.ai/api/v1"

        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "choices": [{"message": {"content": '{"proposed_command":"choose 0","confidence":0.7,"explanation":"ok"}'}}],
                "usage": {"total_tokens": 42},
            }

        with patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
            response, total_tokens = llm_utils.ask_llm_once(
                message="test",
                model="openrouter/openai/gpt-5-mini",
                struct=DecisionSchema,
                temperature=1.0,
                enable_cache=False,
            )

        llm_utils.config.openrouter_key = original_key
        llm_utils.config.openrouter_base_url = original_url

        self.assertIsInstance(response, DecisionSchema)
        assert isinstance(response, DecisionSchema)
        self.assertEqual("choose 0", response.proposed_command)
        self.assertEqual(42, total_tokens)
        self.assertEqual("openrouter/openai/gpt-5-mini", captured["model"])
        self.assertEqual("https://openrouter.ai/api/v1", captured["api_base"])
        self.assertEqual("test-openrouter", captured["api_key"])
        self.assertIs(DecisionSchema, captured["response_format"])

    def test_ask_llm_multi_preserves_none_entries(self):
        batch_calls = {}

        def fake_batch_completion(**kwargs):
            batch_calls.update(kwargs)
            return [
                {
                    "choices": [{"message": {"content": '{"x": 1}'}}],
                    "usage": {"total_tokens": 3},
                },
                {
                    "choices": [{"message": {"content": '{"x": 2}'}}],
                    "usage": {"total_tokens": 5},
                },
            ]

        with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=True), \
                patch("rs.utils.llm_utils.batch_completion", side_effect=fake_batch_completion):
            responses, total_tokens = llm_utils.ask_llm_multi(
                messages=["a", None, "b"],
                model="gpt-5-mini",
                struct=dict,
                temperature=1.0,
                enable_cache=False,
            )

        self.assertEqual([{"x": 1}, None, {"x": 2}], responses)
        self.assertEqual(8, total_tokens)
        self.assertEqual(2, len(batch_calls["messages"]))


if __name__ == "__main__":
    unittest.main()
