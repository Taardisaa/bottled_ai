import unittest
from unittest.mock import patch
import os

from pydantic import BaseModel

from rs.utils import llm_utils


class DecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class IntSchema(BaseModel):
    x: int


class TestLiteLlmUtils(unittest.TestCase):
    def test_ensure_api_key_overwrites_stale_openai_env(self):
        original_env = os.environ.get("OPENAI_API_KEY")
        original_config_key = llm_utils.config.openai_key

        try:
            os.environ["OPENAI_API_KEY"] = "stale-key"
            llm_utils.config.openai_key = "fresh-key"

            ok = llm_utils._ensure_api_key_for_model("gpt-5-mini")

            self.assertTrue(ok)
            self.assertEqual("fresh-key", os.environ.get("OPENAI_API_KEY"))
        finally:
            llm_utils.config.openai_key = original_config_key
            if original_env is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_env

    def test_ask_llm_once_openrouter_uses_litellm_kwargs(self):
        original_key = llm_utils.config.openrouter_key
        original_url = llm_utils.config.openrouter_base_url
        original_openai_env = os.environ.get("OPENAI_API_KEY")
        llm_utils.config.openrouter_key = "test-openrouter"
        llm_utils.config.openrouter_base_url = "https://openrouter.ai/api/v1"

        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "choices": [{"message": {"content": '{"proposed_command":"choose 0","confidence":0.7,"explanation":"ok"}'}}],
                "usage": {"total_tokens": 42},
            }

        try:
            with patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
                response, total_tokens = llm_utils.ask_llm_once(
                    message="test",
                    model="openrouter/openai/gpt-5-mini",
                    struct=DecisionSchema,
                    temperature=1.0,
                    enable_cache=False,
                )
        finally:
            llm_utils.config.openrouter_key = original_key
            llm_utils.config.openrouter_base_url = original_url
            if original_openai_env is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_openai_env

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
                struct=IntSchema,
                temperature=1.0,
                enable_cache=False,
            )

        self.assertIsInstance(responses[0], IntSchema)
        self.assertIsNone(responses[1])
        self.assertIsInstance(responses[2], IntSchema)
        assert isinstance(responses[0], IntSchema)
        assert isinstance(responses[2], IntSchema)
        self.assertEqual(1, responses[0].x)
        self.assertEqual(2, responses[2].x)
        self.assertEqual(8, total_tokens)
        self.assertEqual(2, len(batch_calls["messages"]))

    def test_ask_llm_once_two_layer_struct_convert_calls_twice(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "choices": [{"message": {"content": "Pick choose 0 because safe."}}],
                    "usage": {"total_tokens": 7},
                }
            return {
                "choices": [{"message": {"content": '{"proposed_command":"choose 0","confidence":0.8,"explanation":"converted"}'}}],
                "usage": {"total_tokens": 5},
            }

        with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=True), \
                patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
            response, total_tokens = llm_utils.ask_llm_once(
                message="test",
                model="gpt-5-mini",
                struct=DecisionSchema,
                temperature=1.0,
                enable_cache=False,
                two_layer_struct_convert=True,
            )

        self.assertIsInstance(response, DecisionSchema)
        assert isinstance(response, DecisionSchema)
        self.assertEqual("choose 0", response.proposed_command)
        self.assertEqual(12, total_tokens)
        self.assertEqual(2, len(calls))
        self.assertNotIn("response_format", calls[0])
        self.assertIs(DecisionSchema, calls[1]["response_format"])

    def test_ask_llm_multi_two_layer_preserves_none_entries(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            pair_idx = (len(calls) - 1) // 2
            if len(calls) % 2 == 1:
                return {
                    "choices": [{"message": {"content": f"raw response {pair_idx}"}}],
                    "usage": {"total_tokens": 2},
                }
            return {
                "choices": [{"message": {"content": '{"x": 1}' if pair_idx == 0 else '{"x": 2}'}}],
                "usage": {"total_tokens": 3},
            }

        with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=True), \
                patch("rs.utils.llm_utils.completion", side_effect=fake_completion), \
                patch("rs.utils.llm_utils.batch_completion") as fake_batch:
            responses, total_tokens = llm_utils.ask_llm_multi(
                messages=["a", None, "b"],
                model="gpt-5-mini",
                struct=IntSchema,
                temperature=1.0,
                enable_cache=False,
                two_layer_struct_convert=True,
            )

        self.assertIsInstance(responses[0], IntSchema)
        self.assertIsNone(responses[1])
        self.assertIsInstance(responses[2], IntSchema)
        assert isinstance(responses[0], IntSchema)
        assert isinstance(responses[2], IntSchema)
        self.assertEqual(1, responses[0].x)
        self.assertEqual(2, responses[2].x)
        self.assertEqual(10, total_tokens)
        self.assertEqual(4, len(calls))
        fake_batch.assert_not_called()

    def test_live_litellm_structured_call(self):
        response, total_tokens = llm_utils.ask_llm_once(
            message=(
                "Return a decision JSON. Set proposed_command to 'choose 0', "
                "confidence to 0.75, and explanation to 'live_test'."
            ),
            model=llm_utils.config.fast_llm_model,
            struct=DecisionSchema,
            temperature=1.0,
            enable_cache=False,
        )

        self.assertIsInstance(response, DecisionSchema)
        assert isinstance(response, DecisionSchema)
        self.assertIsNotNone(response.proposed_command)
        self.assertGreater(total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
