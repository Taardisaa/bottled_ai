import unittest
from unittest.mock import patch
import os
import io

from pydantic import BaseModel

from rs.utils import llm_utils


class DecisionSchema(BaseModel):
    proposed_command: str | None = None
    confidence: float = 0.0
    explanation: str = ""


class IntSchema(BaseModel):
    x: int


class TestLiteLlmUtils(unittest.TestCase):
    def test_run_llm_preflight_check_reports_basic_info_on_success(self):
        original_llm_base_url = llm_utils.config.llm_base_url
        original_openai_base_url = llm_utils.config.openai_base_url
        try:
            llm_utils.config.llm_base_url = ""
            llm_utils.config.openai_base_url = ""
            with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=True), \
                    patch("rs.utils.llm_utils.litellm_get_max_tokens", return_value=12345), \
                    patch("rs.utils.llm_utils.completion", return_value={
                        "model": "qwen-mlx-live",
                        "choices": [{"message": {"content": "OK"}}],
                        "usage": {"total_tokens": 9},
                    }) as completion_mock:
                result = llm_utils.run_llm_preflight_check(model="openai/qwen-mlx")
        finally:
            llm_utils.config.llm_base_url = original_llm_base_url
            llm_utils.config.openai_base_url = original_openai_base_url

        self.assertTrue(result.available)
        self.assertEqual("openai/qwen-mlx", result.requested_model)
        self.assertEqual("openai/qwen-mlx", result.routed_model)
        self.assertEqual("qwen-mlx", result.normalized_model)
        self.assertEqual("openai-compatible", result.provider)
        self.assertEqual("provider-default", result.endpoint)
        self.assertEqual(12345, result.max_tokens)
        self.assertEqual("qwen-mlx-live", result.response_model)
        self.assertEqual("OK", result.response_preview)
        self.assertEqual(9, result.total_tokens)
        completion_mock.assert_called_once()

    def test_run_llm_preflight_check_returns_failure_when_api_config_missing(self):
        original_llm_base_url = llm_utils.config.llm_base_url
        original_openai_base_url = llm_utils.config.openai_base_url
        try:
            llm_utils.config.llm_base_url = ""
            llm_utils.config.openai_base_url = ""
            with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=False), \
                    patch("rs.utils.llm_utils.completion") as completion_mock:
                result = llm_utils.run_llm_preflight_check(model="gpt-5-mini")
        finally:
            llm_utils.config.llm_base_url = original_llm_base_url
            llm_utils.config.openai_base_url = original_openai_base_url

        self.assertFalse(result.available)
        self.assertEqual("gpt-5-mini", result.requested_model)
        self.assertEqual("openai-compatible", result.provider)
        self.assertIn("Missing or invalid API configuration", result.error)
        completion_mock.assert_not_called()

    def test_normalize_model_for_tokenizer_strips_provider_prefixes(self):
        self.assertEqual("qwen-mlx", llm_utils._normalize_model_for_tokenizer("openai/qwen-mlx"))
        self.assertEqual(
            "gpt-5-mini",
            llm_utils._normalize_model_for_tokenizer("openrouter/openai/gpt-5-mini"),
        )
        self.assertEqual(
            "Qwen/Qwen3-14B",
            llm_utils._normalize_model_for_tokenizer("hosted_vllm/Qwen/Qwen3-14B"),
        )

    def test_count_tokens_prefers_litellm_for_prefixed_model(self):
        with patch("rs.utils.llm_utils.litellm_token_counter", return_value=11) as counter_mock, \
                patch("rs.utils.llm_utils.tiktoken.encoding_for_model") as tiktoken_mock:
            token_count = llm_utils.count_tokens("abc", "openai/qwen-mlx")

        self.assertEqual(11, token_count)
        counter_mock.assert_called_once_with(model="qwen-mlx", text="abc")
        tiktoken_mock.assert_not_called()

    def test_litellm_stdout_noise_is_redirected_to_stderr(self):
        stderr_capture = io.StringIO()

        def noisy_counter(*args, **kwargs):
            print("Provider List: https://docs.litellm.ai/docs/providers")
            return 11

        with patch("rs.utils.llm_utils.litellm_token_counter", side_effect=noisy_counter), \
                patch.object(llm_utils.sys, "stderr", stderr_capture):
            token_count = llm_utils.count_tokens("abc", "openai/qwen-mlx")

        self.assertEqual(11, token_count)
        self.assertIn("Provider List:", stderr_capture.getvalue())

    def test_get_model_token_limit_uses_litellm_with_normalized_model_name(self):
        with patch("rs.utils.llm_utils.litellm_get_max_tokens", return_value=98765) as max_tokens_mock:
            token_limit = llm_utils.get_model_token_limit("openrouter/openai/gpt-5-mini")

        self.assertEqual(98765, token_limit)
        max_tokens_mock.assert_called_once_with("gpt-5-mini")

    def test_get_model_token_limit_falls_back_to_local_override(self):
        with patch("rs.utils.llm_utils.litellm_get_max_tokens", return_value=None):
            self.assertEqual(272000, llm_utils.get_model_token_limit("openai/gpt-5"))

    def test_get_model_token_limit_falls_back_to_static_qwen_mlx_limit(self):
        with patch("rs.utils.llm_utils.litellm_get_max_tokens", side_effect=Exception("not mapped")):
            self.assertEqual(262144, llm_utils.get_model_token_limit("openai/qwen-mlx"))

    def test_get_model_token_limit_suppresses_known_static_qwen_mlx_fallback_log(self):
        with patch("rs.utils.llm_utils.litellm_get_max_tokens", side_effect=Exception("not mapped")), \
                patch("rs.utils.llm_utils.logger.debug") as debug_mock:
            self.assertEqual(262144, llm_utils.get_model_token_limit("openai/qwen-mlx"))

        debug_mock.assert_not_called()

    def test_count_tokens_falls_back_to_tiktoken_without_warning_for_prefixed_model(self):
        class FakeEncoding:
            def encode(self, text):
                return list(text)

        with patch("rs.utils.llm_utils.litellm_token_counter", side_effect=Exception("litellm tokenizer unavailable")), \
                patch("rs.utils.llm_utils.tiktoken.encoding_for_model", side_effect=KeyError("unknown")), \
                patch("rs.utils.llm_utils.tiktoken.get_encoding", return_value=FakeEncoding()), \
                patch("rs.utils.llm_utils.logger.warning") as warning_mock, \
                patch("rs.utils.llm_utils.logger.debug") as debug_mock:
            token_count = llm_utils.count_tokens("abc", "openai/qwen-mlx")

        self.assertEqual(3, token_count)
        warning_mock.assert_not_called()
        self.assertGreaterEqual(debug_mock.call_count, 1)

    def test_count_tokens_falls_back_to_character_estimate_after_litellm_and_tiktoken_fail(self):
        with patch("rs.utils.llm_utils.litellm_token_counter", side_effect=Exception("litellm tokenizer unavailable")), \
                patch("rs.utils.llm_utils.tiktoken.encoding_for_model", side_effect=KeyError("unknown")), \
                patch("rs.utils.llm_utils.tiktoken.get_encoding", side_effect=KeyError("missing encoding")), \
                patch("rs.utils.llm_utils.logger.warning") as warning_mock:
            token_count = llm_utils.count_tokens("abcdefgh", "openai/qwen-mlx")

        self.assertEqual(2, token_count)
        warning_mock.assert_called_once()

    def test_truncate_message_uses_litellm_codec_and_limit_lookup(self):
        with patch("rs.utils.llm_utils.litellm_get_max_tokens", return_value=6), \
                patch("rs.utils.llm_utils.litellm_token_counter", side_effect=[10, 4]), \
                patch("rs.utils.llm_utils.litellm_encode", return_value=list(range(10))) as encode_mock, \
                patch("rs.utils.llm_utils.litellm_decode", return_value="trim") as decode_mock:
            truncated, remaining = llm_utils.truncate_message(
                "abcdefghij",
                model="openai/qwen-mlx",
                reserve_tokens=2,
            )

        self.assertEqual("trim", truncated)
        self.assertEqual(0, remaining)
        encode_mock.assert_called_once_with(model="qwen-mlx", text="abcdefghij")
        decode_mock.assert_called_once_with(model="qwen-mlx", tokens=[0, 1, 2, 3])

    def test_custom_base_url_prefixes_openai_provider(self):
        original_base = llm_utils.config.llm_base_url
        try:
            llm_utils.config.llm_base_url = "http://127.0.0.1:8000/v1"
            self.assertEqual("hosted_vllm/Qwen/Qwen3-14B", llm_utils._normalize_model_for_litellm("Qwen/Qwen3-14B"))
            self.assertEqual("hosted_vllm/gpt-5-mini", llm_utils._normalize_model_for_litellm("gpt-5-mini"))
            self.assertEqual(
                "openrouter/openai/gpt-5-mini",
                llm_utils._normalize_model_for_litellm("openrouter/openai/gpt-5-mini"),
            )
        finally:
            llm_utils.config.llm_base_url = original_base

    def test_ensure_api_key_overwrites_stale_openai_env(self):
        original_env = os.environ.get("OPENAI_API_KEY")
        original_config_key = llm_utils.config.openai_key
        original_local_base = llm_utils.config.llm_base_url
        original_local_key = llm_utils.config.llm_api_key

        try:
            os.environ["OPENAI_API_KEY"] = "stale-key"
            llm_utils.config.openai_key = "fresh-key"
            llm_utils.config.llm_base_url = ""
            llm_utils.config.llm_api_key = ""

            ok = llm_utils._ensure_api_key_for_model("gpt-5-mini")

            self.assertTrue(ok)
            self.assertEqual("fresh-key", os.environ.get("OPENAI_API_KEY"))
        finally:
            llm_utils.config.openai_key = original_config_key
            llm_utils.config.llm_base_url = original_local_base
            llm_utils.config.llm_api_key = original_local_key
            if original_env is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_env

    def test_ask_llm_once_openrouter_uses_litellm_kwargs(self):
        original_key = llm_utils.config.openrouter_key
        original_url = llm_utils.config.openrouter_base_url
        original_local_base = llm_utils.config.llm_base_url
        original_local_key = llm_utils.config.llm_api_key
        original_openai_env = os.environ.get("OPENAI_API_KEY")
        llm_utils.config.openrouter_key = "test-openrouter"
        llm_utils.config.openrouter_base_url = "https://openrouter.ai/api/v1"
        llm_utils.config.llm_base_url = ""
        llm_utils.config.llm_api_key = ""

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
                    two_layer_struct_convert=False,
                )
        finally:
            llm_utils.config.openrouter_key = original_key
            llm_utils.config.openrouter_base_url = original_url
            llm_utils.config.llm_base_url = original_local_base
            llm_utils.config.llm_api_key = original_local_key
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
                two_layer_struct_convert=False,
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

        if response is None:
            self.skipTest("Live LiteLLM backend is not reachable in this environment.")

        self.assertIsInstance(response, DecisionSchema)
        assert isinstance(response, DecisionSchema)
        self.assertIsNotNone(response.proposed_command)
        self.assertGreater(total_tokens, 0)

    def test_custom_base_url_passes_api_base_and_optional_key(self):
        original_base = llm_utils.config.llm_base_url
        original_key = llm_utils.config.llm_api_key
        original_enable_thinking = llm_utils.config.llm_enable_thinking
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "choices": [{"message": {"content": '{"proposed_command":"choose 0","confidence":0.7,"explanation":"ok"}'}}],
                "usage": {"total_tokens": 42},
            }

        try:
            llm_utils.config.llm_base_url = "http://127.0.0.1:8000/v1"
            llm_utils.config.llm_api_key = ""
            llm_utils.config.llm_enable_thinking = True
            with patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
                response, _ = llm_utils.ask_llm_once(
                    message="test",
                    model="Qwen/Qwen3-32B",
                    struct=DecisionSchema,
                    temperature=1.0,
                    enable_cache=False,
                    two_layer_struct_convert=False,
                )
            self.assertIsInstance(response, DecisionSchema)
            self.assertEqual("http://127.0.0.1:8000/v1", captured.get("api_base"))
            self.assertNotIn("api_key", captured)
            self.assertEqual(True, captured.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking"))

            captured.clear()
            llm_utils.config.llm_api_key = "test-local-secret"
            with patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
                response, _ = llm_utils.ask_llm_once(
                    message="test",
                    model="Qwen/Qwen3-32B",
                    struct=DecisionSchema,
                    temperature=1.0,
                    enable_cache=False,
                    two_layer_struct_convert=False,
                )
            self.assertIsInstance(response, DecisionSchema)
            self.assertEqual("http://127.0.0.1:8000/v1", captured.get("api_base"))
            self.assertEqual("test-local-secret", captured.get("api_key"))
            self.assertEqual(True, captured.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking"))
        finally:
            llm_utils.config.llm_base_url = original_base
            llm_utils.config.llm_api_key = original_key
            llm_utils.config.llm_enable_thinking = original_enable_thinking

    def test_custom_base_url_allows_missing_provider_key(self):
        original_base = llm_utils.config.llm_base_url
        original_local_key = llm_utils.config.llm_api_key
        original_openai_key = llm_utils.config.openai_key
        original_env_openai_key = os.environ.get("OPENAI_API_KEY")

        try:
            llm_utils.config.llm_base_url = "http://127.0.0.1:8000/v1"
            llm_utils.config.llm_api_key = ""
            llm_utils.config.openai_key = ""
            os.environ.pop("OPENAI_API_KEY", None)

            ok = llm_utils._ensure_api_key_for_model("gpt-5-mini")

            self.assertTrue(ok)
        finally:
            llm_utils.config.llm_base_url = original_base
            llm_utils.config.llm_api_key = original_local_key
            llm_utils.config.openai_key = original_openai_key
            if original_env_openai_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_env_openai_key

    def test_ask_llm_once_uses_configured_two_layer_and_disables_thinking_on_second_pass(self):
        original_base = llm_utils.config.llm_base_url
        original_key = llm_utils.config.llm_api_key
        original_two_layer = llm_utils.config.llm_two_layer_struct_convert
        original_enable_thinking = llm_utils.config.llm_enable_thinking
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

        try:
            llm_utils.config.llm_base_url = "http://127.0.0.1:8000/v1"
            llm_utils.config.llm_api_key = "test-local-secret"
            llm_utils.config.llm_two_layer_struct_convert = True
            llm_utils.config.llm_enable_thinking = True

            with patch("rs.utils.llm_utils._ensure_api_key_for_model", return_value=True), \
                    patch("rs.utils.llm_utils.completion", side_effect=fake_completion):
                response, total_tokens = llm_utils.ask_llm_once(
                    message="test",
                    model="qwen-mlx",
                    struct=DecisionSchema,
                    temperature=1.0,
                    enable_cache=False,
                )
        finally:
            llm_utils.config.llm_base_url = original_base
            llm_utils.config.llm_api_key = original_key
            llm_utils.config.llm_two_layer_struct_convert = original_two_layer
            llm_utils.config.llm_enable_thinking = original_enable_thinking

        self.assertIsInstance(response, DecisionSchema)
        self.assertEqual(12, total_tokens)
        self.assertEqual(2, len(calls))
        self.assertEqual(
            True,
            calls[0].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking"),
        )
        self.assertEqual(
            False,
            calls[1].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking"),
        )
        self.assertIs(DecisionSchema, calls[1]["response_format"])


if __name__ == "__main__":
    unittest.main()
