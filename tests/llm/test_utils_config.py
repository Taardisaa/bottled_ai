import tempfile
import unittest
import os
from pathlib import Path

from rs.utils.config import load_config


class TestUtilsConfig(unittest.TestCase):
    def test_load_config_reads_yaml_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_utils_config.yaml"
            env_path = Path(tmp) / ".env"

            config_path.write_text(
                "fast_llm_model: gpt-5\n"
                "llm_enable_thinking: true\n"
                "llm_two_layer_struct_convert: true\n"
                "dataset_dir_path: dataset_local\n"
                "cache:\n"
                "  load_options:\n"
                "    llm_query: false\n"
                "  store_options:\n"
                "    llm_query: true\n",
                encoding="utf-8",
            )
            env_path.write_text(
                "OPENAI_API_KEY=test-openai\n"
                "ANTHROPIC_API_KEY=test-anthropic\n"
                "OPENROUTER_API_KEY=test-openrouter\n"
                "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
                "LLM_BASE_URL=http://127.0.0.1:8000/v1\n"
                "LLM_API_KEY=test-local-secret\n",
                encoding="utf-8",
            )

            env_keys = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_ENABLE_THINKING",
                "LLM_TWO_LAYER_STRUCT_CONVERT",
            ]
            previous = {key: os.environ.get(key) for key in env_keys}
            try:
                for key in env_keys:
                    os.environ.pop(key, None)
                loaded = load_config(str(config_path), str(env_path))
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertEqual("gpt-5", loaded.fast_llm_model)
            self.assertTrue(loaded.llm_enable_thinking)
            self.assertTrue(loaded.llm_two_layer_struct_convert)
            self.assertEqual("dataset_local", loaded.dataset_dir_path)
            self.assertEqual("test-openai", loaded.openai_key)
            self.assertEqual("test-anthropic", loaded.anthropic_key)
            self.assertEqual("test-openrouter", loaded.openrouter_key)
            self.assertEqual("https://openrouter.ai/api/v1", loaded.openrouter_base_url)
            self.assertEqual("http://127.0.0.1:8000/v1", loaded.llm_base_url)
            self.assertEqual("test-local-secret", loaded.llm_api_key)
            self.assertFalse(loaded.get_load_cache_option("llm_query"))
            self.assertTrue(loaded.get_store_cache_option("llm_query"))


if __name__ == "__main__":
    unittest.main()
