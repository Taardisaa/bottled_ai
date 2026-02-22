import tempfile
import unittest
from pathlib import Path

from rs.utils.config import load_config


class TestUtilsConfig(unittest.TestCase):
    def test_load_config_reads_yaml_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_utils_config.yaml"
            env_path = Path(tmp) / ".env"

            config_path.write_text(
                "fast_llm_model: gpt-5\n"
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
                "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
                encoding="utf-8",
            )

            loaded = load_config(str(config_path), str(env_path))

            self.assertEqual("gpt-5", loaded.fast_llm_model)
            self.assertEqual("dataset_local", loaded.dataset_dir_path)
            self.assertEqual("test-openai", loaded.openai_key)
            self.assertEqual("test-anthropic", loaded.anthropic_key)
            self.assertEqual("test-openrouter", loaded.openrouter_key)
            self.assertEqual("https://openrouter.ai/api/v1", loaded.openrouter_base_url)
            self.assertFalse(loaded.get_load_cache_option("llm_query"))
            self.assertTrue(loaded.get_store_cache_option("llm_query"))


if __name__ == "__main__":
    unittest.main()
