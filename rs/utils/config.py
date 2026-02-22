from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from rs.utils.path_utils import get_repo_root
from rs.utils.yaml_utils import load_yaml_mapping


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class Config:
    fast_llm_model: str = "gpt-5-mini"
    dataset_dir_path: str = "dataset"
    openai_key: str = ""
    anthropic_key: str = ""
    load_cache_options: dict[str, bool] = field(default_factory=lambda: {"llm_query": True})
    store_cache_options: dict[str, bool] = field(default_factory=lambda: {"llm_query": True})

    def get_load_cache_option(self, key: str) -> bool:
        return bool(self.load_cache_options.get(key, False))

    def get_store_cache_option(self, key: str) -> bool:
        return bool(self.store_cache_options.get(key, False))


def load_config(config_path: str | None = None, env_path: str | None = None) -> Config:
    root = get_repo_root()
    yaml_path = Path(config_path) if config_path is not None else root / "configs" / "llm_utils_config.yaml"
    dotenv_path = Path(env_path) if env_path is not None else root / ".env"

    values = load_yaml_mapping(yaml_path)
    dotenv_values = _load_dotenv(dotenv_path)

    openai_key = os.environ.get("OPENAI_API_KEY") or dotenv_values.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or dotenv_values.get("ANTHROPIC_API_KEY", "")

    cache_values = values.get("cache", {})
    load_cache_options = dict(cache_values.get("load_options", {"llm_query": True}))
    store_cache_options = dict(cache_values.get("store_options", {"llm_query": True}))

    return Config(
        fast_llm_model=str(values.get("fast_llm_model", "gpt-5-mini")),
        dataset_dir_path=str(values.get("dataset_dir_path", "dataset")),
        openai_key=str(openai_key),
        anthropic_key=str(anthropic_key),
        load_cache_options=load_cache_options,
        store_cache_options=store_cache_options,
    )


config = load_config()
