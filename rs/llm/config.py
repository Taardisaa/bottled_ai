from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from rs.utils.path_utils import get_repo_root
from rs.utils.yaml_utils import load_yaml_mapping


@dataclass
class LlmConfig:
    """Configuration values for the LLM advisor subsystem."""

    enabled: bool = False
    enabled_handlers: List[str] | None = None
    timeout_ms: int = 1500
    max_tokens_per_decision: int = 1200
    confidence_threshold: float = 0.4
    telemetry_enabled: bool = True
    telemetry_path: str = "logs/llm_decisions.jsonl"

    def __post_init__(self):
        """Normalize mutable defaults after dataclass initialization.

        Args:
            None.

        Returns:
            None.
        """
        if self.enabled_handlers is None:
            self.enabled_handlers = []


def load_llm_config(config_path: str | None = None) -> LlmConfig:
    """Load LLM config from yaml with safe defaults.

    Args:
        config_path: Optional path to config file. Defaults to `configs/llm_config.yaml`.

    Returns:
        LlmConfig: Parsed configuration with defaults for missing keys.
    """
    root = get_repo_root()
    path = Path(config_path) if config_path is not None else root / "configs" / "llm_config.yaml"
    values = load_yaml_mapping(path)

    return LlmConfig(
        enabled=bool(values.get("enabled", False)),
        enabled_handlers=list(values.get("enabled_handlers", [])),
        timeout_ms=int(values.get("timeout_ms", 1500)),
        max_tokens_per_decision=int(values.get("max_tokens_per_decision", 1200)),
        confidence_threshold=float(values.get("confidence_threshold", 0.4)),
        telemetry_enabled=bool(values.get("telemetry_enabled", True)),
        telemetry_path=str(values.get("telemetry_path", "logs/llm_decisions.jsonl")),
    )
