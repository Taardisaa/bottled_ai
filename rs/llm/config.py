from __future__ import annotations

from dataclasses import dataclass
import os
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
    max_retries: int = 1
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
        if self.max_retries < 0:
            self.max_retries = 0

    def is_handler_enabled(self, handler_name: str) -> bool:
        if not self.enabled:
            return False
        if not self.enabled_handlers:
            return True
        return handler_name in self.enabled_handlers


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_handlers(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


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

    enabled_default = bool(values.get("enabled", False))
    enabled_handlers_default = list(values.get("enabled_handlers", []))
    timeout_default = int(values.get("timeout_ms", -1))
    max_retries_default = int(values.get("max_retries", 1))
    max_tokens_default = int(values.get("max_tokens_per_decision", 1200))
    confidence_default = float(values.get("confidence_threshold", 0.4))
    telemetry_enabled_default = bool(values.get("telemetry_enabled", True))
    telemetry_path_default = str(values.get("telemetry_path", "logs/llm_decisions.jsonl"))

    enabled = _parse_bool(os.environ.get("LLM_ENABLED"), enabled_default)
    enabled_handlers = _parse_handlers(os.environ.get("LLM_ENABLED_HANDLERS"), enabled_handlers_default)
    timeout_ms = int(os.environ.get("LLM_TIMEOUT_MS", str(timeout_default)))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", str(max_retries_default)))
    max_tokens_per_decision = int(os.environ.get("LLM_MAX_TOKENS_PER_DECISION", str(max_tokens_default)))
    confidence_threshold = float(os.environ.get("LLM_CONFIDENCE_THRESHOLD", str(confidence_default)))
    telemetry_enabled = _parse_bool(os.environ.get("LLM_TELEMETRY_ENABLED"), telemetry_enabled_default)
    telemetry_path = os.environ.get("LLM_TELEMETRY_PATH", telemetry_path_default)

    return LlmConfig(
        enabled=enabled,
        enabled_handlers=enabled_handlers,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        max_tokens_per_decision=max_tokens_per_decision,
        confidence_threshold=confidence_threshold,
        telemetry_enabled=telemetry_enabled,
        telemetry_path=telemetry_path,
    )
