from __future__ import annotations

from dataclasses import dataclass, field
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
    confidence_threshold: float = 0.4
    telemetry_enabled: bool = True
    telemetry_path: str = "logs/llm_decisions.jsonl"
    graph_trace_enabled: bool = True
    graph_trace_path: str = "logs/ai_player_graph.jsonl"
    ai_player_graph_enabled: bool = False
    use_deterministic_battle: bool = False
    langmem_enabled: bool = False
    langmem_sqlite_path: str = "dataset/langmem/memory.sqlite3"
    langmem_embeddings_base_url: str = ""
    langmem_embeddings_api_key: str = ""
    langmem_embeddings_model: str = "BAAI/bge-small-en-v1.5"
    langmem_top_k: int = 3
    langmem_reflection_batch_size: int = 5
    langmem_inject_episodic_memories: bool = True
    langmem_inject_semantic_memories: bool = True
    langmem_max_semantic_memories_per_namespace: int = 50
    langmem_max_retrospective_memories: int = 50
    langmem_min_record_confidence: float = 0.35
    langmem_min_similarity_score: float = 0.65
    langmem_max_reflection_workers: int = 3
    langmem_fail_fast_init: bool = False
    langmem_importance_scoring_enabled: bool = False
    langmem_importance_default: float = 5.0
    langmem_composite_weight_similarity: float = 1.0
    langmem_composite_weight_importance: float = 1.0
    langmem_composite_weight_recency: float = 0.5
    langmem_recency_decay_hours: float = 168.0
    battle_potion_allowed_room_types: List[str] = field(
        default_factory=lambda: ["MonsterRoomElite", "MonsterRoomBoss"]
    )

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
    confidence_default = float(values.get("confidence_threshold", 0.4))
    telemetry_enabled_default = bool(values.get("telemetry_enabled", True))
    telemetry_path_default = str(values.get("telemetry_path", "logs/llm_decisions.jsonl"))
    graph_trace_enabled_default = bool(values.get("graph_trace_enabled", True))
    graph_trace_path_default = str(values.get("graph_trace_path", "logs/ai_player_graph.jsonl"))
    ai_player_graph_enabled_default = bool(values.get("ai_player_graph_enabled", False))
    use_deterministic_battle_default = bool(values.get("use_deterministic_battle", False))
    langmem_values = values.get("langmem", {})
    langmem_enabled_default = bool(langmem_values.get("enabled", False))
    langmem_sqlite_path_default = str(langmem_values.get("sqlite_path", "dataset/langmem/memory.sqlite3"))
    langmem_embeddings_base_url_default = str(langmem_values.get("embeddings_base_url", ""))
    langmem_embeddings_api_key_default = str(langmem_values.get("embeddings_api_key", ""))
    langmem_embeddings_model_default = str(langmem_values.get("embeddings_model", "BAAI/bge-small-en-v1.5"))
    langmem_top_k_default = int(langmem_values.get("top_k", 3))
    langmem_reflection_batch_size_default = int(langmem_values.get("reflection_batch_size", 5))
    langmem_inject_episodic_memories_default = bool(
        langmem_values.get("inject_episodic_memories", True)
    )
    langmem_inject_semantic_memories_default = bool(
        langmem_values.get("inject_semantic_memories", True)
    )
    langmem_max_semantic_memories_per_namespace_default = int(
        langmem_values.get("max_semantic_memories_per_namespace", 50)
    )
    langmem_max_retrospective_memories_default = int(langmem_values.get("max_retrospective_memories", 50))
    langmem_min_record_confidence_default = float(langmem_values.get("min_record_confidence", 0.35))
    langmem_min_similarity_score_default = float(langmem_values.get("min_similarity_score", 0.65))
    langmem_max_reflection_workers_default = int(langmem_values.get("max_reflection_workers", 3))
    langmem_fail_fast_init_default = bool(langmem_values.get("fail_fast_init", False))
    langmem_importance_scoring_enabled_default = bool(
        langmem_values.get("importance_scoring_enabled", False)
    )
    langmem_importance_default_default = float(langmem_values.get("importance_default", 5.0))
    langmem_composite_weight_similarity_default = float(
        langmem_values.get("composite_weight_similarity", 1.0)
    )
    langmem_composite_weight_importance_default = float(
        langmem_values.get("composite_weight_importance", 1.0)
    )
    langmem_composite_weight_recency_default = float(
        langmem_values.get("composite_weight_recency", 0.5)
    )
    langmem_recency_decay_hours_default = float(
        langmem_values.get("recency_decay_hours", 168.0)
    )
    battle_potion_allowed_room_types_default = list(
        values.get("battle_potion_allowed_room_types", ["MonsterRoomElite", "MonsterRoomBoss"])
    )

    enabled = _parse_bool(os.environ.get("LLM_ENABLED"), enabled_default)
    enabled_handlers = _parse_handlers(os.environ.get("LLM_ENABLED_HANDLERS"), enabled_handlers_default)
    timeout_ms = int(os.environ.get("LLM_TIMEOUT_MS", str(timeout_default)))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", str(max_retries_default)))
    confidence_threshold = float(os.environ.get("LLM_CONFIDENCE_THRESHOLD", str(confidence_default)))
    telemetry_enabled = _parse_bool(os.environ.get("LLM_TELEMETRY_ENABLED"), telemetry_enabled_default)
    telemetry_path = os.environ.get("LLM_TELEMETRY_PATH", telemetry_path_default)
    graph_trace_enabled = _parse_bool(
        os.environ.get("AI_PLAYER_GRAPH_TRACE_ENABLED"),
        graph_trace_enabled_default,
    )
    graph_trace_path = os.environ.get("AI_PLAYER_GRAPH_TRACE_PATH", graph_trace_path_default)
    ai_player_graph_enabled = _parse_bool(
        os.environ.get("AI_PLAYER_GRAPH_ENABLED"),
        ai_player_graph_enabled_default,
    )
    use_deterministic_battle = _parse_bool(
        os.environ.get("USE_DETERMINISTIC_BATTLE"),
        use_deterministic_battle_default,
    )
    langmem_enabled = _parse_bool(os.environ.get("LANGMEM_ENABLED"), langmem_enabled_default)
    langmem_sqlite_path = os.environ.get("LANGMEM_SQLITE_PATH", langmem_sqlite_path_default)
    langmem_embeddings_base_url = os.environ.get(
        "LANGMEM_EMBEDDINGS_BASE_URL",
        langmem_embeddings_base_url_default,
    )
    langmem_embeddings_api_key = os.environ.get(
        "LANGMEM_EMBEDDINGS_API_KEY",
        langmem_embeddings_api_key_default,
    )
    langmem_embeddings_model = os.environ.get(
        "LANGMEM_EMBEDDINGS_MODEL",
        langmem_embeddings_model_default,
    )
    langmem_top_k = int(os.environ.get("LANGMEM_TOP_K", str(langmem_top_k_default)))
    langmem_reflection_batch_size = int(
        os.environ.get("LANGMEM_REFLECTION_BATCH_SIZE", str(langmem_reflection_batch_size_default))
    )
    langmem_inject_episodic_memories = _parse_bool(
        os.environ.get("LANGMEM_INJECT_EPISODIC_MEMORIES"),
        langmem_inject_episodic_memories_default,
    )
    langmem_inject_semantic_memories = _parse_bool(
        os.environ.get("LANGMEM_INJECT_SEMANTIC_MEMORIES"),
        langmem_inject_semantic_memories_default,
    )
    langmem_max_semantic_memories_per_namespace = int(
        os.environ.get(
            "LANGMEM_MAX_SEMANTIC_MEMORIES_PER_NAMESPACE",
            str(langmem_max_semantic_memories_per_namespace_default),
        )
    )
    langmem_max_retrospective_memories = int(
        os.environ.get("LANGMEM_MAX_RETROSPECTIVE_MEMORIES", str(langmem_max_retrospective_memories_default))
    )
    langmem_min_record_confidence = float(
        os.environ.get("LANGMEM_MIN_RECORD_CONFIDENCE", str(langmem_min_record_confidence_default))
    )
    langmem_min_similarity_score = float(
        os.environ.get("LANGMEM_MIN_SIMILARITY_SCORE", str(langmem_min_similarity_score_default))
    )
    langmem_max_reflection_workers = int(
        os.environ.get("LANGMEM_MAX_REFLECTION_WORKERS", str(langmem_max_reflection_workers_default))
    )
    langmem_fail_fast_init = _parse_bool(
        os.environ.get("LANGMEM_FAIL_FAST_INIT"),
        langmem_fail_fast_init_default,
    )
    langmem_importance_scoring_enabled = _parse_bool(
        os.environ.get("LANGMEM_IMPORTANCE_SCORING_ENABLED"),
        langmem_importance_scoring_enabled_default,
    )
    langmem_importance_default = float(
        os.environ.get("LANGMEM_IMPORTANCE_DEFAULT", str(langmem_importance_default_default))
    )
    langmem_composite_weight_similarity = float(
        os.environ.get("LANGMEM_COMPOSITE_WEIGHT_SIMILARITY", str(langmem_composite_weight_similarity_default))
    )
    langmem_composite_weight_importance = float(
        os.environ.get("LANGMEM_COMPOSITE_WEIGHT_IMPORTANCE", str(langmem_composite_weight_importance_default))
    )
    langmem_composite_weight_recency = float(
        os.environ.get("LANGMEM_COMPOSITE_WEIGHT_RECENCY", str(langmem_composite_weight_recency_default))
    )
    langmem_recency_decay_hours = float(
        os.environ.get("LANGMEM_RECENCY_DECAY_HOURS", str(langmem_recency_decay_hours_default))
    )

    return LlmConfig(
        enabled=enabled,
        enabled_handlers=enabled_handlers,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        confidence_threshold=confidence_threshold,
        telemetry_enabled=telemetry_enabled,
        telemetry_path=telemetry_path,
        graph_trace_enabled=graph_trace_enabled,
        graph_trace_path=graph_trace_path,
        ai_player_graph_enabled=ai_player_graph_enabled,
        use_deterministic_battle=use_deterministic_battle,
        langmem_enabled=langmem_enabled,
        langmem_sqlite_path=langmem_sqlite_path,
        langmem_embeddings_base_url=langmem_embeddings_base_url,
        langmem_embeddings_api_key=langmem_embeddings_api_key,
        langmem_embeddings_model=langmem_embeddings_model,
        langmem_top_k=langmem_top_k,
        langmem_reflection_batch_size=langmem_reflection_batch_size,
        langmem_inject_episodic_memories=langmem_inject_episodic_memories,
        langmem_inject_semantic_memories=langmem_inject_semantic_memories,
        langmem_max_semantic_memories_per_namespace=langmem_max_semantic_memories_per_namespace,
        langmem_max_retrospective_memories=langmem_max_retrospective_memories,
        langmem_min_record_confidence=langmem_min_record_confidence,
        langmem_min_similarity_score=langmem_min_similarity_score,
        langmem_max_reflection_workers=langmem_max_reflection_workers,
        langmem_fail_fast_init=langmem_fail_fast_init,
        langmem_importance_scoring_enabled=langmem_importance_scoring_enabled,
        langmem_importance_default=langmem_importance_default,
        langmem_composite_weight_similarity=langmem_composite_weight_similarity,
        langmem_composite_weight_importance=langmem_composite_weight_importance,
        langmem_composite_weight_recency=langmem_composite_weight_recency,
        langmem_recency_decay_hours=langmem_recency_decay_hours,
        battle_potion_allowed_room_types=battle_potion_allowed_room_types_default,
    )
