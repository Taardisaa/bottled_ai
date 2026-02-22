"""
Cache implementations for diff-related data.

This module provides cache implementations for:
- Enriched file changes (parsed git diffs)
- Out-of-function change summaries (global scope changes)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..base import BaseCache
from ..utils import compute_hash


class FileChangeCache(BaseCache[Any, str]):
    """
    Cache for enriched file change data (parsed git diffs).

    Cache key: Change ID (from EnrichedFileChange.get_id())
    File format: Serialized EnrichedFileChange object

    Note: This cache fixes a race condition in the original implementation
    by using file locking for all read/write operations.
    """

    cache_type = "file_change"
    default_ttl_days = 21
    shared_by_default = True

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "file_change_cache"

    def compute_cache_key(self, change_id: str) -> str:
        """Use the change ID directly as the cache key."""
        return change_id

    def serialize(self, data: Any) -> Dict[str, Any]:
        """Convert EnrichedFileChange to dict using its to_dict method."""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        return data

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Reconstruct EnrichedFileChange from dict."""
        try:
            from diff_transformer.diff_proto import EnrichedFileChange
            return EnrichedFileChange.from_dict(data)
        except ImportError:
            return data


class OutOfFuncChangeCache(BaseCache[Any, tuple]):
    """
    Cache for out-of-function (global scope) change summaries.

    Cache key: SHA256 hash of (old_path, new_path, prompt_hash, agent_id)
    File format: Serialized CachedGlobalChangeWithSummary object

    This caches LLM responses for analyzing global scope changes like
    macro definitions, struct declarations, and global variables.
    """

    cache_type = "out_of_func_change"
    default_ttl_days = 21
    shared_by_default = True

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "out_change_cache"

    def compute_cache_key(
        self,
        prompt: str,
        out_change: Any,  # OutOfFunctionChange or GlobalChangeWithSummary
        agent_id: str
    ) -> str:
        """
        Compute cache key from prompt, change info, and agent ID.

        The key is a SHA256 hash of the combined identifier string.
        """
        prompt_hash = compute_hash(prompt)
        old_path = getattr(out_change, 'old_path', '')
        new_path = getattr(out_change, 'new_path', '')
        id_str = f"{old_path}##{new_path}##{prompt_hash}##{agent_id}"
        return compute_hash(id_str)

    def serialize(self, data: Any) -> Dict[str, Any]:
        """Convert CachedGlobalChangeWithSummary to dict."""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        return data

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Reconstruct CachedGlobalChangeWithSummary from dict."""
        try:
            from checker_agent.checker_proto import CachedGlobalChangeWithSummary
            return CachedGlobalChangeWithSummary.from_dict(data)
        except ImportError:
            return data
