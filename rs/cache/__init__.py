"""Unified cache package for this repository."""

from .base import BaseCache
from .config import CacheConfig
from .registry import CacheRegistry, _auto_register
from rs.utils.common_utils import do_hash
from rs.utils.file_lock_utils import (
    read_json_from_zip_with_lock,
    read_json_with_lock,
    write_json_as_zip_with_lock,
    write_json_with_lock,
)
from rs.utils.file_utils import (
    format_id,
    is_file_valid,
    safe_remove,
)

from .types.llm import LlmQueryCache

def ensure_cache_registry():
    """Lazily initialize CacheRegistry if not already initialized."""
    if not CacheRegistry.is_initialized():
        from rs.utils.config import config as app_config

        CacheRegistry.initialize(CacheConfig.from_rs_config(app_config))


__all__ = [
    # Core classes
    "BaseCache",
    "CacheConfig",
    "CacheRegistry",
    "ensure_cache_registry",
    # Utility functions
    "do_hash",
    "format_id",
    "is_file_valid",
    "read_json_with_lock",
    "write_json_with_lock",
    "read_json_from_zip_with_lock",
    "write_json_as_zip_with_lock",
    "safe_remove",
    "LlmQueryCache",
]

# Auto-register all built-in cache types
_auto_register()
