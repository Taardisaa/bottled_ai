"""
Unified caching system for CHPC.

This module provides a centralized caching infrastructure with:
- Abstract BaseCache class for implementing new cache types
- CacheRegistry for centralized access to cache instances
- CacheConfig for cache configuration management
- 16 built-in cache implementations for various data types

Usage:
    # Initialize the cache system (typically done once at startup)
    from config import config
    from cache import CacheRegistry, CacheConfig

    cache_config = CacheConfig.from_chpc_config(config)
    CacheRegistry.initialize(cache_config)

    # Get a cache instance and use it
    from cache import CacheRegistry

    nvd_cache = CacheRegistry.get("nvd_query")
    data = nvd_cache.load("CVE-2024-1234")
    if data is None:
        data = fetch_from_api()
        nvd_cache.store(data, "CVE-2024-1234")

    # Or use type-specific imports
    from cache.types import NvdQueryCache, CveEntryCache

Available cache types:
    - nvd_query: NVD API query results
    - osv_query: OSV API query results
    - cve_entry: CVE vulnerability entries
    - osv_entry: OSV vulnerability entries
    - patch_info: Patch information with entries
    - generic_entry: Generic vulnerability entries
    - cpe_match: CPE match results
    - file_change: Enriched file changes (parsed diffs)
    - out_of_func_change: Out-of-function change summaries
    - agent_check_result: Patch verification results
    - di_agent_state: Decompile interactor agent state
    - ro_agent_state: Repo operator agent state
    - fpc_agent_state: Final patch checker agent state
    - decompile_result: Decompilation pseudocode
    - ida_index: IDA pre-indexing results (function metadata, call graphs)
    - ida_database: IDA database files (.i64/.idb) for session reuse
    - llm_query: LLM query/response pairs
    - git_api: Git API responses
    - web_handler: Web scraping results
"""

from .base import BaseCache
from .config import CacheConfig
from .registry import CacheRegistry, _auto_register
from .utils import (
    compute_hash,
    format_cache_id,
    is_file_valid,
    read_json_with_lock,
    write_json_with_lock,
    read_json_from_zip_with_lock,
    write_json_as_zip_with_lock,
    safe_remove,
)

# Import all cache types for convenience
from .types import (
    # Vulnerability caches
    NvdQueryCache,
    OsvQueryCache,
    CveEntryCache,
    OsvEntryCache,
    PatchInfoCache,
    GenericEntryCache,
    CpeMatchCache,
    # Diff caches
    FileChangeCache,
    OutOfFuncChangeCache,
    # Agent caches
    AgentCheckResultCache,
    BranchTagCache,
    DiAgentStateCache,
    RoAgentStateCache,
    FpcAgentStateCache,
    # Other caches
    DecompileResultCache,
    IDAIndexCache,
    IDADatabaseCache,
    LlmQueryCache,
    GitApiCache,
    WebHandlerCache,
)

def ensure_cache_registry():
    """Lazily initialize CacheRegistry if not already initialized."""
    if not CacheRegistry.is_initialized():
        from config import config as chpc_config
        CacheRegistry.initialize(CacheConfig.from_chpc_config(chpc_config))


__all__ = [
    # Core classes
    "BaseCache",
    "CacheConfig",
    "CacheRegistry",
    "ensure_cache_registry",
    # Utility functions
    "compute_hash",
    "format_cache_id",
    "is_file_valid",
    "read_json_with_lock",
    "write_json_with_lock",
    "read_json_from_zip_with_lock",
    "write_json_as_zip_with_lock",
    "safe_remove",
    # Vulnerability caches
    "NvdQueryCache",
    "OsvQueryCache",
    "CveEntryCache",
    "OsvEntryCache",
    "PatchInfoCache",
    "GenericEntryCache",
    "CpeMatchCache",
    # Diff caches
    "FileChangeCache",
    "OutOfFuncChangeCache",
    # Agent caches
    "AgentCheckResultCache",
    "BranchTagCache",
    "DiAgentStateCache",
    "RoAgentStateCache",
    "FpcAgentStateCache",
    # Other caches
    "DecompileResultCache",
    "IDAIndexCache",
    "IDADatabaseCache",
    "LlmQueryCache",
    "GitApiCache",
    "WebHandlerCache",
]

# Auto-register all built-in cache types
_auto_register()
