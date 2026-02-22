# Unified Cache System

This directory contains a new unified caching infrastructure for CHPC.

> **NOTE:** This module is still under development and is **NOT yet connected to the main pipeline**. The existing caching logic throughout the codebase has not been migrated to use this system yet.

## Overview

The cache system provides:
- Abstract `BaseCache` class for implementing cache types
- `CacheRegistry` for centralized access to cache instances
- `CacheConfig` for cache configuration management
- 16 built-in cache implementations for various data types

## Available Cache Types

| Cache Type | Description | Default TTL | Shared by Default |
|------------|-------------|-------------|-------------------|
| `nvd_query` | NVD API query results | 21 days | Yes |
| `osv_query` | OSV API query results | 21 days | Yes |
| `cve_entry` | CVE vulnerability entries | 21 days | Yes |
| `osv_entry` | OSV vulnerability entries | 21 days | Yes |
| `patch_info` | Patch information with entries | 21 days | Yes |
| `generic_entry` | Generic vulnerability entries | 21 days | Yes |
| `cpe_match` | CPE match results | 21 days | Yes |
| `file_change` | Enriched file changes (parsed diffs) | None | Yes |
| `out_of_func_change` | Out-of-function change summaries | None | Yes |
| `agent_check_result` | Patch verification results | None | No |
| `di_agent_state` | Decompile interactor agent state | None | No |
| `ro_agent_state` | Repo operator agent state | None | No |
| `decompile_result` | Decompilation pseudocode | None | Yes |
| `llm_query` | LLM query/response pairs | None | No |
| `git_api` | Git API responses | 30 days | Yes |
| `web_handler` | Web scraping results | 21 days | Yes |

## Planned Usage

Once integrated, the cache system will be used as follows:

```python
# Initialize the cache system (typically done once at startup)
from config import config
from cache import CacheRegistry, CacheConfig

cache_config = CacheConfig.from_chpc_config(config)
CacheRegistry.initialize(cache_config)

# Get a cache instance and use it
nvd_cache = CacheRegistry.get("nvd_query")
data = nvd_cache.load("CVE-2024-1234")
if data is None:
    data = fetch_from_api()
    nvd_cache.store(data, "CVE-2024-1234")

# Or use type-specific imports
from cache.types import NvdQueryCache, CveEntryCache
```

## Module Structure

```
cache/
├── __init__.py      # Module exports and auto-registration
├── base.py          # Abstract BaseCache class
├── config.py        # CacheConfig for configuration management
├── registry.py      # CacheRegistry for centralized access
├── utils.py         # Utility functions (hashing, file I/O)
└── types/           # Built-in cache implementations
    ├── vulnerability.py  # NVD, OSV, CVE, patch caches
    ├── diff.py           # File change caches
    ├── agent.py          # Agent state caches
    ├── decompile.py      # Decompilation result cache
    ├── llm.py            # LLM query cache
    ├── git_api.py        # Git API response cache
    └── web.py            # Web handler cache
```

## TODO

- [ ] Migrate existing cache usage throughout the codebase
- [ ] Add migration scripts for existing cache files
- [ ] Add cache statistics and monitoring
- [ ] Add cache warming utilities
