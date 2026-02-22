"""
Cache configuration management.

This module provides the CacheConfig dataclass that encapsulates all cache-related
settings, decoupling the cache system from the main application config.
"""

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheConfig:
    """
    Unified cache configuration.

    This dataclass holds all settings needed by the cache system, including:
    - Profile name for cache isolation
    - Base dataset directory
    - Per-cache enable/disable toggles
    - Per-cache sharing settings (shared vs profile-isolated)
    - TTL overrides per cache type
    """

    # Profile settings
    profile: str = "default"
    dataset_dir: Path = field(default_factory=lambda: Path("."))

    # Cache enable/disable toggles (from [LoadCacheOptions] and [StoreCacheOptions] sections)
    load_enabled_caches: Dict[str, bool] = field(default_factory=dict)
    store_enabled_caches: Dict[str, bool] = field(default_factory=dict)

    # Cache sharing settings (from [ShareCache] section)
    # True = shared across profiles, False = isolated per profile
    shared_caches: Dict[str, bool] = field(default_factory=dict)

    # TTL overrides per cache type (days, None = use default)
    ttl_overrides: Dict[str, Optional[int]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Handle backward compatibility for old enabled_caches parameter.

        If load_enabled_caches is empty and this was created with the old
        enabled_caches parameter (in constructor), populate both load and store.
        """
        # No special handling needed here - backward compat is in from_config_parser
        pass

    @property
    def enabled_caches(self) -> Dict[str, bool]:
        """
        Backward compatibility property.

        Returns load_enabled_caches for compatibility with code that expects
        the old single enabled_caches dictionary.
        """
        return self.load_enabled_caches

    # ShareCache key mappings (cache_type -> ShareCache config key)
    _share_cache_key_map: Dict[str, str] = field(default_factory=lambda: {
        "git_api": "git_api_cache",
        "nvd_query": "nvd_query_cache",
        "osv_query": "osv_query_cache",
        "osv_entry": "osv_query_cache",
        "cve_entry": "nvd_query_cache",
        "patch_info": "nvd_query_cache",
        "generic_entry": "nvd_query_cache",
        "cpe_match": "nvd_query_cache",
        "file_change": "file_change_cache",
        "out_of_func_change": "out_change_cache",
        "decompile_result": "decompile_output",
        "ida_index": "decompile_output",
        "ida_database": "decompile_output",
        "agent_check_result": "agent_cache",
        "llm_query": "llm_query_cache",
        "di_agent_state": "di_agent_cache",
        "ro_agent_state": "ro_agent_cache",
        "fpc_agent_state": "fpc_agent_cache",
        "web_handler": "agent_cache",
        "branch_tag": "agent_cache",
    })

    @classmethod
    def from_config_parser(
        cls,
        config: configparser.ConfigParser,
        profile: str = "default",
        dataset_dir: Optional[Path] = None
    ) -> 'CacheConfig':
        """
        Create CacheConfig from a ConfigParser instance.

        This allows the cache system to work with the existing config.ini format.

        Args:
            config: ConfigParser instance with configuration loaded
            profile: Configuration profile name for cache isolation
            dataset_dir: Base dataset directory path

        Returns:
            CacheConfig instance
        """
        # Extract load-enabled caches from [LoadCacheOptions] or [CacheOptions] (backward compat)
        load_section = "LoadCacheOptions" if config.has_section("LoadCacheOptions") else "CacheOptions"
        load_enabled_caches = {}
        if config.has_section(load_section):
            for option in config.options(load_section):
                load_enabled_caches[option] = config.getboolean(
                    load_section, option, fallback=True
                )

        # Extract store-enabled caches from [StoreCacheOptions] (defaults to all true)
        store_enabled_caches = {}
        if config.has_section("StoreCacheOptions"):
            for option in config.options("StoreCacheOptions"):
                store_enabled_caches[option] = config.getboolean(
                    "StoreCacheOptions", option, fallback=True
                )

        # Extract sharing settings from [ShareCache] section
        shared_caches = {}
        if config.has_section("ShareCache"):
            for option in config.options("ShareCache"):
                shared_caches[option] = config.getboolean(
                    "ShareCache", option, fallback=True
                )

        return cls(
            profile=profile,
            dataset_dir=dataset_dir or Path("."),
            load_enabled_caches=load_enabled_caches,
            store_enabled_caches=store_enabled_caches,
            shared_caches=shared_caches,
        )

    @classmethod
    def from_chpc_config(cls, chpc_config: Any) -> 'CacheConfig':
        """
        Create CacheConfig from an existing CHPCVulnCheckerConfig instance.

        This provides seamless integration with the existing configuration system.

        Args:
            chpc_config: The main CHPC configuration object

        Returns:
            CacheConfig instance
        """
        # Build load_enabled_caches from LoadCacheOptions
        load_enabled_caches = {}
        if hasattr(chpc_config, 'load_cache_options') and chpc_config.load_cache_options:
            for attr in dir(chpc_config.load_cache_options):
                if not attr.startswith('_') and attr != 'get':
                    value = getattr(chpc_config.load_cache_options, attr, None)
                    if isinstance(value, bool):
                        load_enabled_caches[attr] = value

        # Build store_enabled_caches from StoreCacheOptions
        store_enabled_caches = {}
        if hasattr(chpc_config, 'store_cache_options') and chpc_config.store_cache_options:
            for attr in dir(chpc_config.store_cache_options):
                if not attr.startswith('_') and attr != 'get':
                    value = getattr(chpc_config.store_cache_options, attr, None)
                    if isinstance(value, bool):
                        store_enabled_caches[attr] = value

        # Build shared_caches from ShareCache
        shared_caches = {}
        if hasattr(chpc_config, 'share_cache') and chpc_config.share_cache:
            for attr in dir(chpc_config.share_cache):
                if not attr.startswith('_'):
                    value = getattr(chpc_config.share_cache, attr, None)
                    if isinstance(value, bool):
                        shared_caches[attr] = value

        dataset_dir = Path(chpc_config.dataset_dir_path) if chpc_config.dataset_dir_path else Path(".")

        return cls(
            profile=chpc_config.config_profile,
            dataset_dir=dataset_dir,
            load_enabled_caches=load_enabled_caches,
            store_enabled_caches=store_enabled_caches,
            shared_caches=shared_caches,
        )

    @classmethod
    def from_rs_config(cls, app_config: Any) -> "CacheConfig":
        profile = str(getattr(app_config, "config_profile", "default"))
        dataset_dir_path = str(getattr(app_config, "dataset_dir_path", "."))
        dataset_dir = Path(dataset_dir_path) if dataset_dir_path else Path(".")

        load_enabled_caches: Dict[str, bool] = {}
        store_enabled_caches: Dict[str, bool] = {}

        load_options = getattr(app_config, "load_cache_options", {})
        if isinstance(load_options, dict):
            for key, value in load_options.items():
                load_enabled_caches[str(key)] = bool(value)

        store_options = getattr(app_config, "store_cache_options", {})
        if isinstance(store_options, dict):
            for key, value in store_options.items():
                store_enabled_caches[str(key)] = bool(value)

        share_cache_obj: Any = getattr(app_config, "share_cache", {})
        shared_caches: Dict[str, bool] = {}
        if isinstance(share_cache_obj, dict):
            for key, value in share_cache_obj.items():
                shared_caches[str(key)] = bool(value)

        return cls(
            profile=profile,
            dataset_dir=dataset_dir,
            load_enabled_caches=load_enabled_caches,
            store_enabled_caches=store_enabled_caches,
            shared_caches=shared_caches,
        )

    def is_cache_load_enabled(self, cache_type: str) -> bool:
        """
        Check if loading from cache is enabled for a cache type.

        Args:
            cache_type: Cache type identifier (e.g., "nvd_query", "llm_query")

        Returns:
            True if loading from cache is enabled, False otherwise (default: True)
        """
        normalized = cache_type.replace("-", "_").lower()
        return self.load_enabled_caches.get(normalized, True)  # Default to enabled

    def is_cache_store_enabled(self, cache_type: str) -> bool:
        """
        Check if storing to cache is enabled for a cache type.

        Args:
            cache_type: Cache type identifier (e.g., "nvd_query", "llm_query")

        Returns:
            True if storing to cache is enabled, False otherwise (default: True)
        """
        normalized = cache_type.replace("-", "_").lower()
        return self.store_enabled_caches.get(normalized, True)  # Default to enabled

    def is_cache_enabled(self, cache_type: str) -> bool:
        """
        Check if a cache type is enabled (backward compatibility).

        DEPRECATED: Use is_cache_load_enabled() or is_cache_store_enabled() instead.
        Returns True only if BOTH load and store are enabled.

        Args:
            cache_type: Cache type identifier (e.g., "nvd_query", "llm_query")

        Returns:
            True if both load and store are enabled, False otherwise
        """
        return self.is_cache_load_enabled(cache_type) and self.is_cache_store_enabled(cache_type)

    def is_cache_shared(self, cache_type: str, default: bool = True) -> bool:
        """
        Check if a cache type should be shared across profiles.

        Args:
            cache_type: Cache type identifier
            default: Default value if not configured

        Returns:
            True if cache should be shared, False if profile-isolated
        """
        normalized = cache_type.replace("-", "_").lower()
        share_cache_key = self._share_cache_key_map.get(normalized, normalized)
        return self.shared_caches.get(share_cache_key, default)

    def get_ttl(self, cache_type: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get TTL for a cache type.

        Args:
            cache_type: Cache type identifier
            default: Default TTL if not configured

        Returns:
            TTL in days, or None for no expiration
        """
        normalized = cache_type.replace("-", "_").lower()
        return self.ttl_overrides.get(normalized, default)

    def get_cache_dir(self, cache_name: str, shared: bool = True) -> Path:
        """
        Get the directory path for a specific cache.

        Args:
            cache_name: Name of the cache directory (e.g., "nvd_query_cache")
            shared: If True, returns shared path; if False, includes profile subdirectory

        Returns:
            Path to cache directory
        """
        base_path = self.dataset_dir / cache_name
        if shared:
            return base_path
        return base_path / self.profile
