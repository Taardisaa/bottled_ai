"""
Cache registry for centralized access to cache instances.

This module provides the CacheRegistry class that manages all cache type
registrations and provides factory access to cache instances.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import BaseCache
from .config import CacheConfig

from utils.logging_utils import setup_logger_stdout

logger = setup_logger_stdout("cache.registry")


class CacheRegistry:
    """
    Registry for cache types - provides factory pattern and singleton access.

    The CacheRegistry manages cache type registrations and provides a centralized
    way to access cache instances. It must be initialized with a CacheConfig
    before cache instances can be retrieved.

    Usage:
        # Initialize the registry (typically done once at startup)
        from config import config
        from cache import CacheRegistry, CacheConfig

        cache_config = CacheConfig.from_chpc_config(config)
        CacheRegistry.initialize(cache_config)

        # Get a cache instance
        nvd_cache = CacheRegistry.get("nvd_query")
        data = nvd_cache.load("CVE-2024-1234")

        # Register a custom cache type
        CacheRegistry.register("my_cache", MyCacheClass)

        # List all registered cache types
        types = CacheRegistry.get_all_types()
    """

    _instance: Optional['CacheRegistry'] = None
    _caches: Dict[str, BaseCache] = {}
    _cache_classes: Dict[str, Type[BaseCache]] = {}
    _config: Optional[CacheConfig] = None
    _initialized: bool = False

    def __new__(cls) -> 'CacheRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: CacheConfig) -> 'CacheRegistry':
        """
        Initialize the registry with configuration.

        This must be called before any cache instances can be retrieved.
        Calling this again will reinitialize with the new config.

        Args:
            config: CacheConfig instance with cache settings

        Returns:
            The CacheRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()

        cls._config = config
        cls._caches = {}  # Clear cached instances to re-create with new config
        cls._initialized = True

        logger.info(f"CacheRegistry initialized with profile: {config.profile}")
        return cls._instance

    @classmethod
    def register(cls, cache_type: str, cache_class: Type[BaseCache]) -> None:
        """
        Register a cache class for a given type.

        This allows new cache types to be added to the registry. The cache_type
        should match the cache_type class attribute of the cache class.

        Args:
            cache_type: Unique identifier for this cache type
            cache_class: The cache class to register
        """
        cls._cache_classes[cache_type] = cache_class
        logger.debug(f"Registered cache type: {cache_type}")

    @classmethod
    def get(cls, cache_type: str) -> BaseCache:
        """
        Get a cache instance by type.

        Cache instances are created lazily and cached for reuse.

        Args:
            cache_type: The cache type identifier

        Returns:
            Cache instance for the given type

        Raises:
            ValueError: If cache type is not registered
            RuntimeError: If registry not initialized
        """
        if not cls._initialized or cls._config is None:
            raise RuntimeError(
                "CacheRegistry not initialized. Call CacheRegistry.initialize(config) first."
            )

        if cache_type not in cls._caches:
            if cache_type not in cls._cache_classes:
                raise ValueError(
                    f"Unknown cache type: {cache_type}. "
                    f"Available types: {list(cls._cache_classes.keys())}"
                )

            cache_class = cls._cache_classes[cache_type]
            cls._caches[cache_type] = cache_class(cls._config)

        return cls._caches[cache_type]

    @classmethod
    def get_all_types(cls) -> List[str]:
        """
        Get list of all registered cache types.

        Returns:
            List of cache type identifiers
        """
        return list(cls._cache_classes.keys())

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the registry has been initialized.

        Returns:
            True if initialized with a config
        """
        return cls._initialized and cls._config is not None

    @classmethod
    def get_config(cls) -> Optional[CacheConfig]:
        """
        Get the current configuration.

        Returns:
            Current CacheConfig or None if not initialized
        """
        return cls._config

    @classmethod
    def clear(cls) -> None:
        """
        Clear all cached instances.

        This forces cache instances to be recreated on next access.
        Useful for testing or when configuration changes.
        """
        cls._caches = {}

    @classmethod
    def reset(cls) -> None:
        """
        Fully reset the registry.

        Clears all cached instances, registered classes, and configuration.
        Use with caution - primarily for testing.
        """
        cls._caches = {}
        cls._cache_classes = {}
        cls._config = None
        cls._initialized = False
        cls._instance = None


def _auto_register() -> None:
    """
    Auto-register all built-in cache types.

    This function is called during module initialization to register
    all the standard cache implementations.
    """
    # Import cache type implementations
    try:
        from .types.vulnerability import (
            NvdQueryCache,
            OsvQueryCache,
            CveEntryCache,
            OsvEntryCache,
            PatchInfoCache,
            GenericEntryCache,
            CpeMatchCache,
        )
        from .types.diff import FileChangeCache, OutOfFuncChangeCache
        from .types.agent import (
            AgentCheckResultCache,
            BranchTagCache,
            DiAgentStateCache,
            RoAgentStateCache,
            FpcAgentStateCache,
        )
        from .types.decompile import DecompileResultCache, IDAIndexCache, IDADatabaseCache
        from .types.llm import LlmQueryCache
        from .types.git_api import GitApiCache
        from .types.web import WebHandlerCache

        # Register all cache types
        for cache_class in [
            NvdQueryCache,
            OsvQueryCache,
            CveEntryCache,
            OsvEntryCache,
            PatchInfoCache,
            GenericEntryCache,
            CpeMatchCache,
            FileChangeCache,
            OutOfFuncChangeCache,
            AgentCheckResultCache,
            BranchTagCache,
            DiAgentStateCache,
            RoAgentStateCache,
            FpcAgentStateCache,
            DecompileResultCache,
            IDAIndexCache,
            IDADatabaseCache,
            LlmQueryCache,
            GitApiCache,
            WebHandlerCache,
        ]:
            CacheRegistry.register(cache_class.cache_type, cache_class)

        logger.debug(f"Auto-registered {len(CacheRegistry.get_all_types())} cache types")
    except ImportError as e:
        # Cache types not yet implemented - this is fine during initial setup
        logger.debug(f"Cache types not fully available yet: {e}")
