"""
Abstract base class for all cache implementations.

This module provides the BaseCache abstract class that defines the interface
and common functionality for all cache types in the system.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

from .config import CacheConfig
from .utils import (
    is_file_valid,
    read_json_from_zip_with_lock,
    read_json_with_lock,
    safe_remove,
    write_json_as_zip_with_lock,
    write_json_with_lock,
)

# Type variables for cache data and key types
T = TypeVar('T')  # Cache data type
K = TypeVar('K')  # Cache key type (typically str or tuple)


class BaseCache(ABC, Generic[T, K]):
    """
    Abstract base class for all cache implementations.

    This class provides a unified interface for cache operations and handles
    common functionality like TTL validation, profile isolation, and file locking.

    Type Parameters:
        T: The type of data being cached (e.g., PatchInfoWithEntry, CVEEntry)
        K: The type of cache key (e.g., str, tuple of args)

    Subclasses must implement:
        - cache_type: Unique identifier for this cache type
        - default_ttl_days: Default TTL in days (None = never expires)
        - shared_by_default: Whether to share across profiles by default
        - get_base_dir(): Returns base directory for cache files
        - compute_cache_key(*args, **kwargs): Generates cache key from inputs
        - serialize(data: T) -> Dict: Converts data to JSON-serializable dict
        - deserialize(data: Dict) -> T: Reconstructs object from dict

    Example:
        class MyCacheType(BaseCache[MyDataType, str]):
            cache_type = "my_cache"
            default_ttl_days = 21
            shared_by_default = True

            def get_base_dir(self) -> Path:
                return self._config.dataset_dir / "my_cache"

            def compute_cache_key(self, item_id: str) -> str:
                return format_cache_id(item_id)

            def serialize(self, data: MyDataType) -> Dict[str, Any]:
                return data.to_dict()

            def deserialize(self, data: Dict[str, Any]) -> MyDataType:
                return MyDataType.from_dict(data)
    """

    # Class-level configuration (must be set by subclasses)
    cache_type: str = ""  # e.g., "nvd_query", "git_api"
    default_ttl_days: Optional[int] = 21  # None = never expires
    shared_by_default: bool = True  # Profile sharing default
    file_extension: str = ".json"  # Cache file extension
    use_compression: bool = False  # Whether to compress cache files with zip

    def __init__(
        self,
        config: CacheConfig,
        base_dir_override: Optional[Path] = None
    ):
        """
        Initialize cache with configuration.

        Args:
            config: CacheConfig instance with global settings
            base_dir_override: Optional override for base directory
        """
        self._config = config
        self._base_dir_override = base_dir_override
        self._logger = logging.getLogger(f"cache.{self.cache_type}")

    @abstractmethod
    def get_base_dir(self) -> Path:
        """
        Return the base directory for this cache type.

        This should return the directory where cache files are stored,
        without considering profile isolation.

        Returns:
            Base cache directory path
        """
        ...

    @abstractmethod
    def compute_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Compute a unique cache key from the input arguments.

        This method should generate a filesystem-safe cache key that uniquely
        identifies the cached data. Common patterns:
        - SHA256 hash of input parameters
        - Formatted identifier (using format_cache_id)
        - Combination of multiple fields

        Args:
            *args, **kwargs: Input parameters that identify the cached data

        Returns:
            Filesystem-safe cache key string
        """
        ...

    @abstractmethod
    def serialize(self, data: T) -> Optional[Dict[str, Any]]:
        """
        Convert cached data to JSON-serializable dictionary.

        Args:
            data: The data object to serialize

        Returns:
            JSON-serializable dictionary, or None if serialization fails
        """
        ...

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Optional[T]:
        """
        Reconstruct cached data from dictionary.

        Args:
            data: Dictionary loaded from cache file

        Returns:
            Reconstructed data object, or None if deserialization fails
        """
        ...

    # Provided implementations

    def get_cache_dir(self) -> Path:
        """
        Get the final cache directory, considering profile isolation.

        Returns:
            Cache directory path (with or without profile subdirectory)
        """
        base_dir = self._base_dir_override or self.get_base_dir()

        if self._is_profile_isolated():
            return base_dir / self._config.profile
        return base_dir

    def get_cache_path(self, key: str) -> Path:
        """
        Get the canonical (uncompressed) path to a cache file.

        Always returns the .json path regardless of compression setting.
        Compression is a storage detail handled internally by load/store.

        Keys may contain ``/`` separators (e.g. ``namespace/hash``) to
        store entries in subdirectories. Parent directories are created
        automatically.

        Args:
            key: Cache key (from compute_cache_key)

        Returns:
            Full path to cache file (canonical .json path)
        """
        cache_dir = self.get_cache_dir()
        cache_path = cache_dir / f"{key}{self.file_extension}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _get_compressed_path(self, canonical_path: Path) -> Path:
        """Get the compressed (.json.zip) companion path for a canonical cache path."""
        return canonical_path.parent / (canonical_path.name + ".zip")

    def _resolve_actual_path(self, canonical_path: Path) -> Optional[Path]:
        """
        Resolve the actual file path, checking both compressed and uncompressed variants.

        Always checks both variants regardless of use_compression setting,
        ensuring backward compatibility when the compression setting is toggled.

        Args:
            canonical_path: The canonical (.json) path from get_cache_path()

        Returns:
            Path to the existing file, or None if neither exists
        """
        compressed_path = self._get_compressed_path(canonical_path)
        if compressed_path.is_file():
            return compressed_path
        if canonical_path.is_file():
            return canonical_path
        return None

    def load(
        self,
        *args: Any,
        max_age_days: Optional[int] = None,
        **kwargs: Any
    ) -> Optional[T]:
        """
        Load cached data if valid.

        Transparently reads from either compressed (.json.zip) or
        uncompressed (.json) files for backward compatibility.

        Args:
            *args, **kwargs: Arguments passed to compute_cache_key
            max_age_days: Override default TTL for this load

        Returns:
            Cached data if exists and valid, None otherwise
        """
        if not self._is_load_enabled():
            return None

        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        ttl = max_age_days if max_age_days is not None else self.default_ttl_days

        actual_path = self._resolve_actual_path(cache_path)
        if actual_path is None:
            return None

        if not is_file_valid(actual_path, ttl):
            return None

        try:
            if actual_path.name.endswith('.zip'):
                data = read_json_from_zip_with_lock(actual_path)
            else:
                data = read_json_with_lock(actual_path)

            if data is None:
                return None
            return self.deserialize(data)
        except Exception as e:
            self._logger.error(f"Failed to load cache for {key}: {e}")
            return None

    def store(
        self,
        data: T,
        *args: Any,
        override: bool = True,
        **kwargs: Any
    ) -> Optional[Path]:
        """
        Store data to cache.

        When use_compression is True, writes a .json.zip file and cleans up
        any stale .json file. When False, writes .json and cleans up any
        stale .json.zip file.

        Args:
            data: Data to cache
            *args, **kwargs: Arguments passed to compute_cache_key
            override: If False, skip if cache already exists (checks both formats)

        Returns:
            Path to cache file if successful, None otherwise
        """
        if not self._is_store_enabled():
            return None

        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        compressed_path = self._get_compressed_path(cache_path)

        if not override:
            existing = self._resolve_actual_path(cache_path)
            if existing is not None:
                return existing

        try:
            serialized = self.serialize(data)
            if serialized is None:
                self._logger.error(f"Serialization returned None for {key}")
                return None

            if self.use_compression:
                if write_json_as_zip_with_lock(compressed_path, serialized):
                    # Clean up stale uncompressed variant
                    if cache_path.is_file():
                        safe_remove(cache_path, allowed_dir_path=self._config.dataset_dir)
                    return compressed_path
                return None
            else:
                if write_json_with_lock(cache_path, serialized):
                    # Clean up stale compressed variant
                    if compressed_path.is_file():
                        safe_remove(compressed_path, allowed_dir_path=self._config.dataset_dir)
                    return cache_path
                return None
        except Exception as e:
            self._logger.error(f"Failed to store cache for {key}: {e}")
            return None

    def invalidate(self, *args: Any, **kwargs: Any) -> bool:
        """
        Remove a specific cache entry.

        Removes both compressed and uncompressed variants to ensure
        complete invalidation.

        Args:
            *args, **kwargs: Arguments passed to compute_cache_key

        Returns:
            True if removed or didn't exist, False on error
        """
        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        compressed_path = self._get_compressed_path(cache_path)

        success = True
        if cache_path.is_file():
            if not safe_remove(cache_path, allowed_dir_path=self._config.dataset_dir):
                success = False
        if compressed_path.is_file():
            if not safe_remove(compressed_path, allowed_dir_path=self._config.dataset_dir):
                success = False
        return success

    def clear(self, max_age_days: Optional[int] = None) -> int:
        """
        Clear cache files.

        Clears both compressed (.json.zip) and uncompressed (.json) files.

        Args:
            max_age_days: If specified, only clear files older than this

        Returns:
            Number of files cleared
        """
        cache_dir = self.get_cache_dir()
        if not cache_dir.exists():
            return 0

        cleared = 0
        for ext in (self.file_extension, self.file_extension + ".zip"):
            for cache_file in cache_dir.glob(f"*{ext}"):
                should_clear = (
                    max_age_days is None or
                    not is_file_valid(cache_file, max_age_days)
                )
                if should_clear:
                    if safe_remove(cache_file, allowed_dir_path=self._config.dataset_dir):
                        cleared += 1

        return cleared

    def exists(self, *args: Any, **kwargs: Any) -> bool:
        """
        Check if cache entry exists (ignoring TTL).

        Checks both compressed and uncompressed variants.

        Args:
            *args, **kwargs: Arguments passed to compute_cache_key

        Returns:
            True if cache file exists in either format
        """
        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        return self._resolve_actual_path(cache_path) is not None

    def is_valid(
        self,
        *args: Any,
        max_age_days: Optional[int] = None,
        **kwargs: Any
    ) -> bool:
        """
        Check if cache entry exists and is valid (respecting TTL).

        Checks both compressed and uncompressed variants.

        Args:
            *args, **kwargs: Arguments passed to compute_cache_key
            max_age_days: Override default TTL for this check

        Returns:
            True if cache exists and is within TTL
        """
        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        ttl = max_age_days if max_age_days is not None else self.default_ttl_days

        actual_path = self._resolve_actual_path(cache_path)
        if actual_path is None:
            return False
        return is_file_valid(actual_path, ttl)

    # Private helper methods

    def _is_load_enabled(self) -> bool:
        """Check if loading from cache is enabled for this cache type."""
        return self._config.is_cache_load_enabled(self.cache_type)

    def _is_store_enabled(self) -> bool:
        """Check if storing to cache is enabled for this cache type."""
        return self._config.is_cache_store_enabled(self.cache_type)

    def _is_enabled(self) -> bool:
        """
        Check if this cache type is enabled (backward compatibility).

        DEPRECATED: Use _is_load_enabled() or _is_store_enabled() instead.
        Returns True only if BOTH load and store are enabled.
        """
        return self._config.is_cache_enabled(self.cache_type)

    def _is_profile_isolated(self) -> bool:
        """Check if this cache should be isolated by profile."""
        return not self._config.is_cache_shared(
            self.cache_type,
            default=self.shared_by_default
        )
