"""
Cache implementations for decompilation-related data.

This module provides cache implementations for:
- DecompileResultCache: Decompiled pseudocode from Ghidra, IDA Pro, or Binary Ninja
- IDAIndexCache: IDA pre-indexing results (function metadata, call graphs, etc.)
- IDADatabaseCache: IDA database files (.i64/.idb) for reuse across sessions
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import BaseCache
from ..utils import compute_hash, is_file_valid, safe_remove

if TYPE_CHECKING:
    from decompile_scripts.ida_backend.ida_proto import CachableIDAIndexCache


class DecompileResultCache(BaseCache[Any, tuple]):
    """
    Cache for decompilation results (pseudocode from binaries).

    Cache key: SHA256 hash of (agent_id + proj_id) combined with binary name
    File format: Serialized DecompilationResult object

    Decompilation is deterministic (same binary always produces same
    pseudocode), so this cache is shared by default to avoid redundant
    decompilation across profiles.
    """

    cache_type = "decompile_result"
    default_ttl_days = None  # Never expires (deterministic)
    shared_by_default = True  # Shared (deterministic output)

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "decompile_output"

    def compute_cache_key(self, agent_id: str, proj_id: str, binary_name: str) -> str:
        """
        Compute cache key from agent ID, project ID, and binary name.

        The key is organized as: hash(agent_id+proj_id)/binary_name
        This allows grouping related decompilations together.
        """
        dir_hash = compute_hash(agent_id + proj_id)
        return f"{dir_hash}/{binary_name}_result"

    def get_cache_path(self, key: str) -> Path:
        """
        Override to handle subdirectory structure.

        The key format is "hash/binary_name_result", so we need to
        create the hash subdirectory.
        """
        cache_dir = self.get_cache_dir()
        parts = key.split('/', 1)
        if len(parts) == 2:
            subdir = cache_dir / parts[0]
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"{parts[1]}{self.file_extension}"
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir / f"{key}{self.file_extension}"

    def serialize(self, data: Any) -> Optional[Dict[str, Any]]:
        """Convert DecompilationResult to dict using its to_dict method."""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        return data

    def deserialize(self, data: Dict[str, Any]) -> Optional[Any]:
        """Reconstruct DecompilationResult from dict."""
        try:
            from decompile_scripts.decompile_proto import DecompilationResult
            return DecompilationResult.from_dict(data)
        except ImportError:
            return data

    def get_decompile_dir(self, agent_id: str, proj_id: str) -> Path:
        """
        Get the decompilation output directory for a given agent and project.

        This is a convenience method that returns the directory where
        decompilation results for a specific agent/project combination
        are stored.
        """
        dir_hash = compute_hash(agent_id + proj_id)
        output_dir = self.get_cache_dir() / dir_hash
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class IDAIndexCache(BaseCache["CachableIDAIndexCache", str]):
    """
    Cache for IDA pre-indexing results (function metadata, call graphs, etc.).

    Cache key: SHA256 hash of the binary file contents (pre-computed by caller)
    File format: JSON serialization of CachableIDAIndexCache

    This cache stores pre-indexed binary analysis data including function names,
    addresses, signatures, call graphs, strings, and cross-references. It is
    shared by default since the indexing is deterministic for the same binary.
    """

    cache_type = "ida_index"
    default_ttl_days = None  # Never expires (deterministic)
    shared_by_default = True  # Shared (deterministic output)

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "ida_index_cache"

    def compute_cache_key(self, binary_hash: str) -> str:
        """
        Compute cache key from pre-computed binary file hash.

        Args:
            binary_hash: SHA256 hash of the binary file contents
                         (typically from do_hash_file())
        """
        return binary_hash

    def serialize(self, data: CachableIDAIndexCache) -> Optional[Dict[str, Any]]:
        """Convert CachableIDAIndexCache to dict."""
        return data.to_dict()

    def deserialize(self, data: Dict[str, Any]) -> Optional[CachableIDAIndexCache]:
        """Reconstruct CachableIDAIndexCache from dict."""
        from decompile_scripts.ida_backend.ida_proto import CachableIDAIndexCache as _CachableIDAIndexCache
        return _CachableIDAIndexCache.from_dict(data)


class IDADatabaseCache(BaseCache[Path, str]):
    """
    Cache for IDA database files (.i64/.idb).

    Unlike other caches that store JSON, this cache manages binary IDA
    database files. The store() method copies/moves an .i64 file into
    the cache directory, and load() returns the Path to the cached file.

    Cache key: SHA256 hash of the original binary file contents
    File format: .i64 binary database (optionally compressed to .i64.zip)

    This cache is shared by default since IDA databases are deterministic
    for the same binary input.
    """

    cache_type = "ida_database"
    default_ttl_days = None  # Never expires (deterministic)
    shared_by_default = True  # Same binary -> same IDB
    file_extension = ".i64"  # Canonical extension
    use_compression = True  # Store as .i64.zip by default

    _IDB_EXTENSIONS = (".i64", ".i64.zip", ".idb", ".idb.zip")

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "ida_database_cache"

    def compute_cache_key(self, binary_hash: str) -> str:
        """
        Compute cache key from pre-computed binary file hash.

        Args:
            binary_hash: SHA256 hash of the binary file contents
                         (typically from do_hash_file())
        """
        return binary_hash

    def serialize(self, data: Path) -> Optional[Dict[str, Any]]:
        """Not used — store() is overridden for binary file I/O."""
        return None

    def deserialize(self, data: Dict[str, Any]) -> Optional[Path]:
        """Not used — load() is overridden for binary file I/O."""
        return None

    def _resolve_actual_path(self, canonical_path: Path) -> Optional[Path]:
        """
        Resolve the actual IDB file path, checking all format variants.

        Checks .i64.zip, .i64, .idb.zip, .idb in that order (compressed first).
        """
        stem = canonical_path.stem  # the hash
        parent = canonical_path.parent
        for ext in self._IDB_EXTENSIONS:
            candidate = parent / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def load(
        self,
        *args: Any,
        max_age_days: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Path]:
        """
        Load a cached IDA database path.

        Returns the Path to the cached .i64 (or .i64.zip) file if it exists
        and is within TTL, None otherwise. The caller is responsible for
        decompressing .zip variants if needed.
        """
        if not self._is_enabled():
            return None

        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        ttl = max_age_days if max_age_days is not None else self.default_ttl_days

        actual_path = self._resolve_actual_path(cache_path)
        if actual_path is None:
            return None

        if not is_file_valid(actual_path, ttl):
            return None

        return actual_path

    def store(
        self,
        data: Path,
        *args: Any,
        override: bool = True,
        move: bool = False,
        **kwargs: Any,
    ) -> Optional[Path]:
        """
        Store an IDA database file in the cache.

        The target extension is determined by the source file's suffix:
        IDA saves .i64 for 64-bit binaries and .idb for 32-bit binaries.
        This ensures the cached file preserves the correct bitness indicator.

        Args:
            data: Path to the source .i64/.idb file to cache.
            *args, **kwargs: Arguments passed to compute_cache_key.
            override: If False, skip if cache already exists.
            move: If True, remove the source file after storing.

        Returns:
            Path to the cached file if successful, None otherwise.
        """
        if not self._is_enabled():
            return None

        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)  # canonical: {dir}/{hash}.i64

        if not override:
            existing = self._resolve_actual_path(cache_path)
            if existing is not None:
                return existing

        source = Path(data)
        if not source.is_file():
            self._logger.error(f"Source IDB not found: {source}")
            return None

        # Use the source file's extension to preserve bitness (.i64 vs .idb)
        src_ext = source.suffix.lower()  # ".i64" or ".idb"
        if src_ext not in (".i64", ".idb"):
            src_ext = self.file_extension  # fallback to canonical .i64
        stem = cache_path.stem  # the hash
        parent = cache_path.parent

        try:
            parent.mkdir(parents=True, exist_ok=True)

            if self.use_compression:
                from utils.compression_utils import zip_file

                compressed_dest = parent / f"{stem}{src_ext}.zip"
                if zip_file(source, compressed_dest):
                    # Clean up stale variants (other extension, uncompressed)
                    for ext in self._IDB_EXTENSIONS:
                        stale = parent / f"{stem}{ext}"
                        if stale.is_file() and stale != compressed_dest:
                            safe_remove(stale, allowed_dir_path=self._config.dataset_dir)
                    if move:
                        source.unlink(missing_ok=True)
                    return compressed_dest
                return None
            else:
                dest = parent / f"{stem}{src_ext}"
                if move:
                    shutil.move(str(source), str(dest))
                else:
                    shutil.copy2(str(source), str(dest))
                # Clean up stale variants (other extension, compressed)
                for ext in self._IDB_EXTENSIONS:
                    stale = parent / f"{stem}{ext}"
                    if stale.is_file() and stale != dest:
                        safe_remove(stale, allowed_dir_path=self._config.dataset_dir)
                return dest
        except Exception as e:
            self._logger.error(f"Failed to store IDA database for {key}: {e}")
            return None

    def invalidate(self, *args: Any, **kwargs: Any) -> bool:
        """Remove all IDB variants for a given binary hash."""
        key = self.compute_cache_key(*args, **kwargs)
        cache_path = self.get_cache_path(key)
        stem = cache_path.stem
        parent = cache_path.parent

        success = True
        for ext in self._IDB_EXTENSIONS:
            candidate = parent / f"{stem}{ext}"
            if candidate.is_file():
                if not safe_remove(candidate, allowed_dir_path=self._config.dataset_dir):
                    success = False
        return success

    def clear(self, max_age_days: Optional[int] = None) -> int:
        """Clear cached IDB files, optionally filtering by age."""
        cache_dir = self.get_cache_dir()
        if not cache_dir.exists():
            return 0

        cleared = 0
        for ext in self._IDB_EXTENSIONS:
            for cache_file in cache_dir.glob(f"*{ext}"):
                should_clear = (
                    max_age_days is None
                    or not is_file_valid(cache_file, max_age_days)
                )
                if should_clear:
                    if safe_remove(cache_file, allowed_dir_path=self._config.dataset_dir):
                        cleared += 1
        return cleared
