"""
Cache implementation for Git API responses.

This module provides the cache implementation for storing REST API
responses from Git hosting platforms (GitHub, GitLab, etc.).
"""

import json

from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseCache
from ..utils import compute_hash


class GitApiCache(BaseCache[Dict[str, Any], tuple]):
    """
    Cache for Git hosting platform REST API responses.

    Cache key: SHA256 hash of JSON-serialized {url, params} dict
    File format: JSON with url, params, timestamp, and response

    This cache stores responses from GitHub, GitLab, Pagure, SourceWare,
    and GNU Savannah APIs. It is shared by default since API responses
    are deterministic for the same URL/params.

    Note: This cache fixes a race condition in the original implementation
    by using file locking for all read/write operations.
    """

    cache_type = "git_api"
    default_ttl_days = None
    shared_by_default = True  # Shared (deterministic API responses)

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "git_api_cache"

    def compute_cache_key(
        self,
        url: str,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute cache key from URL and optional extra parameters.

        Uses JSON serialization of a normalized dict for consistent hashing.
        """
        cache_input = {
            "url": url.strip(),
            "params": extra_params or {}
        }
        cache_str = json.dumps(cache_input, sort_keys=True)
        return compute_hash(cache_str)

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize API response with metadata.

        The stored format includes:
        - url: The original request URL
        - params: Request parameters
        - timestamp: When the response was cached
        - response: The actual API response data
        """
        return data

    def deserialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize API response."""
        return data

    def store_with_metadata(
        self,
        response: Any,
        url: str,
        extra_params: Optional[Dict[str, Any]] = None,
        override: bool = True
    ) -> Optional[Path]:
        """
        Store API response with URL and timestamp metadata.

        This is a convenience method that wraps the response with
        metadata before storing.

        Args:
            response: The API response to cache
            url: The request URL
            extra_params: Optional request parameters
            override: Whether to override existing cache

        Returns:
            Path to cache file if successful, None otherwise
        """
        import time

        cache_data = {
            "url": url,
            "params": extra_params or {},
            "timestamp": time.time(),
            "response": response,
        }

        return self.store(cache_data, url, extra_params, override=override)

    def load_response(
        self,
        url: str,
        extra_params: Optional[Dict[str, Any]] = None,
        max_age_days: Optional[int] = None
    ) -> Optional[Any]:
        """
        Load just the response data (without metadata).

        This is a convenience method that extracts the response
        from the cached metadata structure.

        Args:
            url: The request URL
            extra_params: Optional request parameters
            max_age_days: Override default TTL

        Returns:
            The cached API response, or None if not found/expired
        """
        cache_data = self.load(url, extra_params, max_age_days=max_age_days)
        if cache_data is None:
            return None

        # Handle both old format (direct response) and new format (with metadata)
        if isinstance(cache_data, dict) and "response" in cache_data:
            return cache_data["response"]

        return cache_data
