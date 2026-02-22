"""
Cache implementation for web scraping results.

This module provides the cache implementation for storing web
scraping results from CVE-related URLs and commit discovery.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseCache
from ..utils import compute_hash


class WebHandlerCache(BaseCache[Any, str]):
    """
    Cache for web scraping/handler results.

    Cache key: SHA256 hash of URL
    File format: Serialized TryHandleResult object

    This caches results from web scraping operations used for
    discovering CVE-related commits and patch information from
    various web sources.
    """

    cache_type = "web_handler"
    default_ttl_days = None
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "web_handler"

    def compute_cache_key(self, url: str) -> str:
        """Hash the URL for filesystem-safe filename."""
        return compute_hash(url)

    def serialize(self, data: Any) -> Dict[str, Any]:
        """
        Serialize web handler result.

        Handles TryHandleResult Pydantic models and plain dicts.
        """
        # Handle Pydantic models
        if hasattr(data, 'model_dump'):
            return data.model_dump()

        # Handle objects with to_dict
        if hasattr(data, 'to_dict'):
            return data.to_dict()

        # Handle plain dicts
        if isinstance(data, dict):
            return data

        # Fallback
        return {"_raw": str(data)}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize web handler result.

        Attempts to reconstruct TryHandleResult if available.
        """
        if not isinstance(data, dict):
            return data

        # Handle raw values
        if "_raw" in data:
            return data["_raw"]

        # Try to reconstruct TryHandleResult
        try:
            from patch_fetcher.web_fetcher import TryHandleResult
            return TryHandleResult.model_validate(data)
        except (ImportError, Exception):
            return data
