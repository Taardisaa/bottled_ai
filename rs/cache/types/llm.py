"""
Cache implementation for LLM query responses.

This module provides the cache implementation for storing LLM
(Large Language Model) query/response pairs to reduce API costs
and improve response times.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseCache
from ..utils import compute_hash


class LlmQueryCache(BaseCache[Any, tuple]):
    """
    Cache for LLM query/response pairs.

    Cache key: SHA256 hash of (model, temperature, message, struct_name, tool_info)
    File format: Serialized LLM response (AIMessage, Pydantic model, or dict)

    LLM responses are profile-isolated by default since different profiles
    may use different models or temperatures.

    Note: This cache fixes a race condition in the original implementation
    by using file locking for all read/write operations.
    """

    cache_type = "llm_query"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated (model/temp dependent)

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "llm_query_cache"

    def compute_cache_key(
        self,
        message: str,
        model: str,
        temperature: float,
        struct: Optional[Any] = None,
        tools: Optional[Any] = None,
        cache_namespace: Optional[str] = None
    ) -> str:
        """
        Compute cache key from LLM query parameters.

        The key is a SHA256 hash of the combined parameters, ensuring
        that the same query with the same parameters always returns
        the same cached response.

        When cache_namespace is provided (e.g., a config profile name),
        it is included in the hash and prepended as a path prefix so
        that namespaced entries are stored in a subdirectory.
        """
        if struct is not None:
            if hasattr(struct, '__name__'):
                struct_name = struct.__name__
            elif hasattr(struct, '_name'):  # TypedDict
                struct_name = struct._name
            elif struct is dict:
                struct_name = "dict"
            else:
                struct_name = str(struct)
        else:
            struct_name = "no_struct"

        key_str = f"{model}##{str(temperature)}##{message}##{struct_name}"

        if tools is not None:
            if isinstance(tools, list):
                tool_names = []
                for tool in tools:
                    if hasattr(tool, 'name'):
                        tool_names.append(tool.name)
                    elif hasattr(tool, '__name__'):
                        tool_names.append(tool.__name__)
                tool_info = ",".join(sorted(tool_names))
            else:
                tool_info = str(tools)
            key_str = f"{key_str}##{tool_info}"

        if cache_namespace:
            key_str = f"{cache_namespace}##{key_str}"

        hash_key = compute_hash(key_str)

        # Prepend namespace as path prefix for directory-level separation
        if cache_namespace:
            return f"{cache_namespace}/{hash_key}"
        return hash_key

    def serialize(self, data: Any) -> Dict[str, Any]:
        """
        Serialize LLM response to JSON-compatible format.

        Handles various response types:
        - AIMessage objects (from LangChain)
        - Pydantic BaseModel instances
        - Plain dictionaries
        """
        # Handle AIMessage
        if hasattr(data, 'content') and hasattr(data, 'type'):
            return {
                "_type": "AIMessage",
                "content": data.content,
                "additional_kwargs": getattr(data, 'additional_kwargs', {}),
                "response_metadata": getattr(data, 'response_metadata', {}),
            }

        # Handle Pydantic models
        if hasattr(data, 'model_dump'):
            return {
                "_type": "pydantic",
                "_class": data.__class__.__name__,
                "data": data.model_dump(),
            }

        # Handle plain dicts
        if isinstance(data, dict):
            return data

        # Fallback: try to convert to dict
        if hasattr(data, 'to_dict'):
            return data.to_dict()

        # Last resort: wrap in dict
        return {"_raw": str(data)}

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize LLM response from JSON format.

        Attempts to reconstruct the original object type.
        """
        if not isinstance(data, dict):
            return data

        response_type = data.get("_type")

        # Reconstruct AIMessage
        if response_type == "AIMessage":
            try:
                from langchain_core.messages import AIMessage
                return AIMessage(
                    content=data.get("content", ""),
                    additional_kwargs=data.get("additional_kwargs", {}),
                    response_metadata=data.get("response_metadata", {}),
                )
            except ImportError:
                return data

        # Pydantic model - return as dict (caller must reconstruct)
        if response_type == "pydantic":
            return data.get("data", data)

        # Raw value
        if "_raw" in data:
            return data["_raw"]

        # Old format: detect AIMessage by `type` field (legacy cache entries)
        if data.get("type") == "ai" and "content" in data:
            try:
                from langchain_core.messages import AIMessage
                return AIMessage(
                    content=data.get("content", ""),
                    additional_kwargs=data.get("additional_kwargs", {}),
                    response_metadata=data.get("response_metadata", {}),
                )
            except ImportError:
                return data

        return data
