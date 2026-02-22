"""
Cache implementations for agent-related data.

This module provides cache implementations for:
- Agent check results (patch verification results)
- Decompile interactor agent state
- Repo operator agent state
- Final patch checker agent state
"""

from pathlib import Path
from typing import Any, Dict

from ..base import BaseCache


class AgentCheckResultCache(BaseCache[Any, str]):
    """
    Cache for agent check results (patch verification results).

    Cache key: Agent ID
    File format: Serialized AgentCheckResult object

    This caches the final results of patch presence verification,
    including the status (APPLIED/NOT_APPLIED/UNKNOWN), reasoning,
    and confidence score.
    """

    cache_type = "agent_check_result"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "agent_cache"

    def compute_cache_key(self, agent_id: str) -> str:
        """Use the agent ID directly as the cache key."""
        return agent_id

    def serialize(self, data: Any) -> Dict[str, Any]:
        """Convert AgentCheckResult to dict using its to_dict method."""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        return data

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Reconstruct AgentCheckResult from dict."""
        try:
            from checker_agent.checker_proto import AgentCheckResult
            return AgentCheckResult.from_dict(data)
        except ImportError:
            return data


class DiAgentStateCache(BaseCache[Dict[str, Any], str]):
    """
    Cache for Decompile Interactor agent task state.

    Cache key: Agent ID (prefixed with "DI_" for decompile interactor)
    File format: Serialized task state dictionary

    This caches the state of decompile interactor agent tasks for
    resumable workflows.
    """

    cache_type = "di_agent_state"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "true_agent_cache" / "di"

    def compute_cache_key(self, agent_id: str) -> str:
        """Use the agent ID with DI prefix as the cache key."""
        if not agent_id.startswith("DI_"):
            return f"DI_{agent_id}"
        return agent_id

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data

    def deserialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data


class BranchTagCache(BaseCache[Any, str]):
    """
    Cache for branch/tag resolution results.

    Cache key: cache_id (project front part identifier)
    File format: Serialized BranchTagResult (Pydantic model)
    """

    cache_type = "branch_tag"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "branch_tag_results"

    def compute_cache_key(self, cache_id: str) -> str:
        """Use the cache_id directly as the cache key."""
        return cache_id

    def serialize(self, data: Any) -> Dict[str, Any]:
        """Convert BranchTagResult to dict."""
        if hasattr(data, 'model_dump'):
            return data.model_dump()
        return data

    def deserialize(self, data: Dict[str, Any]) -> Any:
        """Reconstruct BranchTagResult from dict."""
        try:
            from checker_agent.commit_window_checker import BranchTagResult
            return BranchTagResult.model_validate(data)
        except (ImportError, Exception):
            return data


class RoAgentStateCache(BaseCache[Dict[str, Any], str]):
    """
    Cache for Repo Operator agent task state.

    Cache key: Agent ID (prefixed with "RO_" for repo operator)
    File format: Serialized task state dictionary

    This caches the state of repo operator agent tasks for
    resumable workflows.
    """

    cache_type = "ro_agent_state"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "true_agent_cache" / "ro"

    def compute_cache_key(self, agent_id: str) -> str:
        """Use the agent ID with RO prefix as the cache key."""
        if not agent_id.startswith("RO_"):
            return f"RO_{agent_id}"
        return agent_id

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data

    def deserialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data


class FpcAgentStateCache(BaseCache[Dict[str, Any], str]):
    """
    Cache for Final Patch Checker agent task state.

    Cache key: Agent ID (prefixed with "FPC_" for final patch checker)
    File format: Serialized task state dictionary

    This caches the state of final patch checker agent tasks.
    Isolated per profile by default since the final check is model-sensitive.
    """

    cache_type = "fpc_agent_state"
    default_ttl_days = None  # Never expires by default
    shared_by_default = False  # Profile-isolated

    def get_base_dir(self) -> Path:
        return self._config.dataset_dir / "true_agent_cache" / "fpc"

    def compute_cache_key(self, agent_id: str) -> str:
        """Use the agent ID with FPC prefix as the cache key."""
        if not agent_id.startswith("FPC_"):
            return f"FPC_{agent_id}"
        return agent_id

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data

    def deserialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Task state is already a dict."""
        return data
