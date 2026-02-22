"""
Cache type implementations.

This package contains all the concrete cache implementations for different
data types in the CHPC system.
"""

from .vulnerability import (
    NvdQueryCache,
    OsvQueryCache,
    CveEntryCache,
    OsvEntryCache,
    PatchInfoCache,
    GenericEntryCache,
    CpeMatchCache,
)
from .diff import FileChangeCache, OutOfFuncChangeCache
from .agent import AgentCheckResultCache, BranchTagCache, DiAgentStateCache, RoAgentStateCache, FpcAgentStateCache
from .llm import LlmQueryCache
from .git_api import GitApiCache
from .web import WebHandlerCache

__all__ = [
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
    "LlmQueryCache",
    "GitApiCache",
    "WebHandlerCache",
]
