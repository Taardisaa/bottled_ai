from __future__ import annotations

from threading import Lock

DEFAULT_AGENT_IDENTITY = "neo_primates"
_current_agent_identity = DEFAULT_AGENT_IDENTITY
_lock = Lock()


def set_current_agent_identity(agent_identity: str) -> None:
    normalized = str(agent_identity).strip() or DEFAULT_AGENT_IDENTITY
    with _lock:
        global _current_agent_identity
        _current_agent_identity = normalized


def get_current_agent_identity() -> str:
    with _lock:
        return _current_agent_identity
