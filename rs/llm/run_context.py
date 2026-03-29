from __future__ import annotations

from threading import Lock

_current_strategy_name = "unknown_strategy"
_lock = Lock()


def set_current_strategy_name(strategy_name: str) -> None:
    normalized = str(strategy_name).strip() or "unknown_strategy"
    with _lock:
        global _current_strategy_name
        _current_strategy_name = normalized


def get_current_strategy_name() -> str:
    with _lock:
        return _current_strategy_name
