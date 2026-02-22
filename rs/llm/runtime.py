from __future__ import annotations

from rs.llm.agents.event_advisor_agent import EventAdvisorAgent
from rs.llm.config import LlmConfig
from rs.llm.orchestrator import AIPlayerAgent

_event_orchestrator: AIPlayerAgent | None = None


def get_event_orchestrator() -> AIPlayerAgent:
    """Return a singleton orchestrator for event advisor decisions."""
    global _event_orchestrator
    if _event_orchestrator is None:
        orchestrator = AIPlayerAgent(config=LlmConfig(telemetry_enabled=False))
        orchestrator.register_agent("EventHandler", EventAdvisorAgent())
        _event_orchestrator = orchestrator
    return _event_orchestrator
