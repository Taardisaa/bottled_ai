from __future__ import annotations

import time
from typing import Dict

from rs.llm.agents.base_agent import AgentContext, AgentDecision, BaseAgent
from rs.llm.config import LlmConfig, load_llm_config
from rs.llm.telemetry import build_decision_telemetry, write_decision_telemetry


class AIPlayerAgent:
    """Registry and execution wrapper for handler-specific advisor agents.
    """

    def __init__(self, config: LlmConfig | None = None):
        """Create orchestrator with optional LLM configuration.

        Args:
            config: Optional explicit config. Loads from file when omitted.

        Returns:
            None.
        """
        self._agents_by_handler: Dict[str, BaseAgent] = {}
        self._config: LlmConfig = load_llm_config() if config is None else config

    def register_agent(self, handler_name: str, agent: BaseAgent) -> None:
        """Register an advisor agent.

        Args:
            handler_name: Handler identifier used as lookup key.
            agent: Advisor agent implementation.

        Returns:
            None.
        """
        self._agents_by_handler[handler_name] = agent

    def has_agent(self, handler_name: str) -> bool:
        """Check whether a handler has a registered advisor.

        Args:
            handler_name: Handler identifier.

        Returns:
            bool: True when an advisor is registered.
        """
        return handler_name in self._agents_by_handler

    def decide(self, handler_name: str, context: AgentContext) -> AgentDecision | None:
        """Request a decision from a registered advisor.

        Args:
            handler_name: Handler identifier.
            context: Decision context for the current game state.

        Returns:
            AgentDecision | None: Advisor decision, safe fallback on agent error,
            or None if no advisor is registered.
        """
        agent = self._agents_by_handler.get(handler_name)
        if agent is None:
            return None

        started_at = time.perf_counter()
        try:
            decision = agent.decide(context)
        except Exception as e:
            decision = AgentDecision(
                proposed_command=None,
                confidence=0.0,
                explanation=f"agent_error:{e}",
                fallback_recommended=True,
                metadata={"handler_name": handler_name, "error": str(e)},
            )

        if self._config.telemetry_enabled:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            telemetry = build_decision_telemetry(context, decision, latency_ms)
            try:
                write_decision_telemetry(telemetry, self._config.telemetry_path)
            except Exception:
                pass

        return decision
