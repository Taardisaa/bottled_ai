from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
from typing import Dict

from rs.llm.agents.base_agent import AgentContext, AgentDecision, BaseAgent
from rs.llm.config import LlmConfig, load_llm_config
from rs.llm.telemetry import build_decision_telemetry, write_decision_telemetry
from rs.llm.validator import validate_command


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

    def _decide_with_timeout(self, agent: BaseAgent, context: AgentContext) -> AgentDecision:
        if self._config.timeout_ms < 0:
            return agent.decide(context)

        timeout_seconds = max(0.001, self._config.timeout_ms / 1000.0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.decide, context)
            return future.result(timeout=timeout_seconds)

    def decide(self, handler_name: str, context: AgentContext) -> AgentDecision | None:
        """Request a decision from a registered advisor.

        Args:
            handler_name: Handler identifier.
            context: Decision context for the current game state.

        Returns:
            AgentDecision | None: Advisor decision, safe fallback on agent error,
            or None if no advisor is registered.
        """
        if not self._config.is_handler_enabled(handler_name):
            return None

        agent = self._agents_by_handler.get(handler_name)
        if agent is None:
            return None

        started_at = time.perf_counter()
        decision: AgentDecision | None = None
        should_return_none = False
        last_error: str | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                decision = self._decide_with_timeout(agent, context)
                last_error = None
                break
            except FutureTimeoutError:
                last_error = f"timeout_after_{self._config.timeout_ms}ms"
                if attempt < self._config.max_retries:
                    continue
                should_return_none = True
            except Exception as e:
                last_error = str(e)
                if attempt < self._config.max_retries:
                    continue
                should_return_none = True
            if last_error is not None:
                break

        if last_error is None and decision is not None:
            if decision.fallback_recommended or decision.proposed_command is None:
                should_return_none = True

            elif decision.confidence < self._config.confidence_threshold:
                should_return_none = True

            else:
                validation = validate_command(context, decision.proposed_command)
                if not validation.is_valid:
                    should_return_none = True

        if self._config.telemetry_enabled and decision is not None and not should_return_none:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            telemetry = build_decision_telemetry(context, decision, latency_ms)
            try:
                write_decision_telemetry(telemetry, self._config.telemetry_path)
            except Exception:
                pass

        if should_return_none:
            return None
        return decision
