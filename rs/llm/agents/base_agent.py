from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rs.llm.validator import validate_command


@dataclass
class AgentContext:
    """Normalized input payload for a handler-specific advisor.
    """

    handler_name: str
    screen_type: str
    available_commands: List[str]
    choice_list: List[str]
    game_state: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    """Structured advisor result before command execution.
    """

    proposed_command: str | None
    confidence: float
    explanation: str
    required_tools_used: List[str] = field(default_factory=list)
    fallback_recommended: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTool(ABC):
    """Tool contract for deterministic capabilities exposed to agents.
    """

    name: str

    @abstractmethod
    def run(self, context: AgentContext, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool request.

        Args:
            context: Current decision context.
            payload: Tool-specific input payload.

        Returns:
            Dict[str, Any]: Tool result payload.
        """
        raise Exception("must be implemented by children")


class BaseAgent(ABC):
    """Base class for handler advisor agents.

    Subclasses implement `_decide` and may use registered tools. The public
    `decide` method enforces a consistent output contract and basic validation.

    """

    def __init__(self, name: str, timeout_ms: int = 1500):
        self.name: str = name
        self.timeout_ms: int = timeout_ms
        self._tools: Dict[str, AgentTool] = {}

    def register_tool(self, tool: AgentTool) -> None:
        """Register a named tool.

        Args:
            tool: Tool implementation to expose to this agent.

        Returns:
            None.
        """
        self._tools[tool.name] = tool

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered.

        Args:
            name: Tool name.

        Returns:
            bool: True when the tool is registered.
        """
        return name in self._tools

    def call_tool(self, name: str, context: AgentContext, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool.

        Args:
            name: Registered tool name.
            context: Current decision context.
            payload: Tool-specific input payload.

        Returns:
            Dict[str, Any]: Tool execution result.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered for agent '{self.name}'")
        return self._tools[name].run(context, payload)

    def decide(self, context: AgentContext) -> AgentDecision:
        """Run the full decision flow.

        Args:
            context: Current decision context.

        Returns:
            AgentDecision: Parsed and normalized decision.
        """
        raw_output = self._decide(context)
        decision = self._parse_decision(raw_output)
        return self._normalize_decision(decision, context)

    @abstractmethod
    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        """Produce raw model/tool output.

        Args:
            context: Current decision context.

        Returns:
            Dict[str, Any]: Raw decision payload before parsing/normalization.
        """
        raise Exception("must be implemented by children")

    def _parse_decision(self, raw_output: Dict[str, Any]) -> AgentDecision:
        """Parse raw output into canonical decision format.

        Args:
            raw_output: Raw output dictionary from `_decide`.

        Returns:
            AgentDecision: Parsed decision object.
        """
        return AgentDecision(
            proposed_command=raw_output.get("proposed_command"),
            confidence=float(raw_output.get("confidence", 0.0)),
            explanation=str(raw_output.get("explanation", "")),
            required_tools_used=list(raw_output.get("required_tools_used", [])),
            fallback_recommended=bool(raw_output.get("fallback_recommended", False)),
            metadata=dict(raw_output.get("metadata", {})),
        )

    def _normalize_decision(self, decision: AgentDecision, context: AgentContext) -> AgentDecision:
        """Normalize and validate a decision.

        Args:
            decision: Parsed decision from `_parse_decision`.
            context: Current decision context.

        Returns:
            AgentDecision: Normalized decision with validation metadata.
        """
        decision.confidence = min(1.0, max(0.0, decision.confidence))

        if decision.proposed_command is None:
            decision.metadata["validation_error"] = "empty_command"
            decision.fallback_recommended = True
            return decision

        validation = validate_command(context, decision.proposed_command)
        decision.metadata["validation_error"] = validation.code
        if not validation.is_valid:
            decision.metadata["validation_message"] = validation.message
            decision.fallback_recommended = True
        return decision
