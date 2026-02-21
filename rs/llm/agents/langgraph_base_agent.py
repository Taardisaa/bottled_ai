from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from rs.llm.agents.base_agent import AgentContext, BaseAgent


class LangGraphBaseAgent(BaseAgent):
    """Base class for agents implemented on top of a LangGraph workflow."""

    def __init__(self, name: str, timeout_ms: int = 1500):
        """Initialize a LangGraph-backed agent shell.

        Args:
            name: Human-readable agent name.
            timeout_ms: Soft timeout budget for decision calls.

        Returns:
            None.
        """
        super().__init__(name=name, timeout_ms=timeout_ms)
        self._compiled_graph: Any | None = None

    @abstractmethod
    def build_graph(self) -> Any:
        """Build and return the LangGraph graph definition.

        Args:
            None.

        Returns:
            Any: Graph object prior to compilation.
        """
        raise Exception("must be implemented by children")

    @abstractmethod
    def compile_graph(self, graph: Any) -> Any:
        """Compile a LangGraph graph into an invokable runtime.

        Args:
            graph: Graph returned by `build_graph`.

        Returns:
            Any: Compiled graph runtime.
        """
        raise Exception("must be implemented by children")

    @abstractmethod
    def build_graph_input(self, context: AgentContext) -> Dict[str, Any]:
        """Translate agent context into graph input payload.

        Args:
            context: Current decision context.

        Returns:
            Dict[str, Any]: Input payload for graph invocation.
        """
        raise Exception("must be implemented by children")

    @abstractmethod
    def parse_graph_output(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        """Translate graph output into raw BaseAgent decision payload.

        Args:
            graph_output: Result returned from graph invocation.

        Returns:
            Dict[str, Any]: Raw decision payload consumed by BaseAgent parser.
        """
        raise Exception("must be implemented by children")

    def get_compiled_graph(self) -> Any:
        """Get compiled graph instance, compiling lazily on first use.

        Args:
            None.

        Returns:
            Any: Compiled graph runtime.
        """
        if self._compiled_graph is None:
            graph = self.build_graph()
            self._compiled_graph = self.compile_graph(graph)
        return self._compiled_graph

    def invoke_graph(self, graph_input: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke compiled graph runtime with the provided input payload.

        Args:
            graph_input: Input payload expected by the graph runtime.

        Returns:
            Dict[str, Any]: Graph output dictionary.
        """
        graph = self.get_compiled_graph()
        output = graph.invoke(graph_input)
        return dict(output)

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        """Execute LangGraph workflow and return raw decision payload.

        Args:
            context: Current decision context.

        Returns:
            Dict[str, Any]: Raw decision payload for BaseAgent normalization.
        """
        graph_input = self.build_graph_input(context)
        graph_output = self.invoke_graph(graph_input)
        return self.parse_graph_output(graph_output)
