from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.langgraph_base_agent import LangGraphBaseAgent
from rs.llm.langmem_service import LangMemService, get_langmem_service


class ProposalLike(Protocol):
    proposed_command: str | None


class MemoryLangGraphState(TypedDict, total=False):
    context: AgentContext
    decision_context: AgentContext
    run_memory_summary: str
    recent_llm_decisions: str
    retrieved_episodic_memories: str
    retrieved_semantic_memories: str
    langmem_status: str
    proposal: ProposalLike
    decision_payload: Dict[str, Any]


class MemoryAugmentedLangGraphAgent(LangGraphBaseAgent):
    """Reusable LangGraph advisor flow with run-memory enrichment.

    Subclasses only need to provide the proposal call and the success/fallback
    payload builders. This keeps most advisor ports from re-implementing the
    same LangGraph node plumbing.
    """

    def __init__(
            self,
            name: str,
            timeout_ms: int = 1500,
            langmem_service: LangMemService | None = None,
    ):
        super().__init__(name=name, timeout_ms=timeout_ms)
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service

    def build_graph(self) -> Any:
        graph = StateGraph(MemoryLangGraphState)
        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("enrich_context", self._enrich_context_node)
        graph.add_node("retrieve_langmem", self._retrieve_langmem_node)
        graph.add_node("propose_decision", self._propose_decision_node)
        graph.add_node("finalize_decision", self._finalize_decision_node)
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "enrich_context")
        graph.add_edge("enrich_context", "retrieve_langmem")
        graph.add_edge("retrieve_langmem", "propose_decision")
        graph.add_edge("propose_decision", "finalize_decision")
        graph.add_edge("finalize_decision", END)
        return graph

    def compile_graph(self, graph: Any) -> Any:
        return graph.compile()

    def build_graph_input(self, context: AgentContext) -> Dict[str, Any]:
        return {"context": context}

    def parse_graph_output(self, graph_output: Dict[str, Any]) -> Dict[str, Any]:
        return dict(graph_output.get("decision_payload", {}))

    def _load_memory_node(self, state: MemoryLangGraphState) -> Dict[str, Any]:
        context = state["context"]
        return {
            "run_memory_summary": str(context.extras.get("run_memory_summary", "")),
            "recent_llm_decisions": str(context.extras.get("recent_llm_decisions", "none")),
        }

    def _enrich_context_node(self, state: MemoryLangGraphState) -> Dict[str, Any]:
        context = state["context"]
        extras = dict(context.extras)
        extras["run_memory_summary"] = state.get("run_memory_summary", extras.get("run_memory_summary", ""))
        extras["recent_llm_decisions"] = state.get("recent_llm_decisions", extras.get("recent_llm_decisions", "none"))
        return {"decision_context": replace(context, extras=extras)}

    def _retrieve_langmem_node(self, state: MemoryLangGraphState) -> Dict[str, Any]:
        decision_context = state.get("decision_context", state["context"])
        return self._langmem_service.build_context_memory(decision_context)

    def _propose_decision_node(self, state: MemoryLangGraphState) -> Dict[str, Any]:
        decision_context = state.get("decision_context", state["context"])
        extras = dict(decision_context.extras)
        extras["retrieved_episodic_memories"] = state.get("retrieved_episodic_memories", extras.get("retrieved_episodic_memories", "none"))
        extras["retrieved_semantic_memories"] = state.get("retrieved_semantic_memories", extras.get("retrieved_semantic_memories", "none"))
        extras["langmem_status"] = state.get("langmem_status", extras.get("langmem_status", "disabled_by_config"))
        decision_context = replace(decision_context, extras=extras)
        return {"proposal": self.propose_with_context(decision_context)}

    def _finalize_decision_node(self, state: MemoryLangGraphState) -> Dict[str, Any]:
        decision_context = state.get("decision_context", state["context"])
        proposal = state["proposal"]
        if proposal.proposed_command is not None:
            return {"decision_payload": self.build_success_payload(decision_context, proposal)}
        return {"decision_payload": self.build_fallback_payload(decision_context, proposal)}

    def graph_node_names(self) -> list[str]:
        return ["load_memory", "enrich_context", "retrieve_langmem", "propose_decision", "finalize_decision"]

    def propose_with_context(self, context: AgentContext) -> ProposalLike:
        raise Exception("must be implemented by children")

    def build_success_payload(self, context: AgentContext, proposal: ProposalLike) -> Dict[str, Any]:
        raise Exception("must be implemented by children")

    def build_fallback_payload(self, context: AgentContext, proposal: ProposalLike) -> Dict[str, Any]:
        raise Exception("must be implemented by children")
