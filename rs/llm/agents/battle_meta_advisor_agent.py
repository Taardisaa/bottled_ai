from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.decision_memory import DecisionMemoryStore
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.battle_meta_llm_provider import BattleMetaLlmProposal, BattleMetaLlmProvider


@dataclass
class BattleMetaDecision:
    comparator_profile: str
    confidence: float
    explanation: str
    required_tools_used: list[str] = field(default_factory=list)
    fallback_recommended: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BattleMetaProposalProvider(Protocol):
    def propose(self, context: AgentContext) -> BattleMetaLlmProposal:
        ...


class BattleMetaGraphState(TypedDict, total=False):
    context: AgentContext
    decision_context: AgentContext
    deterministic_profile: str
    available_profiles: list[str]
    recent_llm_decisions: str
    retrieved_episodic_memories: str
    retrieved_semantic_memories: str
    langmem_status: str
    proposal: BattleMetaLlmProposal
    decision: BattleMetaDecision


class BattleMetaAdvisorAgent:
    """Select a battle comparator profile while preserving deterministic fallback."""

    def __init__(
            self,
            llm_provider: BattleMetaProposalProvider | None = None,
            min_confidence: float = 0.65,
            memory_store: DecisionMemoryStore | None = None,
            langmem_service: LangMemService | None = None,
    ):
        self._llm_provider: BattleMetaProposalProvider = (
            BattleMetaLlmProvider() if llm_provider is None else llm_provider
        )
        self._min_confidence = min_confidence
        self._memory_store = DecisionMemoryStore() if memory_store is None else memory_store
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._compiled_graph: Any | None = None

    def build_graph(self) -> Any:
        graph = StateGraph(BattleMetaGraphState)
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

    def get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    def decide(self, context: AgentContext) -> BattleMetaDecision:
        deterministic_profile = str(context.extras.get("deterministic_profile", "general")).strip().lower() or "general"
        available_profiles_raw = context.extras.get("available_profiles", [])
        available_profiles = sorted({
            str(profile).strip().lower()
            for profile in available_profiles_raw
            if str(profile).strip() != ""
        })
        if deterministic_profile not in available_profiles:
            available_profiles.append(deterministic_profile)

        graph_output = self.get_compiled_graph().invoke({
            "context": context,
            "deterministic_profile": deterministic_profile,
            "available_profiles": available_profiles,
        })
        decision = graph_output["decision"]

        if not decision.fallback_recommended and decision.comparator_profile in available_profiles:
            accepted_decision = AgentDecision(
                proposed_command=f"profile {decision.comparator_profile}",
                confidence=decision.confidence,
                explanation=decision.explanation,
                required_tools_used=decision.required_tools_used,
                fallback_recommended=False,
                metadata=dict(decision.metadata),
            )
            self._memory_store.record(context, accepted_decision)
            self._langmem_service.record_accepted_decision(context, accepted_decision)
        return decision

    def graph_node_names(self) -> list[str]:
        return ["load_memory", "enrich_context", "retrieve_langmem", "propose_decision", "finalize_decision"]

    def _load_memory_node(self, state: BattleMetaGraphState) -> Dict[str, Any]:
        context = state["context"]
        return {
            "recent_llm_decisions": self._memory_store.build_recent_decisions_summary(context),
        }

    def _enrich_context_node(self, state: BattleMetaGraphState) -> Dict[str, Any]:
        context = state["context"]
        extras = dict(context.extras)
        extras["recent_llm_decisions"] = state.get("recent_llm_decisions", extras.get("recent_llm_decisions", "none"))
        return {"decision_context": replace(context, extras=extras)}

    def _retrieve_langmem_node(self, state: BattleMetaGraphState) -> Dict[str, Any]:
        decision_context = state.get("decision_context", state["context"])
        return self._langmem_service.build_context_memory(decision_context)

    def _propose_decision_node(self, state: BattleMetaGraphState) -> Dict[str, Any]:
        decision_context = state.get("decision_context", state["context"])
        extras = dict(decision_context.extras)
        extras["retrieved_episodic_memories"] = state.get("retrieved_episodic_memories", extras.get("retrieved_episodic_memories", "none"))
        extras["retrieved_semantic_memories"] = state.get("retrieved_semantic_memories", extras.get("retrieved_semantic_memories", "none"))
        extras["langmem_status"] = state.get("langmem_status", extras.get("langmem_status", "disabled_by_config"))
        decision_context = replace(decision_context, extras=extras)
        return {"proposal": self._llm_provider.propose(decision_context)}

    def _finalize_decision_node(self, state: BattleMetaGraphState) -> Dict[str, BattleMetaDecision]:
        deterministic_profile = state["deterministic_profile"]
        available_profiles = set(state.get("available_profiles", []))
        proposal = state["proposal"]
        proposed_profile = proposal.comparator_profile

        if (
                proposed_profile is not None
                and proposed_profile in available_profiles
                and proposal.confidence >= self._min_confidence
        ):
            metadata = {
                "phase": "phase_4_battle_meta_advisor",
                "deterministic_profile": deterministic_profile,
                "graph_runtime": "langgraph",
                "graph_nodes": self.graph_node_names(),
            }
            metadata.update(proposal.metadata)
            return {
                "decision": BattleMetaDecision(
                    comparator_profile=proposed_profile,
                    confidence=proposal.confidence,
                    explanation=proposal.explanation,
                    required_tools_used=["deterministic_battle_profiles", "llm_battle_meta_advisor", "langgraph_workflow"],
                    metadata=metadata,
                )
            }

        fallback_reason = "low_confidence" if proposed_profile in available_profiles else "llm_no_valid_profile"
        metadata = {
            "phase": "phase_4_battle_meta_advisor",
            "deterministic_profile": deterministic_profile,
            "fallback_reason": fallback_reason,
            "graph_runtime": "langgraph",
            "graph_nodes": self.graph_node_names(),
        }
        metadata.update(proposal.metadata)
        return {
            "decision": BattleMetaDecision(
                comparator_profile=deterministic_profile,
                confidence=0.0,
                explanation="deterministic_battle_profile",
                required_tools_used=["deterministic_battle_profiles", "langgraph_workflow"],
                fallback_recommended=True,
                metadata=metadata,
            )
        }
