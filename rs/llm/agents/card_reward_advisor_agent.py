from __future__ import annotations

from typing import Any, Dict, Protocol

from rs.llm.agents.base_agent import AgentContext
from rs.llm.langmem_service import LangMemService
from rs.llm.agents.memory_langgraph_agent import MemoryAugmentedLangGraphAgent
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProposal, CardRewardLlmProvider


class CardRewardProposalProvider(Protocol):
    def propose(self, context: AgentContext) -> CardRewardLlmProposal:
        ...


class CardRewardAdvisorAgent(MemoryAugmentedLangGraphAgent):
    """LLM-backed advisor for card reward decisions."""

    def __init__(
            self,
            timeout_ms: int = 1500,
            llm_provider: CardRewardProposalProvider | None = None,
            langmem_service: LangMemService | None = None,
    ):
        super().__init__(name="card_reward_advisor", timeout_ms=timeout_ms, langmem_service=langmem_service)
        self._llm_provider: CardRewardProposalProvider = CardRewardLlmProvider() if llm_provider is None else llm_provider

    def propose_with_context(self, context: AgentContext) -> CardRewardLlmProposal:
        return self._llm_provider.propose(context)

    def build_success_payload(self, context: AgentContext, proposal: CardRewardLlmProposal) -> Dict[str, Any]:
        metadata = {
            "phase": "phase_2_card_reward",
            "graph_runtime": "langgraph",
            "graph_nodes": self.graph_node_names(),
        }
        metadata.update(proposal.metadata)
        return {
            "proposed_command": proposal.proposed_command,
            "confidence": proposal.confidence,
            "explanation": proposal.explanation,
            "required_tools_used": ["llm_card_reward_advisor", "langgraph_workflow"],
            "fallback_recommended": False,
            "metadata": metadata,
        }

    def build_fallback_payload(self, context: AgentContext, proposal: CardRewardLlmProposal) -> Dict[str, Any]:
        metadata = {
            "phase": "phase_2_card_reward",
            "fallback_reason": "llm_no_decision",
            "graph_runtime": "langgraph",
            "graph_nodes": self.graph_node_names(),
        }
        metadata.update(proposal.metadata)
        return {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": proposal.explanation,
            "required_tools_used": [],
            "fallback_recommended": True,
            "metadata": metadata,
        }
