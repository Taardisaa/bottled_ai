from __future__ import annotations

from typing import Any, Dict, Protocol

from rs.llm.agents.base_agent import AgentContext, BaseAgent
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProposal, CardRewardLlmProvider


class CardRewardProposalProvider(Protocol):
    def propose(self, context: AgentContext) -> CardRewardLlmProposal:
        ...


class CardRewardAdvisorAgent(BaseAgent):
    """LLM-backed advisor for card reward decisions."""

    def __init__(
            self,
            timeout_ms: int = 1500,
            llm_provider: CardRewardProposalProvider | None = None,
    ):
        super().__init__(name="card_reward_advisor", timeout_ms=timeout_ms)
        self._llm_provider: CardRewardProposalProvider = CardRewardLlmProvider() if llm_provider is None else llm_provider

    def _decide(self, context: AgentContext) -> Dict[str, Any]:
        proposal = self._llm_provider.propose(context)
        if proposal.proposed_command is not None:
            metadata = {
                "phase": "phase_2_card_reward",
            }
            metadata.update(proposal.metadata)
            return {
                "proposed_command": proposal.proposed_command,
                "confidence": proposal.confidence,
                "explanation": proposal.explanation,
                "required_tools_used": ["llm_card_reward_advisor"],
                "fallback_recommended": False,
                "metadata": metadata,
            }

        metadata = {
            "phase": "phase_2_card_reward",
            "fallback_reason": "llm_no_decision",
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
