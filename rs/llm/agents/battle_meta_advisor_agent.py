from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from rs.llm.agents.base_agent import AgentContext
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


class BattleMetaAdvisorAgent:
    """Select a battle comparator profile while preserving deterministic fallback."""

    def __init__(
            self,
            llm_provider: BattleMetaProposalProvider | None = None,
            min_confidence: float = 0.65,
    ):
        self._llm_provider: BattleMetaProposalProvider = (
            BattleMetaLlmProvider() if llm_provider is None else llm_provider
        )
        self._min_confidence = min_confidence

    def decide(self, context: AgentContext) -> BattleMetaDecision:
        deterministic_profile = str(context.extras.get("deterministic_profile", "general")).strip().lower() or "general"
        available_profiles_raw = context.extras.get("available_profiles", [])
        available_profiles = {
            str(profile).strip().lower()
            for profile in available_profiles_raw
            if str(profile).strip() != ""
        }
        if deterministic_profile not in available_profiles:
            available_profiles.add(deterministic_profile)

        proposal = self._llm_provider.propose(context)
        proposed_profile = proposal.comparator_profile

        if (
                proposed_profile is not None
                and proposed_profile in available_profiles
                and proposal.confidence >= self._min_confidence
        ):
            metadata = {
                "phase": "phase_4_battle_meta_advisor",
                "deterministic_profile": deterministic_profile,
            }
            metadata.update(proposal.metadata)
            return BattleMetaDecision(
                comparator_profile=proposed_profile,
                confidence=proposal.confidence,
                explanation=proposal.explanation,
                required_tools_used=["deterministic_battle_profiles", "llm_battle_meta_advisor"],
                metadata=metadata,
            )

        fallback_reason = "low_confidence" if proposed_profile in available_profiles else "llm_no_valid_profile"
        metadata = {
            "phase": "phase_4_battle_meta_advisor",
            "deterministic_profile": deterministic_profile,
            "fallback_reason": fallback_reason,
        }
        metadata.update(proposal.metadata)
        return BattleMetaDecision(
            comparator_profile=deterministic_profile,
            confidence=0.0,
            explanation="deterministic_battle_profile",
            required_tools_used=["deterministic_battle_profiles"],
            metadata=metadata,
        )
