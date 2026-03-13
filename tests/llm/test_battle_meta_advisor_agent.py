import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.battle_meta_advisor_agent import BattleMetaAdvisorAgent
from rs.llm.providers.battle_meta_llm_provider import BattleMetaLlmProposal


class StubBattleMetaProvider:
    def __init__(self, proposal: BattleMetaLlmProposal):
        self._proposal = proposal

    def propose(self, context: AgentContext) -> BattleMetaLlmProposal:
        return self._proposal


class TestBattleMetaAdvisorAgent(unittest.TestCase):
    def test_uses_valid_high_confidence_profile(self):
        agent = BattleMetaAdvisorAgent(
            llm_provider=StubBattleMetaProvider(
                BattleMetaLlmProposal(
                    comparator_profile="general",
                    confidence=0.9,
                    explanation="damage race is more important here",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT",
            available_commands=["play", "end"],
            choice_list=[],
            extras={
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("general", decision.comparator_profile)
        self.assertFalse(decision.fallback_recommended)
        self.assertIn("llm_battle_meta_advisor", decision.required_tools_used)

    def test_falls_back_when_profile_is_not_available(self):
        agent = BattleMetaAdvisorAgent(
            llm_provider=StubBattleMetaProvider(
                BattleMetaLlmProposal(
                    comparator_profile="transient",
                    confidence=0.95,
                    explanation="not a valid profile here",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT",
            available_commands=["play", "end"],
            choice_list=[],
            extras={
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("big_fight", decision.comparator_profile)
        self.assertEqual("llm_no_valid_profile", decision.metadata["fallback_reason"])

    def test_falls_back_when_confidence_is_too_low(self):
        agent = BattleMetaAdvisorAgent(
            llm_provider=StubBattleMetaProvider(
                BattleMetaLlmProposal(
                    comparator_profile="general",
                    confidence=0.4,
                    explanation="weak preference",
                )
            ),
            min_confidence=0.65,
        )
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT",
            available_commands=["play", "end"],
            choice_list=[],
            extras={
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("big_fight", decision.comparator_profile)
        self.assertEqual("low_confidence", decision.metadata["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
