import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.battle_meta_advisor_agent import BattleMetaAdvisorAgent
from rs.llm.decision_memory import DecisionMemoryStore
from rs.llm.providers.battle_meta_llm_provider import BattleMetaLlmProposal


class StubBattleMetaProvider:
    def __init__(self, proposal: BattleMetaLlmProposal):
        self._proposal = proposal
        self.seen_recent_decisions = []

    def propose(self, context: AgentContext) -> BattleMetaLlmProposal:
        self.seen_recent_decisions.append(context.extras.get("recent_llm_decisions"))
        return self._proposal


class TestBattleMetaAdvisorAgent(unittest.TestCase):
    def test_uses_valid_high_confidence_profile(self):
        provider = StubBattleMetaProvider(
            BattleMetaLlmProposal(
                comparator_profile="general",
                confidence=0.9,
                explanation="damage race is more important here",
            )
        )
        agent = BattleMetaAdvisorAgent(llm_provider=provider)
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT",
            available_commands=["play", "end"],
            choice_list=[],
            extras={
                "run_id": "ironclad:seed1",
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("general", decision.comparator_profile)
        self.assertFalse(decision.fallback_recommended)
        self.assertIn("llm_battle_meta_advisor", decision.required_tools_used)
        self.assertIn("langgraph_workflow", decision.required_tools_used)
        self.assertEqual("langgraph", decision.metadata["graph_runtime"])
        self.assertEqual("none", provider.seen_recent_decisions[-1])

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
        self.assertTrue(decision.fallback_recommended)

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
        self.assertTrue(decision.fallback_recommended)

    def test_recent_battle_profiles_are_reused_for_same_run(self):
        provider = StubBattleMetaProvider(
            BattleMetaLlmProposal(
                comparator_profile="general",
                confidence=0.9,
                explanation="damage race is more important here",
            )
        )
        agent = BattleMetaAdvisorAgent(
            llm_provider=provider,
            memory_store=DecisionMemoryStore(),
        )
        context = AgentContext(
            handler_name="CommonBattleHandler",
            screen_type="COMBAT",
            available_commands=["play", "end"],
            choice_list=[],
            game_state={"floor": 7, "act": 1},
            extras={
                "run_id": "ironclad:seed1",
                "deterministic_profile": "big_fight",
                "available_profiles": ["big_fight", "general"],
            },
        )

        first = agent.decide(context)
        second = agent.decide(context)

        self.assertEqual("general", first.comparator_profile)
        self.assertEqual("general", second.comparator_profile)
        self.assertIn("profile general", provider.seen_recent_decisions[-1])


if __name__ == "__main__":
    unittest.main()
