import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.card_reward_advisor_agent import CardRewardAdvisorAgent
from rs.llm.agents.langgraph_base_agent import LangGraphBaseAgent
from rs.llm.agents.memory_langgraph_agent import MemoryAugmentedLangGraphAgent
from rs.llm.agents.shop_purchase_advisor_agent import ShopPurchaseAdvisorAgent
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProposal
from rs.llm.providers.shop_purchase_llm_provider import ShopPurchaseLlmProposal


class StubCardRewardProvider:
    def __init__(self, proposal: CardRewardLlmProposal):
        self._proposal = proposal
        self.seen_recent_decisions = []

    def propose(self, context: AgentContext) -> CardRewardLlmProposal:
        self.seen_recent_decisions.append(context.extras.get("recent_llm_decisions"))
        return self._proposal


class StubShopPurchaseProvider:
    def __init__(self, proposal: ShopPurchaseLlmProposal):
        self._proposal = proposal
        self.seen_recent_decisions = []

    def propose(self, context: AgentContext) -> ShopPurchaseLlmProposal:
        self.seen_recent_decisions.append(context.extras.get("recent_llm_decisions"))
        return self._proposal


class TestPhase2AdvisorAgents(unittest.TestCase):
    def test_card_reward_agent_uses_llm_proposal(self):
        provider = StubCardRewardProvider(
            CardRewardLlmProposal(
                proposed_command="choose 1",
                confidence=0.88,
                explanation="llm_pick",
            )
        )
        agent = CardRewardAdvisorAgent(llm_provider=provider)
        context = AgentContext(
            handler_name="CardRewardHandler",
            screen_type="CARD_REWARD",
            available_commands=["choose", "skip"],
            choice_list=["strike", "offering", "pommel strike"],
            extras={"recent_llm_decisions": "A1 F7 EventHandler -> choose 0 (0.80, took heal)"},
        )

        decision = agent.decide(context)

        self.assertEqual("choose 1", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)
        self.assertIsInstance(agent, LangGraphBaseAgent)
        self.assertIsInstance(agent, MemoryAugmentedLangGraphAgent)
        self.assertEqual(
            "A1 F7 EventHandler -> choose 0 (0.80, took heal)",
            provider.seen_recent_decisions[-1],
        )
        self.assertIn("langgraph_workflow", decision.required_tools_used)
        self.assertEqual("langgraph", decision.metadata["graph_runtime"])

    def test_card_reward_agent_falls_back_when_llm_has_no_decision(self):
        agent = CardRewardAdvisorAgent(
            llm_provider=StubCardRewardProvider(
                CardRewardLlmProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="no_decision",
                )
            )
        )
        context = AgentContext(
            handler_name="CardRewardHandler",
            screen_type="CARD_REWARD",
            available_commands=["choose", "skip"],
            choice_list=["strike", "pommel strike", "cleave"],
        )

        decision = agent.decide(context)

        self.assertIsNone(decision.proposed_command)
        self.assertTrue(decision.fallback_recommended)

    def test_shop_agent_uses_llm_proposal(self):
        provider = StubShopPurchaseProvider(
            ShopPurchaseLlmProposal(
                proposed_command="choose 0",
                confidence=0.77,
                explanation="llm_shop_pick",
            )
        )
        agent = ShopPurchaseAdvisorAgent(llm_provider=provider)
        context = AgentContext(
            handler_name="ShopPurchaseHandler",
            screen_type="SHOP_SCREEN",
            available_commands=["choose", "return"],
            choice_list=["purge", "bag of preparation"],
            extras={
                "has_removable_curse": True,
                "recent_llm_decisions": "A1 F8 CardRewardHandler -> choose 1 (0.77, strength scaling)",
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 0", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)
        self.assertIsInstance(agent, MemoryAugmentedLangGraphAgent)
        self.assertEqual(
            "A1 F8 CardRewardHandler -> choose 1 (0.77, strength scaling)",
            provider.seen_recent_decisions[-1],
        )
        self.assertIn("langgraph_workflow", decision.required_tools_used)
        self.assertEqual("langgraph", decision.metadata["graph_runtime"])


if __name__ == "__main__":
    unittest.main()
