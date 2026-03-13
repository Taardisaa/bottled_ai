import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.langgraph_base_agent import LangGraphBaseAgent
from rs.llm.agents.map_advisor_agent import MapAdvisorAgent
from rs.llm.agents.memory_langgraph_agent import MemoryAugmentedLangGraphAgent
from rs.llm.providers.map_llm_provider import MapLlmProposal


class StubMapProvider:
    def __init__(self, proposal: MapLlmProposal):
        self._proposal = proposal
        self.seen_recent_decisions = []

    def propose(self, context: AgentContext) -> MapLlmProposal:
        self.seen_recent_decisions.append(context.extras.get("recent_llm_decisions"))
        return self._proposal


class TestMapAdvisorAgent(unittest.TestCase):
    def test_map_agent_uses_llm_proposal(self):
        provider = StubMapProvider(
            MapLlmProposal(
                proposed_command="choose 2",
                confidence=0.81,
                explanation="take the greedier path",
            )
        )
        agent = MapAdvisorAgent(llm_provider=provider)
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose", "return"],
            choice_list=["x=0", "x=3", "x=4", "x=6"],
            extras={
                "deterministic_best_command": "choose 3",
                "recent_llm_decisions": "A1 F8 ShopPurchaseHandler -> choose 1 (0.74, remove curse)",
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 2", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)
        self.assertIn("deterministic_path_scores", decision.required_tools_used)
        self.assertIn("langgraph_workflow", decision.required_tools_used)
        self.assertEqual("langgraph", decision.metadata["graph_runtime"])
        self.assertEqual(
            "A1 F8 ShopPurchaseHandler -> choose 1 (0.74, remove curse)",
            provider.seen_recent_decisions[-1],
        )
        self.assertIsInstance(agent, LangGraphBaseAgent)
        self.assertIsInstance(agent, MemoryAugmentedLangGraphAgent)

    def test_map_agent_uses_rule_based_deterministic_choice_when_llm_has_no_decision(self):
        agent = MapAdvisorAgent(
            llm_provider=StubMapProvider(
                MapLlmProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="no_decision",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose", "return"],
            choice_list=["x=0", "x=3", "x=4", "x=6"],
            extras={
                "deterministic_best_command": "choose 3",
                "choice_path_overviews": [
                    {
                        "choice_label": "x=6",
                        "choice_command": "choose 3",
                        "reward_survivability": 5.1,
                        "survivability": 1.0,
                        "room_counts": {"E": 1},
                    },
                ],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 3", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)

    def test_map_agent_prefers_safer_path_when_low_hp(self):
        agent = MapAdvisorAgent(
            llm_provider=StubMapProvider(
                MapLlmProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="no_decision",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose"],
            choice_list=["x=0", "x=1"],
            game_state={"current_hp": 20, "max_hp": 80, "gold": 80, "act": 2, "floor": 20},
            extras={
                "deterministic_best_command": "choose 0",
                "choice_path_overviews": [
                    {
                        "choice_label": "x=0",
                        "choice_command": "choose 0",
                        "reward_survivability": 12.0,
                        "survivability": 0.72,
                        "room_counts": {"E": 1},
                    },
                    {
                        "choice_label": "x=1",
                        "choice_command": "choose 1",
                        "reward_survivability": 11.4,
                        "survivability": 0.85,
                        "room_counts": {"E": 0},
                    },
                ],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 1", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)

    def test_map_agent_prefers_early_shop_when_rich_and_healthy(self):
        agent = MapAdvisorAgent(
            llm_provider=StubMapProvider(
                MapLlmProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="no_decision",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose"],
            choice_list=["x=0", "x=1"],
            game_state={"current_hp": 70, "max_hp": 80, "gold": 220, "act": 2, "floor": 18},
            extras={
                "deterministic_best_command": "choose 0",
                "choice_path_overviews": [
                    {
                        "choice_label": "x=0",
                        "choice_command": "choose 0",
                        "reward_survivability": 12.2,
                        "survivability": 1.0,
                        "room_counts": {"E": 1},
                        "shop_distance": None,
                    },
                    {
                        "choice_label": "x=1",
                        "choice_command": "choose 1",
                        "reward_survivability": 11.6,
                        "survivability": 0.98,
                        "room_counts": {"E": 1},
                        "shop_distance": 1,
                    },
                ],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 1", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)

    def test_map_agent_prefers_elite_path_early_when_healthy(self):
        agent = MapAdvisorAgent(
            llm_provider=StubMapProvider(
                MapLlmProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="no_decision",
                )
            )
        )
        context = AgentContext(
            handler_name="CommonMapHandler",
            screen_type="MAP",
            available_commands=["choose"],
            choice_list=["x=0", "x=1"],
            game_state={"current_hp": 80, "max_hp": 80, "gold": 90, "act": 1, "floor": 3},
            extras={
                "deterministic_best_command": "choose 0",
                "choice_path_overviews": [
                    {
                        "choice_label": "x=0",
                        "choice_command": "choose 0",
                        "reward_survivability": 10.8,
                        "survivability": 1.0,
                        "room_counts": {"E": 0},
                    },
                    {
                        "choice_label": "x=1",
                        "choice_command": "choose 1",
                        "reward_survivability": 10.5,
                        "survivability": 0.95,
                        "room_counts": {"E": 1},
                    },
                ],
            },
        )

        decision = agent.decide(context)

        self.assertEqual("choose 1", decision.proposed_command)
        self.assertFalse(decision.fallback_recommended)


if __name__ == "__main__":
    unittest.main()
