import unittest

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.decision_memory import DecisionMemoryStore


class TestDecisionMemoryStore(unittest.TestCase):
    def test_recent_decisions_are_scoped_to_run_id(self):
        store = DecisionMemoryStore()
        first_context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
            game_state={"floor": 5, "act": 1},
            extras={"run_id": "watcher:seed1"},
        )
        second_context = AgentContext(
            handler_name="ShopPurchaseHandler",
            screen_type="SHOP_SCREEN",
            available_commands=["choose"],
            choice_list=["a"],
            game_state={"floor": 6, "act": 1},
            extras={"run_id": "watcher:seed2"},
        )

        store.record(
            first_context,
            AgentDecision(
                proposed_command="choose 0",
                confidence=0.9,
                explanation="take the event reward",
            ),
        )

        self.assertIn("EventHandler -> choose 0", store.build_recent_decisions_summary(first_context))
        self.assertEqual("none", store.build_recent_decisions_summary(second_context))

    def test_fallback_decisions_are_not_recorded(self):
        store = DecisionMemoryStore()
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
            game_state={"floor": 5, "act": 1},
            extras={"run_id": "watcher:seed1"},
        )

        store.record(
            context,
            AgentDecision(
                proposed_command="choose 0",
                confidence=0.2,
                explanation="unsafe",
                fallback_recommended=True,
            ),
        )

        self.assertEqual("none", store.build_recent_decisions_summary(context))
