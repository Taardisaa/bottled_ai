import unittest

from rs.helper.seed import get_seed_string
from rs.llm.benchmark_suite import get_fixed_llm_benchmark_suite, load_benchmark_case_state
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.llm.integration.event_context import build_event_agent_context
from rs.llm.integration.shop_purchase_context import build_shop_purchase_agent_context


class TestBenchmarkReplaySmoke(unittest.TestCase):
    def test_current_phase_benchmark_cases_build_llm_contexts(self):
        builders = {
            "event": build_event_agent_context,
            "shop": build_shop_purchase_agent_context,
            "card_reward": build_card_reward_agent_context,
        }
        current_cases = get_fixed_llm_benchmark_suite(include_future_phases=False)

        for case in current_cases:
            with self.subTest(case_id=case.case_id):
                state = load_benchmark_case_state(case)

                self.assertEqual(case.seed, get_seed_string(state.game_state()["seed"]))
                self.assertEqual(case.character.value.upper(), state.game_state()["class"].replace("THE_", ""))

                builder = builders[case.handler_area]
                context = builder(state, f"{case.handler_area}_benchmark_smoke")

                self.assertEqual(state.screen_type(), context.screen_type)
                self.assertTrue(context.available_commands)
                self.assertEqual(int(state.floor()), context.game_state["floor"])
                self.assertEqual(int(state.game_state()["act"]), context.game_state["act"])
                self.assertEqual(state.get_choice_list(), context.choice_list)

    def test_future_phase_benchmark_cases_still_load_cleanly(self):
        future_cases = [
            case for case in get_fixed_llm_benchmark_suite()
            if case.phase.endswith("_future")
        ]

        self.assertTrue(future_cases)
        for case in future_cases:
            with self.subTest(case_id=case.case_id):
                state = load_benchmark_case_state(case)

                self.assertEqual(case.seed, get_seed_string(state.game_state()["seed"]))
                self.assertEqual(int(state.floor()), case.floor)
                self.assertEqual(int(state.game_state()["act"]), case.act)
                self.assertEqual(state.game_state()["room_type"], case.room_type)


if __name__ == "__main__":
    unittest.main()
