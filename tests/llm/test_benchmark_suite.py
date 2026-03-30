import unittest

from rs.llm.benchmark_suite import (
    FIXED_LLM_BENCHMARK_SUITE,
    group_suite_by_agent_identity,
    get_fixed_llm_benchmark_suite,
    summarize_benchmark_suite,
)
from rs.machine.character import Character


class TestBenchmarkSuite(unittest.TestCase):
    def test_suite_contains_current_llm_handler_areas(self):
        handler_areas = {case.handler_area for case in FIXED_LLM_BENCHMARK_SUITE}

        self.assertIn("event", handler_areas)
        self.assertIn("shop", handler_areas)
        self.assertIn("card_reward", handler_areas)

    def test_suite_summary_reports_current_coverage_and_gap(self):
        summary = summarize_benchmark_suite()

        self.assertGreaterEqual(summary["total_cases"], 8)
        self.assertIn("Ironclad", summary["characters_covered"])
        self.assertIn("Silent", summary["characters_covered"])
        self.assertIn("Watcher", summary["characters_covered"])
        self.assertIn("Defect", summary["characters_covered"])
        self.assertEqual([], summary["missing_characters"])
        self.assertIn("phase_2", summary["phases_covered"])

    def test_filtering_excludes_future_phase_cases_when_requested(self):
        current_cases = get_fixed_llm_benchmark_suite(include_future_phases=False)

        self.assertTrue(current_cases)
        self.assertTrue(all(not case.phase.endswith("_future") for case in current_cases))
        self.assertTrue(all(case.handler_area in {"event", "shop", "card_reward"} for case in current_cases))

    def test_filtering_by_handler_area_and_character(self):
        watcher_shop_cases = get_fixed_llm_benchmark_suite(
            handler_areas=["shop"],
            characters=[Character.WATCHER],
        )

        self.assertTrue(watcher_shop_cases)
        self.assertTrue(all(case.handler_area == "shop" for case in watcher_shop_cases))
        self.assertTrue(all(case.character == Character.WATCHER for case in watcher_shop_cases))

    def test_grouping_by_agent_identity_collects_seed_strings(self):
        grouped = group_suite_by_agent_identity(get_fixed_llm_benchmark_suite(include_future_phases=False))

        self.assertIn("neo_primates", grouped)
        self.assertTrue(all(isinstance(seed, str) and seed for seeds in grouped.values() for seed in seeds))


if __name__ == "__main__":
    unittest.main()
