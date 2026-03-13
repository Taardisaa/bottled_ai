from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.llm.benchmark_suite import (
    FIXED_LLM_BENCHMARK_SUITE,
    group_suite_by_strategy_key,
    summarize_benchmark_suite,
)


def main() -> int:
    summary = summarize_benchmark_suite(FIXED_LLM_BENCHMARK_SUITE)
    print("Fixed LLM benchmark suite summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    print("\nSeeds grouped by recommended strategy:")
    for strategy_key, seeds in sorted(group_suite_by_strategy_key(FIXED_LLM_BENCHMARK_SUITE).items()):
        print(f"- {strategy_key}: {' '.join(seeds)}")

    print("\nCases:")
    for case in FIXED_LLM_BENCHMARK_SUITE:
        print(
            f"- {case.case_id}: {case.character.value} {case.handler_area} "
            f"(Act {case.act}, Floor {case.floor}, Seed {case.seed})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
