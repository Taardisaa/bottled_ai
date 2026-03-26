from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.llm.offline_benchmark import run_offline_llm_benchmark_suite, summarize_offline_benchmark_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline LLM benchmark suite against saved fixtures.")
    parser.add_argument(
        "--include-future-phases",
        action="store_true",
        help="Include benchmark cases still tagged as future-phase placeholders.",
    )
    args = parser.parse_args()

    results = run_offline_llm_benchmark_suite(include_future_phases=args.include_future_phases)
    summary = summarize_offline_benchmark_results(results)

    print("Offline LLM benchmark summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    print("\nCase results:")
    for result in results:
        print(
            f"- {result.case_id}: {result.handler_area} "
            f"baseline={result.baseline_commands} assisted={result.assisted_commands} "
            f"advisor_output={result.advisor_output!r} changed={result.changed} "
            f"fallback={result.fallback_used}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
