import argparse
import time
import traceback

from rs.ai import *
from rs.helper.seed import make_random_seed
from rs.api.client import Client
from rs.machine.game import Game
from rs.helper.logger import log, init_log, log_new_run_sequence

# If there are run seeds, it will run them. Otherwise, it will use the run amount.
run_seeds = [
    #'LGZ12EEMFGUK',
]
DEFAULT_RUN_AMOUNT = 1
DEFAULT_STRATEGY = PEACEFUL_PUMMELING


STRATEGIES = {
    "peaceful_pummeling": PEACEFUL_PUMMELING,
    "requested_strike": REQUESTED_STRIKE,
    "pwnder_my_orbs": PWNDER_MY_ORBS,
    "claw_is_law": CLAW_IS_LAW,
    "shivs_and_giggles": SHIVS_AND_GIGGLES,
    "smart_agent": SMART_AGENT,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bottled AI strategy loop.")
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES.keys()),
        default="peaceful_pummeling",
        help="Strategy to run.",
    )
    parser.add_argument(
        "--run-amount",
        type=int,
        default=DEFAULT_RUN_AMOUNT,
        help="Number of runs when no explicit seeds are provided.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=None,
        help="Run a specific seed (repeat flag for multiple seeds).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    selected_strategy = STRATEGIES[args.strategy]
    selected_seeds = args.seed if args.seed is not None else run_seeds
    selected_run_amount = args.run_amount

    init_log()
    log("Starting up")
    log(f"Selected strategy: {selected_strategy.name}")
    log_new_run_sequence()
    try:
        client = Client()
        game = Game(client, selected_strategy)
        if selected_seeds:
            for seed in selected_seeds:
                game.start(seed)
                game.run()
                time.sleep(1)
        else:
            for _ in range(selected_run_amount):
                game.start(make_random_seed())
                game.run()
                time.sleep(1)

    except Exception as e:
        log("Exception! " + str(e))
        log(traceback.format_exc())
