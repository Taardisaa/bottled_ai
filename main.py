import argparse
import threading
import time
import traceback

from rs.api.client import Client
from rs.helper.logger import log, init_log, log_new_run_sequence
from rs.llm.langmem_service import get_langmem_service, shutdown_langmem_service
from rs.llm.run_context import DEFAULT_AGENT_IDENTITY
from rs.machine.character import Character
from rs.machine.game import Game
from rs.helper.seed import make_random_seed
from rs.utils.llm_utils import run_llm_preflight_check

# If there are run seeds, it will run them. Otherwise, it will use the run amount.
run_seeds = [
    #'LGZ12EEMFGUK',
]
DEFAULT_RUN_AMOUNT = 1
DEFAULT_CHARACTER = Character.WATCHER


def _run_preflight_in_background() -> None:
    llm_preflight = run_llm_preflight_check()
    if llm_preflight.available:
        log(
            "LLM preflight succeeded: "
            f"requested_model={llm_preflight.requested_model}, "
            f"response_model={llm_preflight.response_model}, "
            f"provider={llm_preflight.provider}, "
            f"endpoint={llm_preflight.endpoint}, "
            f"max_tokens={llm_preflight.max_tokens}, "
            f"total_tokens={llm_preflight.total_tokens}, "
            f"preview={llm_preflight.response_preview}"
        )
    else:
        log(
            "LLM preflight failed: "
            f"requested_model={llm_preflight.requested_model}, "
            f"provider={llm_preflight.provider}, "
            f"endpoint={llm_preflight.endpoint}, "
            f"max_tokens={llm_preflight.max_tokens}, "
            f"error={llm_preflight.error}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bottled AI agent loop.")
    parser.add_argument(
        "--character",
        choices=sorted(character.name.lower() for character in Character),
        default=DEFAULT_CHARACTER.name.lower(),
        help="Character to run.",
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
    selected_character = Character[args.character.upper()]
    selected_seeds = args.seed if args.seed is not None else run_seeds
    selected_run_amount = args.run_amount

    init_log()
    log("Starting up")
    log(f"Selected character: {selected_character.value}")
    log(f"Agent identity: {DEFAULT_AGENT_IDENTITY}")
    log_new_run_sequence()
    try:
        client = Client()
        log("before langmem init")
        langmem_service = get_langmem_service()
        log("after langmem init")
        log(f"LangMem status: {langmem_service.status()}")
        threading.Thread(
            target=_run_preflight_in_background,
            name="llm-preflight",
            daemon=True,
        ).start()
        game = Game(client, selected_character)
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
    finally:
        shutdown_langmem_service(wait=True)
