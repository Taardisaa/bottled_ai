import argparse
import threading
import time
import traceback
from typing import Callable

from rs.api.client import Client
from rs.helper.logger import log, init_log, log_new_run_sequence
from rs.llm.config import load_llm_config
from rs.llm.langmem_service import get_langmem_service, shutdown_langmem_service
from rs.llm.run_context import DEFAULT_AGENT_IDENTITY
from rs.machine.character import Character
from rs.machine.game import Game
from rs.helper.seed import make_random_seed
from rs.utils.llm_utils import run_llm_preflight_check

# If there are run seeds, it will run them. Otherwise, it will use the run amount.
run_seeds = [
    # 'LGZ12EEMFGUK',
    "114514"
]

DEFAULT_RUN_AMOUNT = 1
DEFAULT_CHARACTER = Character.WATCHER
# DEFAULT_CHARACTER = Character.IRONCLAD
# DEFAULT_CHARACTER = Character.SILENT


def initialize_client_and_langmem(
        client_factory: Callable[[], Client] = Client,
        langmem_factory: Callable[[], object] = get_langmem_service,
) -> tuple[Client, object]:
    client = client_factory()
    log("before langmem init")
    langmem_service = langmem_factory()
    log("after langmem init")
    return client, langmem_service


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


def _assert_langmem_ready_or_fail(langmem_service: object) -> None:
    config = load_llm_config()
    if not config.langmem_enabled:
        return

    is_ready_callable = getattr(langmem_service, "is_ready", None)
    is_ready = bool(is_ready_callable()) if callable(is_ready_callable) else False
    if is_ready:
        return

    status_callable = getattr(langmem_service, "status", None)
    status = status_callable() if callable(status_callable) else "unknown"
    raise RuntimeError(
        "LangMem is enabled but not ready: "
        f"{status}. Fix embeddings setup or disable LangMem via LANGMEM_ENABLED=false."
    )


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
        client, langmem_service = initialize_client_and_langmem()
        log(f"LangMem status: {langmem_service.status()}")
        _assert_langmem_ready_or_fail(langmem_service)
        threading.Thread(
            target=_run_preflight_in_background,
            name="llm-preflight",
            daemon=True,
        ).start()
        game = Game(client, selected_character)
        if selected_seeds:
            for idx in range(selected_run_amount):
                for seed in selected_seeds:
                    try:
                        log(f"Running game {idx+1}/{selected_run_amount} with seed {seed}")
                        game.start(seed)
                        game.run()
                        time.sleep(1)
                    except Exception as e:
                        if "Error code: 502" in str(e):
                            log(f"502 error in game {idx+1} with seed {seed}, skipping rest of runs")
                            break
                        log(f"Exception in game {idx+1} with seed {seed}: " + str(e))
                        log(traceback.format_exc())
        else:
            for idx in range(selected_run_amount):
                try:
                    log(f"Running game {idx+1}/{selected_run_amount} with random seed")
                    game.start(make_random_seed())
                    game.run()
                    time.sleep(1)
                except Exception as e:
                    if "Error code: 502" in str(e):
                        log(f"502 error in game {idx+1} with seed {seed}, skipping rest of runs")
                        break
                    log(f"Exception in game {idx+1} with random seed: " + str(e))
                    log(traceback.format_exc())

    except Exception as e:
        log("Exception! " + str(e))
        log(traceback.format_exc())
    finally:
        shutdown_langmem_service(wait=True)
