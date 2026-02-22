from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger as loguru_logger

from definitions import ROOT_DIR
from rs.calculator.enums.card_id import CardId
from rs.calculator.interfaces.memory_items import MemoryItem, ResetSchedule
from rs.helper.seed import get_seed_string
from rs.machine.state import GameState

current_run_log_file: str = ''
current_run_log_count: int = 0
current_run_calculator_missing_relics: set[str] = set()
current_run_calculator_missing_potions: set[str] = set()
current_run_calculator_missing_powers: set[str] = set()
current_run_calculator_missing_cards: set[str] = set()
current_run_missing_events: set[str] = set()

LOG_DIR = Path(ROOT_DIR) / "logs"


def _log_path(filename: str) -> Path:
    path = LOG_DIR / f"{filename}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_log_line(filename: str, message: str) -> None:
    with _log_path(filename).open("a+", encoding="utf-8") as f:
        f.write(message + "\n")


def init_run_logging(seed: str):
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    global current_run_log_file
    global current_run_log_count
    global current_run_calculator_missing_relics
    global current_run_calculator_missing_powers
    global current_run_calculator_missing_cards
    global current_run_calculator_missing_potions
    global current_run_missing_events
    current_run_log_count = 0
    current_run_log_file = "runs/" + dt + "--" + seed
    _log_path(current_run_log_file).touch(exist_ok=False)
    log("Seed: " + seed, "calculator_missing_enums")
    current_run_calculator_missing_relics = set()
    current_run_calculator_missing_powers = set()
    current_run_calculator_missing_cards = set()
    current_run_calculator_missing_potions = set()
    current_run_missing_events = set()


def log_to_run(message: str):
    if not current_run_log_file:
        return

    global current_run_log_count
    current_run_log_count += 1
    if current_run_log_count > 10000:
        log("Dying due to this seeming to be stuck", current_run_log_file)
        raise Exception("Dying due to this seeming to be stuck...")
    log(message, current_run_log_file)


def log_calculator_missing_relic(relic_id: str):
    global current_run_calculator_missing_relics
    current_run_calculator_missing_relics.add(relic_id)


def log_calculator_missing_card(card_id: str):
    global current_run_calculator_missing_cards
    current_run_calculator_missing_cards.add(card_id)


def log_calculator_missing_power(power_id: str):
    global current_run_calculator_missing_powers
    current_run_calculator_missing_powers.add(power_id)


def log_calculator_missing_potion(potion_id: str):
    global current_run_calculator_missing_potions
    current_run_calculator_missing_potions.add(potion_id)


def log_missing_event(event_name: str):
    global current_run_missing_events
    current_run_missing_events.add(event_name)


def log_missing_calculator_enums_to_run():
    global current_run_calculator_missing_relics
    global current_run_calculator_missing_powers
    global current_run_calculator_missing_cards
    global current_run_calculator_missing_potions
    global current_run_missing_events
    log(f"Missing relic names:{','.join(current_run_calculator_missing_relics)}", "calculator_missing_enums")
    log(f"Missing power ids:{','.join(current_run_calculator_missing_powers)}", "calculator_missing_enums")
    log(f"Missing card ids:{','.join(current_run_calculator_missing_cards)}", "calculator_missing_enums")
    # log(f"Missing potion names:{','.join(current_run_calculator_missing_potions)}", "calculator_missing_enums")
    log(f"Missing event names:{','.join(current_run_missing_events)}", "calculator_missing_enums")


def log_run_results(state: GameState, elites: List[str], bosses: List[str], strategy_name: str):
    message = "Seed:" + get_seed_string(state.game_state()['seed'])
    message += ", Floor:" + str(state.floor())
    message += ", Score:" + str(state.game_state()['screen_state']['score'])
    message += ", Strat: " + strategy_name
    message += ", DiedTo: "
    if state.get_monsters():
        for m in state.get_monsters():
            message += m["name"] + ","
    else:
        message += "N/A,"
    message += " Bosses: " + ",".join(bosses)
    message += " Elites: " + ",".join(elites)
    message += " Relics: "
    for r in state.get_relics():
        message += r["name"] + ","
    if state.memory_general[MemoryItem.KILLED_WITH_LESSON_LEARNED] > 0:
        message += " Killed with Lesson Learned: " + str(state.memory_general[MemoryItem.KILLED_WITH_LESSON_LEARNED])
    if sum(state.memory_by_card[CardId.RITUAL_DAGGER][ResetSchedule.GAME].values()) > 10:
        message += " Extraordinary amount of Ritual Dagger power: " + str(sum(state.memory_by_card[CardId.RITUAL_DAGGER][ResetSchedule.GAME].values()))
    _append_log_line("run_history", message)


def log_new_run_sequence():
    _append_log_line("run_history", "-------------------------")


def log(message: str, filename: str = "default"):
    _append_log_line(filename, message)
    loguru_logger.info(message)


def init_log(filename: str = "default"):
    with _log_path(filename).open("w", encoding="utf-8"):
        pass
