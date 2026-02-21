import time
import traceback

from rs.ai import CLAW_IS_LAW, PEACEFUL_PUMMELING, PWNDER_MY_ORBS, REQUESTED_STRIKE, SHIVS_AND_GIGGLES
from rs.helper.seed import make_random_seed
from rs.api.client import Client
from rs.machine.game import Game
from rs.helper.logger import log, init_log, log_new_run_sequence

# If there are run seeds, it will run them. Otherwise, it will use the run amount.
run_seeds = [
    #'LGZ12EEMFGUK',
]
run_amount = 1
strategy = PEACEFUL_PUMMELING

if __name__ == "__main__":
    init_log()
    log("Starting up")
    log_new_run_sequence()
    try:
        client = Client()
        game = Game(client, strategy)
        if run_seeds:
            for seed in run_seeds:
                game.start(seed)
                game.run()
                time.sleep(1)
        else:
            for i in range(run_amount):
                game.start(make_random_seed())
                game.run()
                time.sleep(1)

    except Exception as e:
        log("Exception! " + str(e))
        log(traceback.format_exc())
