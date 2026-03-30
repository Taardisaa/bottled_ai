import json
from typing import Optional

from rs.api.client import Client
from rs.helper.controller import await_controller
from rs.helper.logger import init_run_logging, log_to_run
from rs.helper.seed import get_seed_string
from rs.llm.agents.base_agent import AgentContext
from rs.llm.langmem_service import get_langmem_service
from rs.llm.run_context import DEFAULT_AGENT_IDENTITY, set_current_agent_identity
from rs.llm.runtime import get_ai_player_graph
from rs.machine.character import Character
from rs.machine.default_game_over import DefaultGameOverHandler
from rs.machine.handlers.default_cancel import DefaultCancelHandler
from rs.machine.handlers.default_choose import DefaultChooseHandler
from rs.machine.handlers.default_confirm import DefaultConfirmHandler
from rs.machine.handlers.default_end import DefaultEndHandler
from rs.machine.handlers.default_leave import DefaultLeaveHandler
from rs.machine.handlers.default_play import DefaultPlayHandler
from rs.machine.handlers.default_shop import DefaultShopHandler
from rs.machine.handlers.default_wait import DefaultWaitHandler
from rs.machine.state import GameState

DEFAULT_GAME_HANDLERS = [
    DefaultLeaveHandler(),
    DefaultShopHandler(),
    DefaultChooseHandler(),
    DefaultConfirmHandler(),
    DefaultPlayHandler(),
    DefaultEndHandler(),
    DefaultCancelHandler(),
    DefaultWaitHandler(),
]


class Game:

    def __init__(self, client: Client, character: Character):
        self.client = client
        self.character = character
        self.last_state: Optional[GameState] = None
        self.game_over_handler: DefaultGameOverHandler = DefaultGameOverHandler()

    def start(self, seed: str = ""):
        set_current_agent_identity(DEFAULT_AGENT_IDENTITY)
        self.run_elites = []
        self.last_elite = ""
        self.run_bosses = []
        self.last_boss = ""
        start_message = f"start {self.character.value}"
        if seed:
            start_message += " 0 " + seed
        self.__send_setup_command(start_message)
        state_seed = get_seed_string(self.last_state.game_state()['seed'])
        init_run_logging(state_seed)
        self.__send_command("choose 0")

    def run(self):
        log_to_run("Starting Run")
        try:
            while self.last_state.is_game_running():
                await_controller(self.last_state)
                self.__handle_state_based_logging()
                handled = False
                # Handle Game Over
                if self.game_over_handler.can_handle(self.last_state):
                    commands = self.game_over_handler.handle(
                        self.last_state,
                        self.run_elites,
                        self.run_bosses,
                        DEFAULT_AGENT_IDENTITY,
                    )
                    for command in commands:
                        self.__send_command(command)
                    break
                graph_commands = self.__decide_with_ai_player_graph()
                if graph_commands is not None:
                    for command in graph_commands:
                        self.__send_command(command)
                    handled = True
                    continue
                for handler in DEFAULT_GAME_HANDLERS:
                    if handler.can_handle(self.last_state):
                        log_to_run("Handler: " + str(handler))
                        action = handler.handle(self.last_state)
                        if not action:
                            continue
                        for command in action.commands:
                            self.__send_command(command)
                        handled = True
                        break
                if not handled:
                    log_to_run("Dying from not knowing what to do next")
                    raise Exception("ah I didn't know what to do!")
        finally:
            self.__finalize_langmem_run()

    def __send_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command)))

    def __send_silent_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command, silent=True)))

    def __send_setup_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command, before_run=True)))

    def __handle_state_based_logging(self):
        monsters = self.last_state.get_monsters()
        if self.last_state.game_state()['room_type'] == 'MonsterRoomElite':
            if monsters:
                self.last_elite = monsters[0]['name']
            elif self.last_elite:
                self.run_elites.append(self.last_elite)
                self.last_elite = ""
        if self.last_state.game_state()['room_type'] == 'MonsterRoomBoss':
            if monsters:
                self.last_boss = monsters[0]['name']
            elif self.last_boss:
                self.run_bosses.append(self.last_boss)
                self.last_boss = ""

    def __finalize_langmem_run(self):
        if self.last_state is None:
            return

        game_state = self.last_state.game_state()
        context = AgentContext(
            handler_name="RunFinalizer",
            screen_type=self.last_state.screen_type(),
            available_commands=[],
            choice_list=[],
            game_state={
                "floor": self.last_state.floor(),
                "act": game_state.get("act"),
                "character_class": game_state.get("class"),
            },
            extras={
                "run_id": f"{str(game_state.get('class', 'unknown')).strip().lower()}:{str(game_state.get('seed', 'unknown')).strip().lower()}",
                "agent_identity": DEFAULT_AGENT_IDENTITY,
                "run_memory_summary": (
                    f"{game_state.get('class', 'unknown')} act {game_state.get('act', 'unknown')} "
                    f"floor {self.last_state.floor()} hp {game_state.get('current_hp', 'unknown')}/"
                    f"{game_state.get('max_hp', 'unknown')} gold {game_state.get('gold', 'unknown')}"
                ),
            },
        )
        payload = {
            "floor": self.last_state.floor(),
            "score": self.last_state.screen_state().get("score"),
            "victory": not self.last_state.is_game_running(),
            "bosses": list(self.run_bosses),
            "elites": list(self.run_elites),
            "run_memory_summary": context.extras["run_memory_summary"],
        }
        get_langmem_service().finalize_run(context, payload)

    def __decide_with_ai_player_graph(self) -> list[str] | None:
        if self.last_state is None:
            return None

        ai_player_graph = get_ai_player_graph()
        if not ai_player_graph.is_enabled() or not ai_player_graph.can_handle(self.last_state):
            return None

        log_to_run("Handler: AIPlayerGraph")
        return ai_player_graph.decide(self.last_state)
