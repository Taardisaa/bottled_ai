import copy
import json
from typing import Any, Optional

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


class _GameBattleRuntimeAdapter:
    def __init__(self, game: "Game"):
        self._game = game

    def current_state(self) -> GameState:
        if self._game.last_state is None:
            raise ValueError("battle runtime requested state before the game produced one")
        return self._game.last_state

    def execute(self, commands: list[str]) -> GameState:
        for command in commands:
            self._game._execute_runtime_command(command)
        if self._game.last_state is None:
            raise ValueError("battle runtime executed commands but produced no state")
        return self._game.last_state


class Game:

    def __init__(self, client: Client, character: Character):
        self.client = client
        self.character = character
        self.last_state: Optional[GameState] = None
        self.game_over_handler: DefaultGameOverHandler = DefaultGameOverHandler()
        # Communication Mod may omit ``game_state`` after run teardown (e.g. after game-over ``proceed``).
        self._langmem_last_game_state: Optional[dict[str, Any]] = None

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
                self._capture_langmem_game_state_snapshot()
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
                graph_handled = self.__decide_with_ai_player_graph()
                if graph_handled:
                    handled = True
                    continue
                # for handler in DEFAULT_GAME_HANDLERS:
                #     if handler.can_handle(self.last_state):
                #         log_to_run("Handler: " + str(handler))
                #         action = handler.handle(self.last_state)
                #         if not action:
                #             continue
                #         for command in action.commands:
                #             self.__send_command(command)
                #         handled = True
                #         break
                if not handled:
                    log_to_run("Dying from not knowing what to do next")
                    raise Exception("ah I didn't know what to do!")
        finally:
            self.__finalize_langmem_run()

    def __send_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command)))
        self._capture_langmem_game_state_snapshot()

    def __send_silent_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command, silent=True)))
        self._capture_langmem_game_state_snapshot()

    def __send_setup_command(self, command: str):
        self.last_state = GameState(json.loads(self.client.send_message(command, before_run=True)))
        self._capture_langmem_game_state_snapshot()

    def _execute_runtime_command(self, command: str):
        self.__send_command(command)

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

    def _capture_langmem_game_state_snapshot(self) -> None:
        if self.last_state is None:
            return
        state_json = getattr(self.last_state, "json", None)
        if not isinstance(state_json, dict):
            return
        raw = state_json.get("game_state")
        if isinstance(raw, dict):
            self._langmem_last_game_state = copy.deepcopy(raw)

    def __finalize_langmem_run(self):
        if self.last_state is None:
            return

        raw_state = getattr(self.last_state, "json", None)
        if not isinstance(raw_state, dict):
            return
        game_state = raw_state.get("game_state")
        if not isinstance(game_state, dict):
            game_state = self._langmem_last_game_state
        if not isinstance(game_state, dict):
            log_to_run(
                "Skipping LangMem finalization: final payload has no game_state and no snapshot was captured."
            )
            return

        floor = game_state.get("floor")
        screen_type = str(game_state.get("screen_type", "UNKNOWN"))
        screen_state = game_state.get("screen_state", {})
        score = screen_state.get("score") if isinstance(screen_state, dict) else None

        log_to_run(
            "Run finished: "
            f"floor={floor if floor is not None else 'unknown'}, "
            f"score={score if score is not None else 'unknown'}"
        )

        context = AgentContext(
            handler_name="RunFinalizer",
            screen_type=screen_type,
            available_commands=[],
            choice_list=[],
            game_state={
                "floor": floor,
                "act": game_state.get("act"),
                "character_class": game_state.get("class"),
            },
            extras={
                "run_id": f"{str(game_state.get('class', 'unknown')).strip().lower()}:{str(game_state.get('seed', 'unknown')).strip().lower()}",
                "agent_identity": DEFAULT_AGENT_IDENTITY,
                "run_memory_summary": (
                    f"{game_state.get('class', 'unknown')} act {game_state.get('act', 'unknown')} "
                    f"floor {floor if floor is not None else 'unknown'} hp {game_state.get('current_hp', 'unknown')}/"
                    f"{game_state.get('max_hp', 'unknown')} gold {game_state.get('gold', 'unknown')}"
                ),
            },
        )
        payload = {
            "floor": floor,
            "score": score,
            "bosses": list(self.run_bosses),
            "elites": list(self.run_elites),
            "run_memory_summary": context.extras["run_memory_summary"],
        }
        get_langmem_service().finalize_run(context, payload)

    def __decide_with_ai_player_graph(self) -> bool:
        if self.last_state is None:
            return False

        ai_player_graph = get_ai_player_graph()
        if not ai_player_graph.is_enabled() or not ai_player_graph.can_handle(self.last_state):
            return False

        log_to_run("Handler: AIPlayerGraph")
        result = ai_player_graph.execute(self.last_state, runtime=_GameBattleRuntimeAdapter(self))
        if result is None or not result.handled:
            return False
        if result.final_state is not None:
            self.last_state = result.final_state
            self._capture_langmem_game_state_snapshot()

        for command in result.commands or []:
            self.__send_command(command)
        return True
