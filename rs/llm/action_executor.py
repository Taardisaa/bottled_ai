from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.helper.logger import log_to_run
from rs.llm.battle_runtime import BattleRuntimeAdapter, BattleSessionResult
from rs.llm.reward_subagent import RewardSessionResult
from rs.machine.state import GameState


@dataclass
class ActionExecutionResult:
    handled: bool
    final_state: GameState | None = None
    commands: list[str] | None = None
    steps: int = 0
    summary: str = ""
    battle_session: BattleSessionResult | None = None
    reward_session: RewardSessionResult | None = None


class RuntimeRunner(Protocol):
    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> ActionExecutionResult:
        ...


@dataclass
class ExecutorConfig:
    max_steps: int = 12
    no_progress_limit: int = 2


class UnifiedActionExecutor:
    def __init__(
            self,
            *,
            battle_runner: RuntimeRunner,
            combat_reward_runner: RuntimeRunner,
            boss_reward_runner: RuntimeRunner,
            astrolabe_runner: RuntimeRunner,
            campfire_runner: RuntimeRunner,
            grid_select_runner: RuntimeRunner,
            context_builder,
            config: ExecutorConfig | None = None,
    ):
        self._battle_runner = battle_runner
        self._combat_reward_runner = combat_reward_runner
        self._boss_reward_runner = boss_reward_runner
        self._astrolabe_runner = astrolabe_runner
        self._campfire_runner = campfire_runner
        self._grid_select_runner = grid_select_runner
        self._context_builder = context_builder
        self._config = ExecutorConfig() if config is None else config

    def execute(self, state: GameState, runtime: BattleRuntimeAdapter) -> ActionExecutionResult:
        context = self._context_builder(state)
        if context is None:
            return ActionExecutionResult(handled=False, final_state=runtime.current_state())

        handler_name = context.handler_name
        log_to_run(f"UnifiedActionExecutor route: {handler_name}")
        if handler_name == "BattleHandler":
            result = self._battle_runner.run(state, runtime)
            return result
        if handler_name == "CombatRewardHandler":
            return self._combat_reward_runner.run(state, runtime)
        if handler_name == "BossRewardHandler":
            return self._boss_reward_runner.run(state, runtime)
        if handler_name == "AstrolabeTransformHandler":
            return self._astrolabe_runner.run(state, runtime)
        if handler_name == "CampfireHandler":
            return self._campfire_runner.run(state, runtime)
        if handler_name == "GridSelectHandler":
            return self._grid_select_runner.run(state, runtime)

        # Non-subagent handlers (event/map/shop/card reward/generic) should
        # still go through graph LLM decision flow, then execute returned commands.
        return self._battle_runner.run(state, runtime)
