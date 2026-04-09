from __future__ import annotations

import uuid
from typing import Any

from rs.ai.peaceful_pummeling.handlers.potions_handler import PotionsBossHandler, PotionsEliteHandler
from rs.common.handlers.common_battle_handler import CommonBattleHandler
from rs.common.handlers.common_mass_discard_handler import CommonMassDiscardHandler
from rs.common.handlers.common_scry_handler import CommonScryHandler
from rs.helper.logger import log_to_run
from rs.llm.battle_runtime import BattleRuntimeAdapter
from rs.llm.battle_subagent import BattleSessionResult, BattleSubagentConfig
from rs.llm.integration.battle_context import is_battle_scope_state
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.machine.command import Command
from rs.machine.state import GameState


class DeterministicBattleSubagent:
    def __init__(
            self,
            *,
            config: BattleSubagentConfig | None = None,
            langmem_service: LangMemService | None = None,
            grid_select_subagent: Any | None = None,
    ):
        self._config = BattleSubagentConfig() if config is None else config
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._grid_select_subagent = grid_select_subagent
        self._battle_handler = CommonBattleHandler(max_path_count=11_000)
        self._potion_handlers = [PotionsBossHandler(), PotionsEliteHandler()]
        self._scry_handler = CommonScryHandler()
        self._mass_discard_handler = CommonMassDiscardHandler()

    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> BattleSessionResult:
        session_id = uuid.uuid4().hex[:12]
        log_to_run(f"DeterministicBattle session started: {session_id}")

        current_state = state
        executed_batches: list[list[str]] = []
        last_signature = ""
        no_progress_count = 0
        max_steps = self._config.max_decision_loops * 4

        for _ in range(max_steps):
            if not is_battle_scope_state(current_state):
                break

            screen = current_state.screen_type()
            signature = self._state_signature(current_state)

            # No-progress guardrail
            if signature == last_signature:
                no_progress_count += 1
                if no_progress_count >= self._config.no_progress_limit:
                    log_to_run("DeterministicBattle no-progress guardrail — forcing end")
                    if current_state.has_command(Command.END):
                        current_state = runtime.execute(["end"])
                        executed_batches.append(["end"])
                    last_signature = ""
                    no_progress_count = 0
                    continue
            else:
                no_progress_count = 0
            last_signature = signature

            # 1. Scry action (GRID with ScryAction)
            if self._scry_handler.can_handle(current_state):
                action = self._scry_handler.handle(current_state)
                if action and action.commands:
                    log_to_run(f"DeterministicBattle scry: {action.commands}")
                    current_state = runtime.execute(action.commands)
                    executed_batches.append(action.commands)
                    continue

            # 2. Mass discard (HAND_SELECT with can_pick_zero)
            if self._mass_discard_handler.can_handle(current_state):
                action = self._mass_discard_handler.handle(current_state)
                if action and action.commands:
                    log_to_run(f"DeterministicBattle discard: {action.commands}")
                    current_state = runtime.execute(action.commands)
                    executed_batches.append(action.commands)
                    continue

            # 3. Other HAND_SELECT / GRID screens — dispatch to grid select subagent (LLM)
            if screen in ("HAND_SELECT", "GRID"):
                if self._grid_select_subagent is not None:
                    log_to_run(f"DeterministicBattle dispatching to grid select subagent: screen={screen}")
                    session = self._grid_select_subagent.run(current_state, runtime)
                    current_state = session.final_state or runtime.current_state()
                    if session.executed_commands:
                        executed_batches.extend(session.executed_commands)
                    continue
                # No subagent available — basic fallback
                available = current_state.game_state().get("available_commands", [])
                if "choose" in available:
                    log_to_run("DeterministicBattle selection fallback: choose 0")
                    current_state = runtime.execute(["choose 0"])
                    executed_batches.append(["choose 0"])
                elif "confirm" in available:
                    current_state = runtime.execute(["confirm", "wait 30"])
                    executed_batches.append(["confirm", "wait 30"])
                else:
                    current_state = runtime.execute(["wait 30"])
                    executed_batches.append(["wait 30"])
                continue

            # 2. Potion handlers (boss turn 1, elite turn 1 at ≤50% HP)
            potion_used = False
            for handler in self._potion_handlers:
                if handler.can_handle(current_state):
                    action = handler.handle(current_state)
                    if action and action.commands:
                        log_to_run(f"DeterministicBattle potion: {action.commands}")
                        current_state = runtime.execute(action.commands)
                        executed_batches.append(action.commands)
                        potion_used = True
                        break
            if potion_used:
                continue

            # 3. CommonBattleHandler (calculator + 41-function comparator pipeline)
            if self._battle_handler.can_handle(current_state):
                action = self._battle_handler.handle(current_state)
                if action and action.commands:
                    log_to_run(f"DeterministicBattle calculator: {action.commands}")
                    current_state = runtime.execute(action.commands)
                    executed_batches.append(action.commands)
                    continue

            # 4. End turn fallback
            if current_state.has_command(Command.END):
                log_to_run("DeterministicBattle end turn")
                current_state = runtime.execute(["end"])
                executed_batches.append(["end"])
            else:
                log_to_run("DeterministicBattle no action available — waiting")
                current_state = runtime.execute(["wait 30"])
                executed_batches.append(["wait 30"])

        log_to_run(
            f"DeterministicBattle session ended: {session_id} | "
            f"steps={len(executed_batches)}"
        )
        return BattleSessionResult(
            handled=True,
            final_state=current_state,
            session_id=session_id,
            executed_commands=executed_batches,
            steps=len(executed_batches),
            summary=f"deterministic_battle steps={len(executed_batches)}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_signature(state: GameState) -> str:
        gs = state.game_state()
        monsters = gs.get("combat_state", {}).get("monsters", [])
        monster_hp = tuple(m.get("current_hp", 0) for m in monsters if not m.get("is_gone"))
        return str((
            state.screen_type(),
            tuple(sorted(gs.get("available_commands", []))),
            gs.get("current_action", ""),
            gs.get("turn", 0),
            gs.get("current_hp", 0),
            gs.get("player", {}).get("block", 0),
            gs.get("player", {}).get("energy", 0),
            monster_hp,
        ))
