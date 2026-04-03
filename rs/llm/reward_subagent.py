from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rs.game.screen_type import ScreenType
from rs.helper.logger import log_to_run
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.battle_runtime import BattleRuntimeAdapter
from rs.llm.integration.astrolabe_transform_context import build_astrolabe_transform_agent_context
from rs.llm.integration.boss_reward_context import build_boss_reward_agent_context, is_astrolabe_transform_state
from rs.llm.integration.combat_reward_context import build_combat_reward_agent_context
from rs.llm.integration.grid_select_context import build_grid_select_agent_context
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.reward_llm_provider import (
    AstrolabeTransformLlmProvider,
    BossRewardLlmProvider,
    CombatRewardLlmProvider,
    GridSelectLlmProvider,
    RewardCommandProposal,
)
from rs.llm.subagent_validation_middleware import validate_proposed_command
from rs.machine.command import Command
from rs.machine.state import GameState


class RewardCommandProvider(Protocol):
    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> RewardCommandProposal:
        ...


class RewardSubagentState(TypedDict, total=False):
    runtime: BattleRuntimeAdapter
    current_state: GameState
    current_context: AgentContext
    working_memory: dict[str, Any]
    session_id: str
    proposal: RewardCommandProposal
    pending_commands: list[str]
    pending_decision_explanation: str
    pending_decision_confidence: float
    action_committed: bool
    handled: bool
    final_state: GameState
    executed_commands: list[list[str]]
    final_summary: str
    steps: int
    in_scope: bool


@dataclass
class RewardSubagentConfig:
    max_decision_loops: int = 16
    max_validation_attempts: int = 2


@dataclass
class RewardSessionResult:
    handled: bool
    final_state: GameState | None = None
    session_id: str = ""
    executed_commands: list[list[str]] = field(default_factory=list)
    steps: int = 0
    summary: str = ""


@dataclass
class RewardWorkingMemory:
    session_id: str
    recent_step_summaries: list[str] = field(default_factory=list)
    executed_command_batches: list[list[str]] = field(default_factory=list)
    retrieved_episodic_memories: str = "none"
    retrieved_semantic_memories: str = "none"
    langmem_status: str = "disabled_by_config"
    decision_loop_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recent_step_summaries": list(self.recent_step_summaries),
            "executed_command_batches": [list(batch) for batch in self.executed_command_batches],
            "retrieved_episodic_memories": self.retrieved_episodic_memories,
            "retrieved_semantic_memories": self.retrieved_semantic_memories,
            "langmem_status": self.langmem_status,
            "decision_loop_count": self.decision_loop_count,
        }


class RewardSubagentBase:
    def __init__(
            self,
            *,
            handler_name: str,
            provider: RewardCommandProvider,
            langmem_service: LangMemService | None = None,
            config: RewardSubagentConfig | None = None,
    ):
        self._handler_name = handler_name
        self._provider = provider
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._config = RewardSubagentConfig() if config is None else config
        self._compiled_graph: Any | None = None

    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> RewardSessionResult:
        output = dict(self.get_compiled_graph().invoke({
            "runtime": runtime,
            "current_state": state,
        }))
        return RewardSessionResult(
            handled=bool(output.get("handled", False)),
            final_state=cast(GameState | None, output.get("final_state")),
            session_id=str(output.get("session_id", "")),
            executed_commands=[list(batch) for batch in output.get("executed_commands", [])],
            steps=int(output.get("steps", 0)),
            summary=str(output.get("final_summary", "")),
        )

    def get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            graph = StateGraph(RewardSubagentState)
            graph.add_node("session_bootstrap", self._session_bootstrap_node)
            graph.add_node("retrieve_reward_experience", self._retrieve_reward_experience_node)
            graph.add_node("state_ingest", self._state_ingest_node)
            graph.add_node("model_decide_next_action", self._model_decide_next_action_node)
            graph.add_node("action_validate", self._action_validate_node)
            graph.add_node("action_commit", self._action_commit_node)
            graph.add_node("exit_check", self._exit_check_node)
            graph.add_node("session_finalize", self._session_finalize_node)

            graph.add_edge(START, "session_bootstrap")
            graph.add_edge("session_bootstrap", "retrieve_reward_experience")
            graph.add_edge("retrieve_reward_experience", "state_ingest")
            graph.add_edge("state_ingest", "model_decide_next_action")
            graph.add_edge("model_decide_next_action", "action_validate")
            graph.add_edge("action_validate", "action_commit")
            graph.add_edge("action_commit", "exit_check")
            graph.add_conditional_edges(
                "exit_check",
                self._route_after_exit_check,
                {
                    "state_ingest": "state_ingest",
                    "session_finalize": "session_finalize",
                },
            )
            graph.add_edge("session_finalize", END)
            self._compiled_graph = graph.compile()
        return self._compiled_graph

    def _session_bootstrap_node(self, _state: RewardSubagentState) -> dict[str, Any]:
        session_id = uuid4().hex
        working_memory = RewardWorkingMemory(session_id=session_id)
        log_to_run(f"{self._handler_name} subagent session started: {session_id}")
        return {
            "session_id": session_id,
            "working_memory": working_memory.to_dict(),
            "pending_commands": [],
            "executed_commands": [],
            "steps": 0,
            "handled": False,
            "action_committed": False,
            "final_summary": "",
        }

    def _retrieve_reward_experience_node(self, state: RewardSubagentState) -> dict[str, Any]:
        current_state = state["current_state"]
        context = self._build_context(current_state)
        if context is None:
            return {}

        working_memory = dict(state.get("working_memory", {}))
        payload = self._langmem_service.build_context_memory(context)
        working_memory["retrieved_episodic_memories"] = payload.get("retrieved_episodic_memories", "none")
        working_memory["retrieved_semantic_memories"] = payload.get("retrieved_semantic_memories", "none")
        working_memory["langmem_status"] = payload.get("langmem_status", self._langmem_service.status())
        return {"working_memory": working_memory}

    def _state_ingest_node(self, state: RewardSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        context = self._build_context(current_state)
        in_scope = context is not None and self.is_in_scope(current_state)
        if context is None:
            context = AgentContext(
                handler_name=self._handler_name,
                screen_type=current_state.screen_type(),
                available_commands=[],
                choice_list=[],
                game_state={},
                extras={},
            )
        context = self._augment_context(context, dict(state.get("working_memory", {})))
        return {
            "current_state": current_state,
            "current_context": context,
            "in_scope": in_scope,
            "action_committed": False,
        }

    def _model_decide_next_action_node(self, state: RewardSubagentState) -> dict[str, Any]:
        working_memory = dict(state.get("working_memory", {}))
        working_memory["decision_loop_count"] = int(working_memory.get("decision_loop_count", 0)) + 1

        if not bool(state.get("in_scope")):
            return {
                "working_memory": working_memory,
                "proposal": RewardCommandProposal(
                    proposed_command=None,
                    confidence=1.0,
                    explanation="reward_scope_ended",
                    metadata={"reason": "scope_ended"},
                ),
            }

        if int(working_memory.get("decision_loop_count", 0)) > self._config.max_decision_loops:
            working_memory = self._append_step_summary(working_memory, "decision loop guardrail reached")
            return {
                "working_memory": working_memory,
                "proposal": RewardCommandProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="reward_subagent_loop_guardrail",
                    metadata={"reason": "loop_guardrail"},
                ),
            }

        context = state["current_context"]
        deterministic_proposal = self._deterministic_combat_reward_card_entry(context)
        if deterministic_proposal is not None:
            working_memory = self._append_step_summary(
                working_memory,
                f"deterministic card handoff: {deterministic_proposal.proposed_command}",
            )
            return {
                "working_memory": working_memory,
                "proposal": deterministic_proposal,
                "pending_decision_explanation": deterministic_proposal.explanation,
                "pending_decision_confidence": deterministic_proposal.confidence,
            }

        proposal = self._provider.propose(context, working_memory)
        return {
            "working_memory": working_memory,
            "proposal": proposal,
            "pending_decision_explanation": proposal.explanation,
            "pending_decision_confidence": proposal.confidence,
        }

    def _action_validate_node(self, state: RewardSubagentState) -> dict[str, Any]:
        proposal = state.get("proposal")
        if proposal is None or proposal.proposed_command is None:
            return {"pending_commands": []}

        context = state["current_context"]
        working_memory = dict(state.get("working_memory", {}))
        latest_explanation = str(state.get("pending_decision_explanation", proposal.explanation))
        latest_confidence = float(state.get("pending_decision_confidence", proposal.confidence))

        for attempt in range(self._config.max_validation_attempts):
            proposed_command = str(proposal.proposed_command).strip()
            is_deterministic_card_handoff = (
                str(proposal.metadata.get("reason", "")) == "deterministic_card_handoff"
            )

            feedback: dict[str, Any] | None = None
            if self._uses_duplicate_choice_token(context, proposed_command):
                feedback = {
                    "rejected_command": proposed_command,
                    "code": "choose_requires_index",
                    "message": "ambiguous choose token detected",
                    "corrective_hint": "choose must use index form `choose <index>`.",
                    "valid_index_range": f"0..{max(0, len(context.choice_list) - 1)}",
                    "valid_example": "choose 0" if context.choice_list else "none",
                }
            elif not is_deterministic_card_handoff and self._is_card_row_choice(context, proposed_command):
                feedback = {
                    "rejected_command": proposed_command,
                    "code": "card_row_handoff_required",
                    "message": "card row should be delegated through deterministic handoff",
                    "corrective_hint": "choose must target non-card reward row, or return null if no non-card action.",
                    "valid_index_range": f"0..{max(0, len(context.choice_list) - 1)}",
                    "valid_example": "choose 0" if context.choice_list else "none",
                }
            else:
                feedback = self._build_reward_specific_feedback(context, proposed_command)
            if feedback is None:
                validation, feedback = validate_proposed_command(context, proposed_command)
                if validation.is_valid:
                    commands = self._commands_for_context(context, proposed_command)
                    if commands is not None:
                        return {
                            "pending_commands": commands,
                            "working_memory": working_memory,
                            "pending_decision_explanation": latest_explanation,
                            "pending_decision_confidence": latest_confidence,
                        }
                    feedback = {
                        "rejected_command": proposed_command,
                        "code": "no_execution_mapping",
                        "message": f"no execution mapping for {proposed_command}",
                        "corrective_hint": "return one command compatible with current handler mapping.",
                        "valid_index_range": f"0..{max(0, len(context.choice_list) - 1)}",
                        "valid_example": "choose 0" if context.choice_list else "none",
                    }

            feedback_payload = dict(feedback or {})
            working_memory = self._append_step_summary(
                working_memory,
                f"validation failed for {proposed_command}: {feedback_payload.get('code', 'unknown')}",
            )
            if attempt + 1 >= self._config.max_validation_attempts:
                fallback = self._validation_exhausted_fallback(context, working_memory)
                return {"pending_commands": fallback, "working_memory": working_memory}

            proposal = self._provider.propose(
                context,
                working_memory,
                validation_feedback=feedback_payload,
            )
            latest_explanation = str(proposal.explanation)
            try:
                latest_confidence = float(proposal.confidence)
            except (TypeError, ValueError):
                latest_confidence = 0.0
            if proposal.proposed_command is None:
                break

        fallback = self._validation_exhausted_fallback(context, working_memory)
        return {"pending_commands": fallback, "working_memory": working_memory}

    def _build_reward_specific_feedback(self, context: AgentContext, proposed_command: str) -> dict[str, Any] | None:
        return None

    def _action_commit_node(self, state: RewardSubagentState) -> dict[str, Any]:
        commands = [str(command).strip() for command in state.get("pending_commands", []) if str(command).strip()]
        if not commands:
            return {"action_committed": False}

        runtime = state["runtime"]
        next_state = runtime.execute(commands)
        working_memory = dict(state.get("working_memory", {}))
        working_memory.setdefault("executed_command_batches", []).append(commands)
        working_memory = self._append_step_summary(working_memory, f"executed command batch: {', '.join(commands)}")
        self._record_accepted_decision(
            state["current_context"],
            commands,
            str(state.get("pending_decision_explanation", "")),
            float(state.get("pending_decision_confidence", 0.0)),
        )
        return {
            "current_state": next_state,
            "working_memory": working_memory,
            "executed_commands": list(working_memory.get("executed_command_batches", [])),
            "steps": len(working_memory.get("executed_command_batches", [])),
            "handled": True,
            "action_committed": True,
        }

    def _exit_check_node(self, state: RewardSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        return {
            "current_state": current_state,
            "in_scope": self.is_in_scope(current_state),
            "handled": bool(state.get("handled", False)),
        }

    def _route_after_exit_check(self, state: RewardSubagentState) -> str:
        if bool(state.get("in_scope")) and bool(state.get("action_committed")):
            return "state_ingest"
        return "session_finalize"

    def _session_finalize_node(self, state: RewardSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        final_state = runtime.current_state()
        final_summary = self._build_final_summary(final_state, dict(state.get("working_memory", {})))
        log_to_run(f"{self._handler_name} subagent session ended: {state.get('session_id', '')}")
        handled = bool(state.get("handled", False))
        if self.is_in_scope(final_state) and not state.get("executed_commands"):
            handled = False
        return {
            "handled": handled,
            "final_state": final_state,
            "executed_commands": [list(batch) for batch in state.get("executed_commands", [])],
            "final_summary": final_summary,
            "steps": int(state.get("steps", 0)),
        }

    def _augment_context(self, context: AgentContext, working_memory: dict[str, Any]) -> AgentContext:
        extras = dict(context.extras)
        extras["retrieved_episodic_memories"] = working_memory.get("retrieved_episodic_memories", "none")
        extras["retrieved_semantic_memories"] = working_memory.get("retrieved_semantic_memories", "none")
        extras["langmem_status"] = working_memory.get("langmem_status", self._langmem_service.status())
        extras["reward_working_memory"] = {
            "session_id": working_memory.get("session_id", ""),
            "recent_step_summaries": list(working_memory.get("recent_step_summaries", []))[-6:],
            "executed_command_batches": [list(batch) for batch in working_memory.get("executed_command_batches", [])][-4:],
        }
        return AgentContext(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=list(context.available_commands),
            choice_list=list(context.choice_list),
            game_state=dict(context.game_state),
            extras=extras,
        )

    def _record_accepted_decision(
            self,
            context: AgentContext,
            commands: list[str],
            explanation: str,
            confidence: float,
    ) -> None:
        decision = AgentDecision(
            proposed_command=" | ".join(commands),
            confidence=confidence,
            explanation=explanation or f"{self._handler_name.lower()}_subagent_step",
            required_tools_used=["reward_subagent"],
            fallback_recommended=False,
            metadata={"reward_session": context.extras.get("reward_working_memory", {}).get("session_id", "")},
        )
        self._langmem_service.record_accepted_decision(context, decision)

    def _build_final_summary(self, final_state: GameState, working_memory: dict[str, Any]) -> str:
        game_state = final_state.game_state()
        floor = game_state.get("floor", "unknown")
        room_type = game_state.get("room_type", "unknown")
        room_phase = game_state.get("room_phase", "unknown")
        recent_steps = " | ".join(working_memory.get("recent_step_summaries", [])[-4:])
        return (
            f"{self._handler_name} session ended on floor {floor} with room_type={room_type} "
            f"room_phase={room_phase}. Executed {len(working_memory.get('executed_command_batches', []))} "
            f"command batches. Recent reward notes: {recent_steps or 'none'}"
        )

    def _commands_for_context(self, context: AgentContext, proposed_command: str) -> list[str] | None:
        if context.handler_name == "CombatRewardHandler":
            if proposed_command.startswith("potion use ") or proposed_command.startswith("potion discard "):
                return ["wait 30", proposed_command]
            return [proposed_command]

        if context.handler_name == "BossRewardHandler":
            if proposed_command == "skip":
                return ["skip", "proceed"]
            return [proposed_command]

        if context.handler_name == "AstrolabeTransformHandler":
            return [proposed_command]

        if context.handler_name == "GridSelectHandler":
            if proposed_command in ("confirm", "proceed"):
                return [proposed_command]
            return [proposed_command, "wait 30"]

        return None

    def _uses_duplicate_choice_token(self, context: AgentContext, proposed_command: str) -> bool:
        tokens = proposed_command.split()
        if len(tokens) != 2 or tokens[0] != "choose":
            return False

        choice = tokens[1].strip().lower()
        if choice.isdigit():
            return False

        matches = [token for token in context.choice_list if str(token).strip().lower() == choice]
        return len(matches) > 1

    def _is_card_row_choice(self, context: AgentContext, proposed_command: str) -> bool:
        if context.handler_name != "CombatRewardHandler":
            return False
        tokens = proposed_command.split()
        if len(tokens) != 2 or tokens[0] != "choose":
            return False
        choice = tokens[1].strip().lower()
        card_choice_indexes = {
            int(index)
            for index in context.extras.get("card_reward_choice_indexes", [])
            if isinstance(index, int)
        }
        card_choice_tokens = {
            str(token).strip().lower()
            for token in context.extras.get("card_reward_choice_tokens", [])
            if str(token).strip() != ""
        }
        if choice.isdigit():
            return int(choice) in card_choice_indexes
        return choice in card_choice_tokens

    def _deterministic_combat_reward_card_entry(self, context: AgentContext) -> RewardCommandProposal | None:
        if context.handler_name != "CombatRewardHandler":
            return None
        non_card_reward_count = int(context.extras.get("non_card_reward_count", 0))
        has_card_reward_row = bool(context.extras.get("has_card_reward_row", False))
        card_choice_index = context.extras.get("card_reward_choice_index")
        if non_card_reward_count != 0 or not has_card_reward_row or not isinstance(card_choice_index, int):
            return None
        return RewardCommandProposal(
            proposed_command=f"choose {card_choice_index}",
            confidence=1.0,
            explanation="deterministic_card_handoff_when_only_card_row_remains",
            metadata={"reason": "deterministic_card_handoff"},
        )

    def _validation_exhausted_fallback(self, context: AgentContext, working_memory: dict[str, Any]) -> list[str]:
        return []

    @staticmethod
    def _append_step_summary(working_memory: dict[str, Any], summary: str) -> dict[str, Any]:
        working_memory.setdefault("recent_step_summaries", []).append(summary)
        working_memory["recent_step_summaries"] = working_memory["recent_step_summaries"][-12:]
        return working_memory

    def is_in_scope(self, state: GameState) -> bool:
        raise Exception("must be implemented by children")

    def _build_context(self, state: GameState) -> AgentContext | None:
        raise Exception("must be implemented by children")


class CombatRewardSubagent(RewardSubagentBase):
    def __init__(
            self,
            *,
            provider: RewardCommandProvider | None = None,
            langmem_service: LangMemService | None = None,
            config: RewardSubagentConfig | None = None,
    ):
        super().__init__(
            handler_name="CombatRewardHandler",
            provider=CombatRewardLlmProvider() if provider is None else provider,
            langmem_service=langmem_service,
            config=config,
        )

    def is_in_scope(self, state: GameState) -> bool:
        return state.screen_type() == ScreenType.COMBAT_REWARD.value

    def _build_context(self, state: GameState) -> AgentContext | None:
        if not self.is_in_scope(state):
            return None
        return build_combat_reward_agent_context(state, "CombatRewardHandler")

    def _build_reward_specific_feedback(self, context: AgentContext, proposed_command: str) -> dict[str, Any] | None:
        selected_reward = self._selected_reward_summary(context, proposed_command)
        if selected_reward is None:
            return None
        if str(selected_reward.get("reward_type", "")).strip().upper() != "POTION":
            return None
        if not bool(context.extras.get("potions_full", False)):
            return None

        allowed_commands = self._full_potion_resolution_commands(context)
        valid_example = allowed_commands[0] if allowed_commands else Command.PROCEED.value
        potion_name = str(selected_reward.get("potion_name", "")).strip() or "reward potion"
        reward_index = selected_reward.get("choice_index", "?")
        return {
            "rejected_command": proposed_command,
            "code": "potion_capacity_resolution_required",
            "message": "cannot take potion reward while potion slots are full",
            "corrective_hint": (
                f"reward_summaries[{reward_index}] is a potion reward ({potion_name}) but potions_full=true. "
                "Return exactly one immediate capacity-resolution action: a legal `potion use <slot_index>`, "
                "a legal `potion discard <slot_index>`, or `proceed` to skip this new potion."
            ),
            "valid_index_range": f"0..{max(0, len(context.choice_list) - 1)}",
            "valid_example": valid_example,
            "allowed_commands": allowed_commands,
            "selected_reward_index": reward_index,
            "selected_reward_type": "POTION",
        }

    def _validation_exhausted_fallback(self, context: AgentContext, working_memory: dict[str, Any]) -> list[str]:
        if self._is_full_potion_resolution_context(context):
            available = {str(command).strip().lower() for command in context.available_commands}
            if Command.PROCEED.value in available:
                return [Command.PROCEED.value]
        return []

    def _is_full_potion_resolution_context(self, context: AgentContext) -> bool:
        if context.handler_name != "CombatRewardHandler":
            return False
        if not bool(context.extras.get("potions_full", False)):
            return False
        return any(
            str(summary.get("reward_type", "")).strip().upper() == "POTION"
            for summary in context.extras.get("reward_summaries", [])
            if isinstance(summary, dict)
        )

    def _selected_reward_summary(self, context: AgentContext, proposed_command: str) -> dict[str, Any] | None:
        tokens = str(proposed_command).split()
        if len(tokens) != 2 or tokens[0] != Command.CHOOSE.value:
            return None

        choice = tokens[1].strip().lower()
        reward_summaries = [
            summary for summary in context.extras.get("reward_summaries", [])
            if isinstance(summary, dict)
        ]
        if choice.isdigit():
            choice_index = int(choice)
            for summary in reward_summaries:
                if int(summary.get("choice_index", -1)) == choice_index:
                    return summary
            return None

        matches = [
            summary for summary in reward_summaries
            if str(summary.get("choice_token", "")).strip().lower() == choice
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _full_potion_resolution_commands(self, context: AgentContext) -> list[str]:
        game_state_ref = context.extras.get("game_state_ref")
        if not isinstance(game_state_ref, GameState):
            return [Command.PROCEED.value]

        commands: list[str] = []
        for slot_index, potion in enumerate(game_state_ref.get_potions()):
            if not isinstance(potion, dict):
                continue
            if bool(potion.get("can_use", False)):
                commands.append(f"{Command.POTION.value} use {slot_index}")
            if bool(potion.get("can_discard", False)):
                commands.append(f"{Command.POTION.value} discard {slot_index}")

        available = {str(command).strip().lower() for command in context.available_commands}
        if Command.PROCEED.value in available:
            commands.append(Command.PROCEED.value)

        seen: set[str] = set()
        deduped: list[str] = []
        for command in commands:
            if command in seen:
                continue
            seen.add(command)
            deduped.append(command)
        return deduped


class BossRewardSubagent(RewardSubagentBase):
    def __init__(
            self,
            *,
            provider: RewardCommandProvider | None = None,
            langmem_service: LangMemService | None = None,
            config: RewardSubagentConfig | None = None,
    ):
        super().__init__(
            handler_name="BossRewardHandler",
            provider=BossRewardLlmProvider() if provider is None else provider,
            langmem_service=langmem_service,
            config=config,
        )

    def is_in_scope(self, state: GameState) -> bool:
        return state.screen_type() == ScreenType.BOSS_REWARD.value

    def _build_context(self, state: GameState) -> AgentContext | None:
        if not self.is_in_scope(state):
            return None
        return build_boss_reward_agent_context(state, "BossRewardHandler")


class AstrolabeTransformSubagent(RewardSubagentBase):
    def __init__(
            self,
            *,
            provider: RewardCommandProvider | None = None,
            langmem_service: LangMemService | None = None,
            config: RewardSubagentConfig | None = None,
    ):
        super().__init__(
            handler_name="AstrolabeTransformHandler",
            provider=AstrolabeTransformLlmProvider() if provider is None else provider,
            langmem_service=langmem_service,
            config=config,
        )

    def is_in_scope(self, state: GameState) -> bool:
        return is_astrolabe_transform_state(state)

    def _build_context(self, state: GameState) -> AgentContext | None:
        if not self.is_in_scope(state):
            return None
        return build_astrolabe_transform_agent_context(state, "AstrolabeTransformHandler")


class GridSelectSubagent(RewardSubagentBase):
    def __init__(
            self,
            *,
            provider: RewardCommandProvider | None = None,
            langmem_service: LangMemService | None = None,
            config: RewardSubagentConfig | None = None,
    ):
        super().__init__(
            handler_name="GridSelectHandler",
            provider=GridSelectLlmProvider() if provider is None else provider,
            langmem_service=langmem_service,
            config=config,
        )

    def is_in_scope(self, state: GameState) -> bool:
        return (
            state.screen_type() == ScreenType.GRID.value
            and not is_astrolabe_transform_state(state)
        )

    def _build_context(self, state: GameState) -> AgentContext | None:
        if not self.is_in_scope(state):
            return None
        return build_grid_select_agent_context(state, "GridSelectHandler")

    def _deterministic_confirm(self, context: AgentContext) -> RewardCommandProposal | None:
        picks_remaining = int(context.extras.get("picks_remaining", 1))
        if picks_remaining > 0:
            return None
        available = context.available_commands
        if "confirm" in available:
            command = "confirm"
        elif "proceed" in available:
            command = "proceed"
        else:
            return None
        return RewardCommandProposal(
            proposed_command=command,
            confidence=1.0,
            explanation="all_cards_selected_confirm",
            metadata={"reason": "deterministic_confirm"},
        )

    def _validation_exhausted_fallback(self, context: AgentContext, working_memory: dict[str, Any]) -> list[str]:
        available = {str(command).strip().lower() for command in context.available_commands}
        picks_remaining = int(context.extras.get("picks_remaining", 1))
        if picks_remaining <= 0:
            if "confirm" in available:
                return ["confirm"]
            if "proceed" in available:
                return ["proceed"]

        selectable_cards = context.extras.get("selectable_cards", [])
        if "choose" in available and selectable_cards and len(context.choice_list) > 0:
            return ["choose 0", "wait 30"]
        return []

    def _model_decide_next_action_node(self, state: RewardSubagentState) -> dict[str, Any]:
        working_memory = dict(state.get("working_memory", {}))
        working_memory["decision_loop_count"] = int(working_memory.get("decision_loop_count", 0)) + 1

        if not bool(state.get("in_scope")):
            return {
                "working_memory": working_memory,
                "proposal": RewardCommandProposal(
                    proposed_command=None,
                    confidence=1.0,
                    explanation="grid_scope_ended",
                    metadata={"reason": "scope_ended"},
                ),
            }

        if int(working_memory.get("decision_loop_count", 0)) > self._config.max_decision_loops:
            working_memory = self._append_step_summary(working_memory, "decision loop guardrail reached")
            return {
                "working_memory": working_memory,
                "proposal": RewardCommandProposal(
                    proposed_command=None,
                    confidence=0.0,
                    explanation="grid_subagent_loop_guardrail",
                    metadata={"reason": "loop_guardrail"},
                ),
            }

        context = state["current_context"]
        confirm_proposal = self._deterministic_confirm(context)
        if confirm_proposal is not None:
            working_memory = self._append_step_summary(working_memory, "deterministic confirm: all cards selected")
            return {
                "working_memory": working_memory,
                "proposal": confirm_proposal,
                "pending_decision_explanation": confirm_proposal.explanation,
                "pending_decision_confidence": confirm_proposal.confidence,
            }

        proposal = self._provider.propose(context, working_memory)
        return {
            "working_memory": working_memory,
            "proposal": proposal,
            "pending_decision_explanation": proposal.explanation,
            "pending_decision_confidence": proposal.confidence,
        }
