from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rs.game.screen_type import ScreenType
from rs.helper.logger import log_to_run
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.battle_runtime import BattleRuntimeAdapter
from rs.llm.integration.campfire_context import build_campfire_agent_context
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.campfire_llm_provider import CampfireCommandProposal, CampfireLlmProvider
from rs.llm.subagent_validation_middleware import validate_proposed_command
from rs.machine.state import GameState


class CampfireCommandProvider(Protocol):
    def propose(
            self,
            context: AgentContext,
            working_memory: dict[str, Any],
            validation_feedback: dict[str, Any] | None = None,
    ) -> CampfireCommandProposal:
        ...


class CampfireSubagentState(TypedDict, total=False):
    runtime: BattleRuntimeAdapter
    current_state: GameState
    current_context: AgentContext
    working_memory: dict[str, Any]
    session_id: str
    proposal: CampfireCommandProposal
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
class CampfireSubagentConfig:
    max_decision_loops: int = 8
    max_validation_attempts: int = 2


@dataclass
class CampfireSessionResult:
    handled: bool
    final_state: GameState | None = None
    session_id: str = ""
    executed_commands: list[list[str]] = field(default_factory=list)
    steps: int = 0
    summary: str = ""


@dataclass
class CampfireWorkingMemory:
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


class CampfireSubagent:
    def __init__(
            self,
            *,
            provider: CampfireCommandProvider | None = None,
            langmem_service: LangMemService | None = None,
            config: CampfireSubagentConfig | None = None,
    ):
        self._provider = CampfireLlmProvider() if provider is None else provider
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._config = CampfireSubagentConfig() if config is None else config
        self._compiled_graph: Any | None = None

    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> CampfireSessionResult:
        output = dict(self.get_compiled_graph().invoke({
            "runtime": runtime,
            "current_state": state,
        }))
        return CampfireSessionResult(
            handled=bool(output.get("handled", False)),
            final_state=cast(GameState | None, output.get("final_state")),
            session_id=str(output.get("session_id", "")),
            executed_commands=[list(batch) for batch in output.get("executed_commands", [])],
            steps=int(output.get("steps", 0)),
            summary=str(output.get("final_summary", "")),
        )

    def get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            graph = StateGraph(CampfireSubagentState)
            graph.add_node("session_bootstrap", self._session_bootstrap_node)
            graph.add_node("retrieve_campfire_experience", self._retrieve_campfire_experience_node)
            graph.add_node("state_ingest", self._state_ingest_node)
            graph.add_node("model_decide_next_action", self._model_decide_next_action_node)
            graph.add_node("action_validate", self._action_validate_node)
            graph.add_node("action_commit", self._action_commit_node)
            graph.add_node("exit_check", self._exit_check_node)
            graph.add_node("session_finalize", self._session_finalize_node)

            graph.add_edge(START, "session_bootstrap")
            graph.add_edge("session_bootstrap", "retrieve_campfire_experience")
            graph.add_edge("retrieve_campfire_experience", "state_ingest")
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

    def _session_bootstrap_node(self, _state: CampfireSubagentState) -> dict[str, Any]:
        session_id = uuid4().hex
        working_memory = CampfireWorkingMemory(session_id=session_id)
        log_to_run(f"CampfireHandler subagent session started: {session_id}")
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

    def _retrieve_campfire_experience_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        current_state = state["current_state"]
        if not self.is_in_scope(current_state):
            return {}
        context = build_campfire_agent_context(current_state, "CampfireHandler")
        working_memory = dict(state.get("working_memory", {}))
        payload = self._langmem_service.build_context_memory(context)
        working_memory["retrieved_episodic_memories"] = payload.get("retrieved_episodic_memories", "none")
        working_memory["retrieved_semantic_memories"] = payload.get("retrieved_semantic_memories", "none")
        working_memory["langmem_status"] = payload.get("langmem_status", self._langmem_service.status())
        return {"working_memory": working_memory}

    def _state_ingest_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        in_scope = self.is_in_scope(current_state)
        if in_scope:
            context = build_campfire_agent_context(current_state, "CampfireHandler")
        else:
            context = AgentContext(
                handler_name="CampfireHandler",
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

    def _model_decide_next_action_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        working_memory = dict(state.get("working_memory", {}))
        working_memory["decision_loop_count"] = int(working_memory.get("decision_loop_count", 0)) + 1

        if not bool(state.get("in_scope")):
            return {
                "working_memory": working_memory,
                "proposal": CampfireCommandProposal(None, 1.0, "campfire_scope_ended", {"reason": "scope_ended"}),
            }

        if int(working_memory.get("decision_loop_count", 0)) > self._config.max_decision_loops:
            working_memory = self._append_step_summary(working_memory, "decision loop guardrail reached")
            return {
                "working_memory": working_memory,
                "proposal": CampfireCommandProposal(None, 0.0, "campfire_subagent_loop_guardrail", {"reason": "loop_guardrail"}),
            }

        context = state["current_context"]
        proposal = self._provider.propose(context, working_memory)
        return {
            "working_memory": working_memory,
            "proposal": proposal,
            "pending_decision_explanation": proposal.explanation,
            "pending_decision_confidence": proposal.confidence,
        }

    def _action_validate_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        proposal = state.get("proposal")
        if proposal is None or proposal.proposed_command is None:
            return {"pending_commands": []}

        context = state["current_context"]
        working_memory = dict(state.get("working_memory", {}))
        latest_explanation = str(state.get("pending_decision_explanation", proposal.explanation))
        latest_confidence = float(state.get("pending_decision_confidence", proposal.confidence))

        for attempt in range(self._config.max_validation_attempts):
            proposed_command = str(proposal.proposed_command).strip()
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
            else:
                validation, feedback = validate_proposed_command(context, proposed_command)
                if validation.is_valid:
                    return {
                        "pending_commands": [proposed_command],
                        "working_memory": working_memory,
                        "pending_decision_explanation": latest_explanation,
                        "pending_decision_confidence": latest_confidence,
                    }

            feedback_payload = dict(feedback or {})
            working_memory = self._append_step_summary(
                working_memory,
                f"validation failed for {proposed_command}: {feedback_payload.get('code', 'unknown')}",
            )
            if attempt + 1 >= self._config.max_validation_attempts:
                return {"pending_commands": [], "working_memory": working_memory}

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

        return {"pending_commands": [], "working_memory": working_memory}

    def _action_commit_node(self, state: CampfireSubagentState) -> dict[str, Any]:
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

    def _exit_check_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        return {
            "current_state": current_state,
            "in_scope": self.is_in_scope(current_state),
            "handled": bool(state.get("handled", False)),
        }

    def _route_after_exit_check(self, state: CampfireSubagentState) -> str:
        if bool(state.get("in_scope")) and bool(state.get("action_committed")):
            return "state_ingest"
        return "session_finalize"

    def _session_finalize_node(self, state: CampfireSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        final_state = runtime.current_state()
        working_memory = dict(state.get("working_memory", {}))
        final_summary = self._build_final_summary(final_state, working_memory)
        log_to_run(f"CampfireHandler subagent session ended: {state.get('session_id', '')}")
        if final_summary.strip():
            final_context = self._augment_context(state["current_context"], working_memory)
            self._langmem_service.record_custom_memory(
                final_context,
                final_summary,
                tags=("campfire_summary", "CampfireHandler"),
                reflect=True,
            )
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
        extras["campfire_working_memory"] = {
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

    def _record_accepted_decision(self, context: AgentContext, commands: list[str], explanation: str, confidence: float) -> None:
        decision = AgentDecision(
            proposed_command=" | ".join(commands),
            confidence=confidence,
            explanation=explanation or "campfire_subagent_step",
            required_tools_used=["campfire_subagent"],
            fallback_recommended=False,
            metadata={"campfire_session": context.extras.get("campfire_working_memory", {}).get("session_id", "")},
        )
        self._langmem_service.record_accepted_decision(context, decision)

    def _build_final_summary(self, final_state: GameState, working_memory: dict[str, Any]) -> str:
        game_state = final_state.game_state()
        floor = game_state.get("floor", "unknown")
        current_hp = game_state.get("current_hp", "unknown")
        max_hp = game_state.get("max_hp", "unknown")
        batches = working_memory.get("executed_command_batches", [])
        commands_str = ", ".join(
            "[" + " | ".join(batch) + "]" for batch in batches
        ) or "none"
        recent_steps = " | ".join(working_memory.get("recent_step_summaries", [])[-4:])
        return (
            f"Floor {floor} REST SITE. Player HP after: {current_hp}/{max_hp}.\n"
            f"Commands: {commands_str}\n"
            f"Step notes: {recent_steps or 'none'}\n"
            f"Review: Was REST vs SMITH (upgrade) the right tradeoff at this HP level? "
            f"Which card would benefit most from upgrading given the current deck? "
            f"What would improve the outcome of the next battle?"
        )

    def is_in_scope(self, state: GameState) -> bool:
        return state.screen_type() == ScreenType.REST.value

    def _uses_duplicate_choice_token(self, context: AgentContext, proposed_command: str) -> bool:
        tokens = proposed_command.split()
        if len(tokens) != 2 or tokens[0] != "choose":
            return False

        choice = tokens[1].strip().lower()
        if choice.isdigit():
            return False

        matches = [token for token in context.choice_list if str(token).strip().lower() == choice]
        return len(matches) > 1

    @staticmethod
    def _append_step_summary(working_memory: dict[str, Any], summary: str) -> dict[str, Any]:
        working_memory.setdefault("recent_step_summaries", []).append(summary)
        working_memory["recent_step_summaries"] = working_memory["recent_step_summaries"][-12:]
        return working_memory
