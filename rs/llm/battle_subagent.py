from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rs.helper.logger import log_to_run
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.battle_runtime import BattleRuntimeAdapter, BattleSessionResult
from rs.llm.integration.battle_context import build_battle_agent_context, is_battle_scope_state
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.battle_llm_provider import BattleDirective, BattleDirectiveProvider, BattleLlmProvider
from rs.machine.state import GameState


class BattleToolProtocol:
    name: str

    def run(self, context: AgentContext, payload: dict[str, Any]) -> dict[str, Any]:
        raise Exception("must be implemented by battle tool")


class BattleSubagentState(TypedDict, total=False):
    runtime: BattleRuntimeAdapter
    current_state: GameState
    current_context: AgentContext
    working_memory: dict[str, Any]
    session_id: str
    directive: BattleDirective
    pending_commands: list[str]
    pending_required_tools_used: list[str]
    pending_decision_explanation: str
    pending_decision_confidence: float
    action_committed: bool
    battle_complete: bool
    handled: bool
    final_state: GameState
    executed_commands: list[list[str]]
    final_summary: str
    steps: int


@dataclass
class BattleSubagentConfig:
    max_decision_loops: int = 32
    max_tool_calls: int = 24
    fallback_max_path_count: int = 200


@dataclass
class BattleWorkingMemory:
    session_id: str
    recent_step_summaries: list[str] = field(default_factory=list)
    recent_tool_results: list[dict[str, Any]] = field(default_factory=list)
    executed_command_batches: list[list[str]] = field(default_factory=list)
    retrieved_episodic_memories: str = "none"
    retrieved_semantic_memories: str = "none"
    langmem_status: str = "disabled_by_config"
    tool_call_count: int = 0
    decision_loop_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recent_step_summaries": list(self.recent_step_summaries),
            "recent_tool_results": list(self.recent_tool_results),
            "executed_command_batches": [list(batch) for batch in self.executed_command_batches],
            "retrieved_episodic_memories": self.retrieved_episodic_memories,
            "retrieved_semantic_memories": self.retrieved_semantic_memories,
            "langmem_status": self.langmem_status,
            "tool_call_count": self.tool_call_count,
            "decision_loop_count": self.decision_loop_count,
        }


class BattleSubagent:
    def __init__(
            self,
            *,
            provider: BattleDirectiveProvider | None = None,
            langmem_service: LangMemService | None = None,
            tools: list[BattleToolProtocol] | None = None,
            config: BattleSubagentConfig | None = None,
    ):
        self._provider = BattleLlmProvider() if provider is None else provider
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._config = BattleSubagentConfig() if config is None else config
        self._tools = {tool.name: tool for tool in (tools or [])}
        self._compiled_graph: Any | None = None

    def register_tool(self, tool: BattleToolProtocol) -> None:
        self._tools[tool.name] = tool
        self._compiled_graph = None

    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> BattleSessionResult:
        output = dict(self.get_compiled_graph().invoke({
            "runtime": runtime,
            "current_state": state,
        }))
        return BattleSessionResult(
            handled=bool(output.get("handled", False)),
            final_state=cast(GameState | None, output.get("final_state")),
            session_id=str(output.get("session_id", "")),
            executed_commands=[list(batch) for batch in output.get("executed_commands", [])],
            steps=int(output.get("steps", 0)),
            summary=str(output.get("final_summary", "")),
        )

    def get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            graph = StateGraph(BattleSubagentState)
            graph.add_node("session_bootstrap", self._session_bootstrap_node)
            graph.add_node("retrieve_battle_experience", self._retrieve_battle_experience_node)
            graph.add_node("state_ingest", self._state_ingest_node)
            graph.add_node("model_decide_next_step", self._model_decide_next_step_node)
            graph.add_node("tool_execute", self._tool_execute_node)
            graph.add_node("action_validate", self._action_validate_node)
            graph.add_node("action_commit", self._action_commit_node)
            graph.add_node("exit_check", self._exit_check_node)
            graph.add_node("session_finalize", self._session_finalize_node)

            graph.add_edge(START, "session_bootstrap")
            graph.add_edge("session_bootstrap", "retrieve_battle_experience")
            graph.add_edge("retrieve_battle_experience", "state_ingest")
            graph.add_edge("state_ingest", "model_decide_next_step")
            graph.add_conditional_edges(
                "model_decide_next_step",
                self._route_after_model_decision,
                {
                    "tool_execute": "tool_execute",
                    "action_validate": "action_validate",
                    "session_finalize": "session_finalize",
                },
            )
            graph.add_conditional_edges(
                "tool_execute",
                self._route_after_tool_execution,
                {
                    "model_decide_next_step": "model_decide_next_step",
                    "exit_check": "exit_check",
                    "action_validate": "action_validate",
                    "session_finalize": "session_finalize",
                },
            )
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

    def _session_bootstrap_node(self, state: BattleSubagentState) -> dict[str, Any]:
        session_id = uuid4().hex
        working_memory = BattleWorkingMemory(session_id=session_id)
        log_to_run(f"BattleSubagent session started: {session_id}")
        return {
            "session_id": session_id,
            "working_memory": working_memory.to_dict(),
            "pending_required_tools_used": [],
            "executed_commands": [],
            "steps": 0,
            "handled": False,
            "battle_complete": False,
            "action_committed": False,
            "final_summary": "",
        }

    def _retrieve_battle_experience_node(self, state: BattleSubagentState) -> dict[str, Any]:
        current_state = state["current_state"]
        runtime = state["runtime"]
        working_memory = dict(state["working_memory"])
        context = build_battle_agent_context(
            current_state,
            "BattleHandler",
            working_memory=working_memory,
            runtime=runtime,
        )
        payload = self._langmem_service.build_context_memory(context)
        working_memory["retrieved_episodic_memories"] = payload.get("retrieved_episodic_memories", "none")
        working_memory["retrieved_semantic_memories"] = payload.get("retrieved_semantic_memories", "none")
        working_memory["langmem_status"] = payload.get("langmem_status", self._langmem_service.status())
        return {"working_memory": working_memory}

    def _state_ingest_node(self, state: BattleSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        working_memory = dict(state["working_memory"])
        context = build_battle_agent_context(
            current_state,
            "BattleHandler",
            working_memory=working_memory,
            runtime=runtime,
            retrieved_episodic_memories=str(working_memory.get("retrieved_episodic_memories", "none")),
            retrieved_semantic_memories=str(working_memory.get("retrieved_semantic_memories", "none")),
            langmem_status=str(working_memory.get("langmem_status", self._langmem_service.status())),
        )
        battle_complete = not is_battle_scope_state(current_state)
        return {
            "current_state": current_state,
            "current_context": context,
            "battle_complete": battle_complete,
            "action_committed": False,
        }

    def _model_decide_next_step_node(self, state: BattleSubagentState) -> dict[str, Any]:
        working_memory = dict(state["working_memory"])
        working_memory["decision_loop_count"] = int(working_memory.get("decision_loop_count", 0)) + 1

        if bool(state.get("battle_complete")):
            directive = BattleDirective(mode="stop", explanation="battle_scope_ended", confidence=1.0)
            return {"directive": directive, "working_memory": working_memory}

        if int(working_memory.get("decision_loop_count", 0)) > self._config.max_decision_loops:
            fallback_commands = self._build_guardrail_fallback_commands(state["current_context"])
            directive = BattleDirective(
                mode="action" if fallback_commands else "stop",
                explanation="battle_subagent_loop_guardrail",
                confidence=0.25,
                commands=fallback_commands,
            )
            return {"directive": directive, "working_memory": working_memory}

        context = state["current_context"]
        directive = self._provider.propose(context, working_memory, self._tool_descriptions())
        return {
            "directive": directive,
            "working_memory": working_memory,
            "pending_decision_explanation": directive.explanation,
            "pending_decision_confidence": directive.confidence,
        }

    def _route_after_model_decision(self, state: BattleSubagentState) -> Literal["tool_execute", "action_validate", "session_finalize"]:
        directive = state.get("directive")
        if directive is None:
            return "session_finalize"
        if directive.mode == "tool":
            return "tool_execute"
        if directive.mode == "action":
            return "action_validate"
        return "session_finalize"

    def _tool_execute_node(self, state: BattleSubagentState) -> dict[str, Any]:
        context = state["current_context"]
        directive = state["directive"]
        working_memory = dict(state["working_memory"])
        tool_names = list(state.get("pending_required_tools_used", []))

        if directive.tool_name is None or directive.tool_name not in self._tools:
            working_memory = self._append_step_summary(working_memory, "invalid tool request")
            return {
                "working_memory": working_memory,
                "directive": BattleDirective(
                    mode="action",
                    explanation="invalid_tool_request",
                    confidence=0.2,
                    commands=self._build_guardrail_fallback_commands(context),
                ),
            }

        working_memory["tool_call_count"] = int(working_memory.get("tool_call_count", 0)) + 1
        if int(working_memory.get("tool_call_count", 0)) > self._config.max_tool_calls:
            working_memory = self._append_step_summary(working_memory, "tool call guardrail reached")
            return {
                "working_memory": working_memory,
                "directive": BattleDirective(
                    mode="action",
                    explanation="tool_call_guardrail",
                    confidence=0.2,
                    commands=self._build_guardrail_fallback_commands(context),
                ),
            }

        tool = self._tools[directive.tool_name]
        result = tool.run(context, dict(directive.tool_payload or {}))
        tool_names.append(tool.name)
        working_memory = self._append_tool_result(working_memory, tool.name, result)

        if tool.name == "retrieve_battle_experience":
            working_memory["retrieved_episodic_memories"] = result.get(
                "retrieved_episodic_memories",
                working_memory.get("retrieved_episodic_memories", "none"),
            )
            working_memory["retrieved_semantic_memories"] = result.get(
                "retrieved_semantic_memories",
                working_memory.get("retrieved_semantic_memories", "none"),
            )
            working_memory["langmem_status"] = result.get(
                "langmem_status",
                working_memory.get("langmem_status", self._langmem_service.status()),
            )

        if tool.name == "execute_battle_command" and bool(result.get("executed")):
            commands = [str(command) for command in result.get("commands", [])]
            working_memory.setdefault("executed_command_batches", []).append(commands)
            working_memory = self._append_step_summary(working_memory, f"executed via tool: {', '.join(commands)}")
            self._record_accepted_decision(
                context,
                commands,
                str(state.get("pending_decision_explanation", directive.explanation)),
                float(state.get("pending_decision_confidence", directive.confidence)),
                tool_names,
            )
            return {
                "working_memory": working_memory,
                "pending_required_tools_used": [],
                "current_state": cast(GameState, result.get("state", state["current_state"])),
                "action_committed": True,
                "executed_commands": list(working_memory.get("executed_command_batches", [])),
                "steps": len(working_memory.get("executed_command_batches", [])),
            }

        return {
            "working_memory": working_memory,
            "pending_required_tools_used": tool_names,
            "action_committed": False,
        }

    def _route_after_tool_execution(self, state: BattleSubagentState) -> Literal["model_decide_next_step", "exit_check", "session_finalize", "action_validate"]:
        if bool(state.get("action_committed")):
            return "exit_check"
        directive = state.get("directive")
        if directive is not None and directive.mode == "action":
            return "action_validate"
        return "model_decide_next_step"

    def _action_validate_node(self, state: BattleSubagentState) -> dict[str, Any]:
        context = state["current_context"]
        directive = state["directive"]
        commands = [str(command) for command in directive.commands]
        validator = self._tools["validate_battle_command"]
        validation = validator.run(context, {"commands": commands})
        if bool(validation.get("is_valid")):
            return {"pending_commands": commands}

        fallback_commands = self._build_guardrail_fallback_commands(context)
        if fallback_commands:
            fallback_validation = validator.run(context, {"commands": fallback_commands})
            if bool(fallback_validation.get("is_valid")):
                working_memory = self._append_step_summary(dict(state["working_memory"]), "validation failed; using guardrail fallback")
                return {
                    "pending_commands": fallback_commands,
                    "working_memory": working_memory,
                }

        working_memory = self._append_step_summary(
            dict(state["working_memory"]),
            f"validation failed with no fallback: {validation.get('errors', [])}",
        )
        return {
            "pending_commands": [],
            "working_memory": working_memory,
        }

    def _action_commit_node(self, state: BattleSubagentState) -> dict[str, Any]:
        commands = [str(command) for command in state.get("pending_commands", []) if str(command).strip()]
        if not commands:
            return {"action_committed": False}

        context = state["current_context"]
        executor = self._tools["execute_battle_command"]
        result = executor.run(context, {"commands": commands})
        if not bool(result.get("executed")):
            working_memory = self._append_step_summary(dict(state["working_memory"]), f"execution failed: {result.get('validation')}")
            return {"working_memory": working_memory, "action_committed": False}

        working_memory = dict(state["working_memory"])
        working_memory.setdefault("executed_command_batches", []).append(commands)
        working_memory = self._append_step_summary(working_memory, f"executed command batch: {', '.join(commands)}")
        self._record_accepted_decision(
            context,
            commands,
            str(state.get("pending_decision_explanation", "")),
            float(state.get("pending_decision_confidence", 0.0)),
            list(state.get("pending_required_tools_used", [])),
        )
        return {
            "current_state": cast(GameState, result.get("state", state["current_state"])),
            "working_memory": working_memory,
            "pending_required_tools_used": [],
            "action_committed": True,
            "executed_commands": list(working_memory.get("executed_command_batches", [])),
            "steps": len(working_memory.get("executed_command_batches", [])),
        }

    def _exit_check_node(self, state: BattleSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        current_state = runtime.current_state()
        battle_complete = not is_battle_scope_state(current_state)
        return {
            "current_state": current_state,
            "battle_complete": battle_complete,
            "handled": bool(state.get("action_committed", False)) or battle_complete,
        }

    def _route_after_exit_check(self, state: BattleSubagentState) -> Literal["state_ingest", "session_finalize"]:
        if bool(state.get("battle_complete")):
            return "session_finalize"
        return "state_ingest"

    def _session_finalize_node(self, state: BattleSubagentState) -> dict[str, Any]:
        runtime = state["runtime"]
        final_state = runtime.current_state()
        working_memory = dict(state["working_memory"])
        final_context = build_battle_agent_context(
            final_state,
            "BattleHandler",
            working_memory=working_memory,
            runtime=runtime,
            retrieved_episodic_memories=str(working_memory.get("retrieved_episodic_memories", "none")),
            retrieved_semantic_memories=str(working_memory.get("retrieved_semantic_memories", "none")),
            langmem_status=str(working_memory.get("langmem_status", self._langmem_service.status())),
        )
        final_summary = self._build_final_summary(final_state, working_memory)
        if final_summary.strip():
            self._langmem_service.record_custom_memory(
                final_context,
                final_summary,
                tags=("battle_summary", "BattleHandler"),
                reflect=True,
            )
        log_to_run(f"BattleSubagent session ended: {state.get('session_id', '')}")
        handled = bool(state.get("handled", False))
        if is_battle_scope_state(final_state) and not state.get("executed_commands"):
            handled = False
        return {
            "handled": handled,
            "final_state": final_state,
            "executed_commands": list(working_memory.get("executed_command_batches", [])),
            "final_summary": final_summary,
            "steps": len(working_memory.get("executed_command_batches", [])),
        }

    def _tool_descriptions(self) -> dict[str, str]:
        return {
            "enumerate_legal_actions": "List currently legal battle commands for the exact text state.",
            "analyze_with_calculator": "Ask the preserved rs.calculator engine for a legacy recommended command batch.",
            "validate_battle_command": "Validate one command or command batch against the exact current battle state.",
            "execute_battle_command": "Execute a validated command batch through the live battle runtime and return the next state.",
            "retrieve_battle_experience": "Retrieve relevant LangMem episodic and semantic battle memories.",
        }

    def _build_guardrail_fallback_commands(self, context: AgentContext) -> list[str]:
        calculator_tool = self._tools.get("analyze_with_calculator")
        validator_tool = self._tools.get("validate_battle_command")
        if calculator_tool is not None and validator_tool is not None:
            calculator_result = calculator_tool.run(context, {"max_path_count": self._config.fallback_max_path_count})
            recommended_commands = [str(command) for command in calculator_result.get("recommended_commands", [])]
            if recommended_commands:
                validation = validator_tool.run(context, {"commands": recommended_commands})
                if bool(validation.get("is_valid")):
                    return recommended_commands

        enumerate_tool = self._tools.get("enumerate_legal_actions")
        if enumerate_tool is not None and validator_tool is not None:
            legal = enumerate_tool.run(context, {})
            for command in legal.get("commands", []):
                candidate = [str(command)]
                validation = validator_tool.run(context, {"commands": candidate})
                if bool(validation.get("is_valid")):
                    return candidate

        return []

    def _record_accepted_decision(
            self,
            context: AgentContext,
            commands: list[str],
            explanation: str,
            confidence: float,
            required_tools_used: list[str],
    ) -> None:
        decision = AgentDecision(
            proposed_command=" | ".join(commands),
            confidence=confidence,
            explanation=explanation or "battle_subagent_step",
            required_tools_used=list(required_tools_used),
            fallback_recommended=False,
            metadata={"battle_session": context.extras.get("battle_working_memory", {}).get("session_id")},
        )
        self._langmem_service.record_accepted_decision(context, decision)

    def _build_final_summary(self, final_state: GameState, working_memory: dict[str, Any]) -> str:
        game_state = final_state.game_state()
        player_hp = game_state.get("current_hp", "unknown")
        max_hp = game_state.get("max_hp", "unknown")
        floor = game_state.get("floor", "unknown")
        room_phase = game_state.get("room_phase", "unknown")
        steps = len(working_memory.get("executed_command_batches", []))
        recent_steps = " | ".join(working_memory.get("recent_step_summaries", [])[-4:])
        return (
            f"Battle session ended on floor {floor} with room_phase={room_phase}. "
            f"Player HP {player_hp}/{max_hp}. "
            f"Executed {steps} command batches. "
            f"Recent battle notes: {recent_steps or 'none'}"
        )

    @staticmethod
    def _append_tool_result(working_memory: dict[str, Any], tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        result_preview = {
            "tool_name": tool_name,
            "summary": str(result.get("summary", "")) or str(result.get("recommended_commands", result.get("commands", ""))),
        }
        working_memory.setdefault("recent_tool_results", []).append(result_preview)
        working_memory["recent_tool_results"] = working_memory["recent_tool_results"][-8:]
        return working_memory

    @staticmethod
    def _append_step_summary(working_memory: dict[str, Any], summary: str) -> dict[str, Any]:
        working_memory.setdefault("recent_step_summaries", []).append(summary)
        working_memory["recent_step_summaries"] = working_memory["recent_step_summaries"][-12:]
        return working_memory
