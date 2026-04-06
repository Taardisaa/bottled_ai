from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rs.helper.logger import log_to_run
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.battle_runtime import BattleRuntimeAdapter, BattleSessionResult
from rs.llm.battle_tools import (
    AnalyzeWithCalculatorTool,
    EnumerateLegalActionsTool,
    ValidateBattleCommandTool,
    analyze_with_calculator,
    enumerate_legal_actions,
    make_retrieve_battle_experience_tool,
    submit_battle_commands,
    validate_battle_command,
)
from rs.llm.integration.battle_context import build_battle_agent_context, is_battle_scope_state
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.battle_llm_provider import build_battle_prompt
from rs.llm.subagent_validation_middleware import validate_proposed_command
from rs.machine.state import GameState


class BattleSubagentState(dict):
    """TypedDict-compatible state class. Using dict subclass so add_messages reducer works."""


# Actual TypedDict definition for type-checker usage
try:
    from typing import TypedDict

    class BattleSubagentState(TypedDict, total=False):  # type: ignore[no-redef]
        runtime: BattleRuntimeAdapter
        current_state: GameState
        current_state_signature: str
        current_context: AgentContext
        working_memory: dict[str, Any]
        session_id: str
        messages: Annotated[list, add_messages]
        pending_commands: list[str]
        pending_decision_explanation: str
        pending_decision_confidence: float
        action_committed: bool
        battle_complete: bool
        skip_agent: bool
        validation_attempt_count: int
        handled: bool
        final_state: GameState
        executed_commands: list[list[str]]
        final_summary: str
        steps: int
except Exception:
    pass


@dataclass
class BattleSubagentConfig:
    max_decision_loops: int = 32
    max_tool_calls: int = 24
    fallback_max_path_count: int = 200
    max_validation_attempts: int = 2
    no_progress_limit: int = 2


@dataclass
class BattleWorkingMemory:
    session_id: str
    recent_step_summaries: list[str] = field(default_factory=list)
    executed_command_batches: list[list[str]] = field(default_factory=list)
    retrieved_episodic_memories: str = "none"
    retrieved_semantic_memories: str = "none"
    langmem_status: str = "disabled_by_config"
    no_progress_count: int = 0
    last_state_signature: str = ""
    last_executed_batch: list[str] = field(default_factory=list)
    battle_start_hp: int | None = None
    battle_start_max_hp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recent_step_summaries": list(self.recent_step_summaries),
            "executed_command_batches": [list(batch) for batch in self.executed_command_batches],
            "retrieved_episodic_memories": self.retrieved_episodic_memories,
            "retrieved_semantic_memories": self.retrieved_semantic_memories,
            "langmem_status": self.langmem_status,
            "no_progress_count": self.no_progress_count,
            "last_state_signature": self.last_state_signature,
            "last_executed_batch": list(self.last_executed_batch),
            "battle_start_hp": self.battle_start_hp,
            "battle_start_max_hp": self.battle_start_max_hp,
        }


@dataclass
class BattleSessionResult:
    handled: bool
    final_state: GameState | None = None
    session_id: str = ""
    executed_commands: list[list[str]] = field(default_factory=list)
    steps: int = 0
    summary: str = ""


class BattleSubagent:
    def __init__(
            self,
            *,
            chat_model: Any | None = None,
            langmem_service: LangMemService | None = None,
            config: BattleSubagentConfig | None = None,
    ):
        self._langmem_service = get_langmem_service() if langmem_service is None else langmem_service
        self._config = BattleSubagentConfig() if config is None else config

        if chat_model is None:
            from langchain_openai import ChatOpenAI
            from rs.utils.config import config as llm_config
            chat_model = ChatOpenAI(
                model=llm_config.fast_llm_model,
                base_url=llm_config.llm_base_url or llm_config.openai_base_url,
                api_key=llm_config.llm_api_key or llm_config.openai_key,
                temperature=0.6,
            )

        retrieve_tool = make_retrieve_battle_experience_tool(self._langmem_service)
        self._langgraph_tools = [
            validate_battle_command,
            retrieve_tool,
            submit_battle_commands,
        ]

        self._model = chat_model
        self._model_with_tools = chat_model.bind_tools(self._langgraph_tools)
        self._tool_node = ToolNode(self._langgraph_tools)

        # Implementation instances for direct guardrail use (not via LLM)
        self._enumerate_tool = EnumerateLegalActionsTool()
        self._calculator_tool = AnalyzeWithCalculatorTool()
        self._validator_tool = ValidateBattleCommandTool()

        self._compiled_graph: Any | None = None

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
            graph.add_node("think_node", self._think_node)
            graph.add_node("agent_node", self._agent_node)
            graph.add_node("tool_node", self._tool_node)
            graph.add_node("action_validate", self._action_validate_node)
            graph.add_node("action_commit", self._action_commit_node)
            graph.add_node("exit_check", self._exit_check_node)
            graph.add_node("session_finalize", self._session_finalize_node)

            graph.add_edge(START, "session_bootstrap")
            graph.add_edge("session_bootstrap", "retrieve_battle_experience")
            graph.add_edge("retrieve_battle_experience", "state_ingest")
            graph.add_conditional_edges(
                "state_ingest",
                self._route_after_state_ingest,
                {
                    "think_node": "think_node",
                    "action_commit": "action_commit",
                    "session_finalize": "session_finalize",
                },
            )
            graph.add_edge("think_node", "agent_node")
            graph.add_conditional_edges(
                "agent_node",
                self._should_continue,
                {
                    "tools": "tool_node",
                    "action_validate": "action_validate",
                    "session_finalize": "session_finalize",
                },
            )
            graph.add_edge("tool_node", "agent_node")
            graph.add_conditional_edges(
                "action_validate",
                self._route_after_action_validate,
                {
                    "agent_node": "agent_node",
                    "action_commit": "action_commit",
                    "session_finalize": "session_finalize",
                },
            )
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

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _session_bootstrap_node(self, state: BattleSubagentState) -> dict[str, Any]:
        self._langmem_service.pause_reflections()
        session_id = uuid4().hex
        working_memory = BattleWorkingMemory(session_id=session_id)
        initial_game_state = state.get("current_state")
        if initial_game_state is not None:
            gs = initial_game_state.game_state()
            working_memory.battle_start_hp = gs.get("current_hp")
            working_memory.battle_start_max_hp = gs.get("max_hp")
        log_to_run(f"BattleSubagent session started: {session_id}")
        return {
            "session_id": session_id,
            "working_memory": working_memory.to_dict(),
            "messages": [],
            "pending_commands": [],
            "executed_commands": [],
            "steps": 0,
            "handled": False,
            "battle_complete": False,
            "action_committed": False,
            "skip_agent": False,
            "validation_attempt_count": 0,
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
        log_to_run(
            "BattleSubagent memory loaded: "
            f"status={working_memory['langmem_status']} | "
            f"episodic={self._preview_log_text(str(working_memory['retrieved_episodic_memories']))} | "
            f"semantic={self._preview_log_text(str(working_memory['retrieved_semantic_memories']))}"
        )
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
        current_signature = self._state_signature(current_state)
        if not str(working_memory.get("last_state_signature", "")).strip():
            working_memory["last_state_signature"] = current_signature
        battle_complete = not is_battle_scope_state(current_state)

        # No-progress guardrail — bypass the agent entirely
        no_progress_count = int(working_memory.get("no_progress_count", 0))
        if not battle_complete and no_progress_count >= self._config.no_progress_limit:
            last_batch = [str(c) for c in working_memory.get("last_executed_batch", []) if str(c).strip()]
            fallback_commands = self._build_guardrail_fallback_commands(
                context, prefer_progression=True, rejected_batches=[last_batch] if last_batch else None
            )
            working_memory = self._append_step_summary(
                working_memory,
                f"no-progress guardrail triggered after {no_progress_count} repeated signatures",
            )
            return {
                "current_state": current_state,
                "current_state_signature": current_signature,
                "current_context": context,
                "working_memory": working_memory,
                "battle_complete": battle_complete,
                "action_committed": False,
                "validation_attempt_count": 0,
                "skip_agent": True,
                "pending_commands": fallback_commands,
                "pending_decision_explanation": "battle_subagent_no_progress_guardrail",
                "pending_decision_confidence": 0.25,
            }

        # Pre-compute dynamic context: legal actions and calculator recommendation
        if not battle_complete:
            legal_actions = self._enumerate_tool.run(context, {})
            working_memory["legal_actions"] = legal_actions
            try:
                calc_result = self._calculator_tool.run(
                    context, {"max_path_count": self._config.fallback_max_path_count}
                )
                working_memory["calculator_recommendation"] = calc_result.get("recommended_commands", [])
            except Exception:
                working_memory["calculator_recommendation"] = []

        # Normal path — reset messages for this turn and let agent decide
        log_to_run(
            "BattleSubagent state ingest: "
            f"floor={context.game_state.get('floor', 'unknown')} | "
            f"turn={context.game_state.get('turn', 'unknown')} | "
            f"hp={context.game_state.get('current_hp', 'unknown')}/{context.game_state.get('max_hp', 'unknown')} | "
            f"commands={len(context.available_commands)} | "
            f"choices={len(context.choice_list)}"
        )
        prompt = build_battle_prompt(context, working_memory)
        existing_messages = state.get("messages", [])
        removals = [RemoveMessage(id=m.id) for m in existing_messages if getattr(m, "id", None)]
        new_message = HumanMessage(content=prompt)

        return {
            "current_state": current_state,
            "current_state_signature": current_signature,
            "current_context": context,
            "working_memory": working_memory,
            "battle_complete": battle_complete,
            "action_committed": False,
            "messages": removals + [new_message],
            "validation_attempt_count": 0,
            "skip_agent": False,
            "pending_commands": [],
            "pending_decision_explanation": "",
            "pending_decision_confidence": 0.0,
        }

    def _think_node(self, state: BattleSubagentState) -> dict[str, Any]:
        import time as _time
        messages = list(state.get("messages", []))
        think_messages = messages + [HumanMessage(
            content="Analyse the current battle state. What are the key threats, opportunities, "
                    "and your plan for this turn? Do NOT call any tools — just reason."
        )]
        _t0 = _time.perf_counter()
        response = self._model.invoke(think_messages)
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        reasoning = ""
        if isinstance(response.content, str):
            reasoning = response.content.strip()
        elif isinstance(response.content, list):
            reasoning = " ".join(
                str(b.get("text", "")) for b in response.content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        log_to_run(
            f"[TIMING] BattleSubagent._think_node took {_elapsed_ms:.0f}ms | "
            f"reasoning={self._preview_log_text(reasoning, limit=-1)}"
        )
        if reasoning:
            return {"messages": [AIMessage(content=reasoning)]}
        return {}

    def _agent_node(self, state: BattleSubagentState) -> dict[str, Any]:
        import time as _time
        messages = list(state.get("messages", []))
        _t0 = _time.perf_counter()
        response = self._model_with_tools.invoke(messages)
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        tool_call_names = [str(tc.get("name", "")) for tc in getattr(response, "tool_calls", []) if str(tc.get("name", "")).strip()]
        log_to_run(
            f"[TIMING] BattleSubagent._agent_node LLM call took {_elapsed_ms:.0f}ms | "
            f"tool_calls={tool_call_names or ['none']} | "
            f"content={self._preview_log_text(self._extract_explanation([response]), limit=-1)}"
        )
        return {"messages": [response]}

    def _action_validate_node(self, state: BattleSubagentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        working_memory = dict(state.get("working_memory", {}))
        context = state["current_context"]
        validation_attempt = int(state.get("validation_attempt_count", 0))

        commands = self._parse_submitted_commands(messages)

        # Enforce one-command-at-a-time: keep only the first command
        if len(commands) > 1:
            rejected = commands[1:]
            commands = commands[:1]
            working_memory = self._append_step_summary(
                working_memory,
                f"truncated to 1 command; rejected {rejected}",
            )
            log_to_run(
                f"BattleSubagent truncated submission to first command: "
                f"accepted={commands}, rejected={rejected}"
            )

        if not commands:
            fallback = self._build_guardrail_fallback_commands(context)
            if fallback:
                working_memory = self._append_step_summary(working_memory, "no submission, using guardrail fallback")
                log_to_run(
                    "BattleSubagent validation produced no commands; "
                    f"using fallback={fallback}"
                )
                return {
                    "pending_commands": fallback,
                    "working_memory": working_memory,
                    "validation_attempt_count": 0,
                    "pending_decision_explanation": "guardrail_no_submission",
                    "pending_decision_confidence": 0.25,
                }
            working_memory = self._append_step_summary(working_memory, "no submission, no guardrail fallback")
            log_to_run("BattleSubagent validation produced no commands and no fallback was available")
            return {"pending_commands": [], "working_memory": working_memory, "validation_attempt_count": 0}

        # Validate submitted commands
        feedback: dict[str, Any] | None = None
        for command in commands:
            if not command.startswith("choose "):
                continue
            command_validation, command_feedback = validate_proposed_command(context, command)
            if not command_validation.is_valid:
                feedback = dict(command_feedback or {})
                break

        if feedback is None:
            validation = self._validator_tool.run(context, {"commands": commands})
            if bool(validation.get("is_valid")):
                explanation = self._extract_explanation(messages)
                log_to_run(
                    "BattleSubagent accepted submission: "
                    f"commands={commands} | "
                    f"explanation={self._preview_log_text(explanation, limit=-1)}"
                )
                return {
                    "pending_commands": commands,
                    "working_memory": working_memory,
                    "validation_attempt_count": 0,
                    "pending_decision_explanation": explanation,
                    "pending_decision_confidence": 0.8,
                }
            feedback = {
                "rejected_command": " | ".join(commands),
                "code": "batch_validation_failed",
                "message": str(validation.get("errors", [])),
                "corrective_hint": "Use validate_battle_command before calling submit_battle_commands.",
            }

        working_memory = self._append_step_summary(
            working_memory,
            f"validation failed for submitted commands: {feedback.get('code', 'unknown')}",
        )
        log_to_run(
            "BattleSubagent validation failed: "
            f"commands={commands} | "
            f"code={feedback.get('code', 'unknown')} | "
            f"message={self._preview_log_text(str(feedback.get('message', '')))}"
        )

        if validation_attempt + 1 < self._config.max_validation_attempts:
            corrective = HumanMessage(content=json.dumps(
                {
                    "validation_error": feedback,
                    "instruction": (
                        "Your last submit_battle_commands call was rejected. "
                        "Use validate_battle_command to verify your commands first, "
                        "then call submit_battle_commands again with valid commands."
                    ),
                },
                ensure_ascii=False,
            ))
            return {
                "messages": [corrective],
                "working_memory": working_memory,
                "validation_attempt_count": validation_attempt + 1,
                "pending_commands": [],
            }

        # Exhausted retries — use guardrail
        fallback = self._build_guardrail_fallback_commands(context, rejected_batches=[list(commands)])
        if fallback:
            working_memory = self._append_step_summary(working_memory, "validation exhausted, using guardrail fallback")
            log_to_run(
                f"BattleSubagent replaced invalid submission with guardrail fallback: "
                f"rejected={commands}, reason={feedback.get('code', 'unknown')}, fallback={fallback}"
            )
            return {
                "pending_commands": fallback,
                "working_memory": working_memory,
                "validation_attempt_count": 0,
                "pending_decision_explanation": "guardrail_validation_exhausted",
                "pending_decision_confidence": 0.25,
            }

        working_memory = self._append_step_summary(working_memory, "validation failed with no fallback")
        return {"pending_commands": [], "working_memory": working_memory, "validation_attempt_count": 0}

    def _action_commit_node(self, state: BattleSubagentState) -> dict[str, Any]:
        commands = [str(c) for c in state.get("pending_commands", []) if str(c).strip()]
        if not commands:
            return {"action_committed": False}

        context = state["current_context"]
        runtime = state["runtime"]
        log_to_run(f"BattleSubagent executing commands: {commands}")

        try:
            next_state = runtime.execute(commands)
        except Exception as exc:
            working_memory = self._append_step_summary(
                dict(state.get("working_memory", {})), f"execution failed: {exc}"
            )
            log_to_run(f"BattleSubagent execution failed: commands={commands}, error={exc}")
            return {
                "working_memory": working_memory,
                "action_committed": False,
                "current_state": runtime.current_state(),
            }

        working_memory = dict(state.get("working_memory", {}))
        previous_signature = str(state.get("current_state_signature", ""))
        next_signature = self._state_signature(next_state)
        no_progress_count = int(working_memory.get("no_progress_count", 0))
        if previous_signature and previous_signature == next_signature:
            no_progress_count += 1
            working_memory = self._append_step_summary(
                working_memory, f"no progress detected after commands: {', '.join(commands)}"
            )
        else:
            no_progress_count = 0
        working_memory["no_progress_count"] = no_progress_count
        working_memory["last_state_signature"] = next_signature
        working_memory["last_executed_batch"] = list(commands)
        working_memory.setdefault("executed_command_batches", []).append(commands)
        working_memory = self._append_step_summary(working_memory, f"executed command batch: {', '.join(commands)}")
        log_to_run(
            "BattleSubagent post-execution state: "
            f"screen={next_state.screen_type()} | "
            f"floor={next_state.floor()} | "
            f"signature={self._preview_log_text(next_signature, limit=96)}"
        )
        self._record_accepted_decision(
            context,
            commands,
            str(state.get("pending_decision_explanation", "")),
            float(state.get("pending_decision_confidence", 0.0)),
            self._extract_tool_names_used(list(state.get("messages", []))),
        )
        return {
            "current_state": next_state,
            "current_state_signature": next_signature,
            "working_memory": working_memory,
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
        self._langmem_service.resume_reflections()
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

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------

    def _route_after_state_ingest(
            self, state: BattleSubagentState
    ) -> Literal["think_node", "action_commit", "session_finalize"]:
        if bool(state.get("battle_complete")):
            return "session_finalize"
        if bool(state.get("skip_agent")):
            return "action_commit"
        return "think_node"

    def _should_continue(
            self, state: BattleSubagentState
    ) -> Literal["tools", "action_validate", "session_finalize"]:
        if bool(state.get("battle_complete")):
            return "session_finalize"

        messages = state.get("messages", [])
        if not messages:
            return "session_finalize"

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        if not tool_calls:
            return "action_validate"

        for tc in tool_calls:
            if tc.get("name") == "submit_battle_commands":
                return "action_validate"

        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        ai_message_count = sum(1 for m in messages if isinstance(m, AIMessage))
        if (
            tool_message_count >= self._config.max_tool_calls
            or ai_message_count >= self._config.max_decision_loops
        ):
            return "action_validate"

        return "tools"

    def _route_after_action_validate(
            self, state: BattleSubagentState
    ) -> Literal["agent_node", "action_commit", "session_finalize"]:
        if int(state.get("validation_attempt_count", 0)) > 0:
            return "agent_node"
        if state.get("pending_commands"):
            return "action_commit"
        return "session_finalize"

    def _route_after_exit_check(
            self, state: BattleSubagentState
    ) -> Literal["state_ingest", "session_finalize"]:
        if bool(state.get("battle_complete")):
            return "session_finalize"
        return "state_ingest"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_guardrail_fallback_commands(
            self,
            context: AgentContext,
            *,
            prefer_progression: bool = False,
            rejected_batches: list[list[str]] | None = None,
    ) -> list[str]:
        rejected = {
            tuple(str(c).strip() for c in batch if str(c).strip())
            for batch in (rejected_batches or [])
            if batch
        }

        if prefer_progression:
            for candidate in [["confirm"], ["proceed"]]:
                normalized = tuple(candidate)
                if normalized in rejected:
                    continue
                validation = self._validator_tool.run(context, {"commands": candidate})
                if bool(validation.get("is_valid")):
                    return candidate

        calculator_result = self._calculator_tool.run(context, {"max_path_count": self._config.fallback_max_path_count})
        recommended = [str(c) for c in calculator_result.get("recommended_commands", [])]
        if recommended:
            normalized = tuple(c.strip() for c in recommended if c.strip())
            if normalized in rejected:
                recommended = []
            elif prefer_progression and self._is_choose_only_batch(recommended):
                recommended = []
        if recommended:
            validation = self._validator_tool.run(context, {"commands": recommended})
            if bool(validation.get("is_valid")):
                return recommended

        legal = self._enumerate_tool.run(context, {})
        commands = [str(c) for c in legal.get("commands", [])]
        if prefer_progression:
            commands = sorted(commands, key=lambda c: 1 if c.startswith("choose ") else 0)
        for command in commands:
            candidate = [command]
            normalized = tuple(item.strip() for item in candidate if item.strip())
            if normalized in rejected:
                continue
            validation = self._validator_tool.run(context, {"commands": candidate})
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
        room_type = game_state.get("room_type", "MONSTER")
        combat_state = final_state.combat_state() or {}
        turn = combat_state.get("turn", "?")
        monsters = [m for m in final_state.get_monsters() if not m.get("is_gone", False)]
        if monsters:
            m = monsters[0]
            enemy_desc = f"{m.get('name', 'enemy')} HP {m.get('current_hp', '?')}/{m.get('max_hp', '?')}"
        else:
            enemy_desc = "no surviving enemy"
        start_hp = working_memory.get("battle_start_hp")
        if start_hp is not None and start_hp != player_hp:
            hp_str = f"HP {start_hp}→{player_hp}/{max_hp}"
        else:
            hp_str = f"HP {player_hp}/{max_hp}"
        batches = working_memory.get("executed_command_batches", [])
        commands_str = ", ".join(
            "[" + " | ".join(batch) + "]" for batch in batches[-8:]
        ) or "none"
        recent_steps = " | ".join(working_memory.get("recent_step_summaries", [])[-4:])
        return (
            f"Floor {floor} {room_type} battle, {turn} turns. "
            f"Player {hp_str}. {enemy_desc}.\n"
            f"Commands executed: {commands_str}\n"
            f"Step notes: {recent_steps or 'none'}\n"
            f"Review: Were the right plays chosen given the enemy's HP and intent? "
            f"Were cards and energy used efficiently this fight? "
            f"What could be done differently to win the fight more efficiently?"
        )

    @staticmethod
    def _parse_submitted_commands(messages: list) -> list[str]:
        for message in reversed(messages):
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            for tc in message.tool_calls:
                if tc.get("name") == "submit_battle_commands":
                    args = tc.get("args", {})
                    if isinstance(args, dict):
                        cmds = args.get("commands", [])
                        return [str(c).strip() for c in cmds if str(c).strip()]
        return []

    @staticmethod
    def _extract_explanation(messages: list) -> str:
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            # Try reasoning from submit_battle_commands tool call args first
            for tc in getattr(message, "tool_calls", []):
                if tc.get("name") == "submit_battle_commands":
                    reasoning = str(tc.get("args", {}).get("reasoning", "")).strip()
                    if reasoning:
                        return reasoning
            # Fall back to message content (skip raw <tool_call> XML)
            content = message.content
            if isinstance(content, str):
                text = content.strip()
                if text and not text.startswith("<tool_call>"):
                    return text
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text", "")).strip()
                        if text and not text.startswith("<tool_call>"):
                            return text
        return "battle_subagent_step"

    @staticmethod
    def _extract_tool_names_used(messages: list) -> list[str]:
        return [
            m.name for m in messages
            if isinstance(m, ToolMessage) and m.name and m.name != "submit_battle_commands"
        ]

    @staticmethod
    def _append_step_summary(working_memory: dict[str, Any], summary: str) -> dict[str, Any]:
        working_memory.setdefault("recent_step_summaries", []).append(summary)
        working_memory["recent_step_summaries"] = working_memory["recent_step_summaries"][-12:]
        return working_memory

    @staticmethod
    def _preview_log_text(value: str, limit: int = 160) -> str:
        normalized = " ".join(str(value).split())
        if limit < 0:
            return normalized
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit - 3] + "..."

    @staticmethod
    def _state_signature(state: GameState) -> str:
        game_state = state.game_state()
        screen_state_raw = game_state.get("screen_state", {})
        screen_state = screen_state_raw if isinstance(screen_state_raw, dict) else {}
        selected = screen_state.get("selected", [])
        selected_marker: tuple[str, ...] = tuple(
            f"{str(card.get('uuid', ''))}:{str(card.get('id', ''))}:{str(card.get('name', ''))}"
            for card in selected
            if isinstance(card, dict)
        ) if isinstance(selected, list) else tuple()
        available_commands_raw = state.json.get("available_commands", [])
        available_commands = tuple(
            sorted(str(c).strip() for c in available_commands_raw if str(c).strip())
        ) if isinstance(available_commands_raw, list) else tuple()
        choice_list = tuple(str(choice).strip().lower() for choice in state.get_choice_list())
        signature = (
            str(game_state.get("screen_type", "")),
            available_commands,
            str(game_state.get("current_action", "")),
            int(screen_state.get("max_cards", 0)) if isinstance(screen_state.get("max_cards", 0), int) else 0,
            bool(screen_state.get("can_pick_zero", False)),
            selected_marker,
            choice_list,
        )
        return str(signature)

    @staticmethod
    def _is_choose_only_batch(commands: list[str]) -> bool:
        normalized = [str(c).strip() for c in commands if str(c).strip()]
        return bool(normalized) and all(c.startswith("choose ") for c in normalized)
