from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, Literal, TypedDict, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from rs.game.path import PathHandlerConfig
from rs.game.screen_type import ScreenType
from rs.llm.action_executor import ActionExecutionResult, UnifiedActionExecutor
from rs.llm.action_policy_registry import ActionPolicyRegistry
from rs.llm.battle_runtime import BattleRuntimeAdapter, BattleSessionResult
from rs.llm.battle_subagent import BattleSubagent
from rs.llm.campfire_subagent import CampfireSessionResult, CampfireSubagent
from rs.llm.battle_tools import (
    AnalyzeWithCalculatorTool,
    EnumerateLegalActionsTool,
    ExecuteBattleCommandTool,
    RetrieveBattleExperienceTool,
    ValidateBattleCommandTool,
)
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig, load_llm_config
from rs.llm.graph_trace import build_graph_trace_record, mirror_graph_trace_to_run_log, write_graph_trace
from rs.llm.integration.astrolabe_transform_context import build_astrolabe_transform_agent_context
from rs.llm.integration.battle_context import build_battle_agent_context, is_battle_scope_state
from rs.llm.integration.campfire_context import build_campfire_agent_context
from rs.llm.integration.boss_reward_context import build_boss_reward_agent_context, is_astrolabe_transform_state
from rs.llm.integration.combat_reward_context import build_combat_reward_agent_context
from rs.llm.integration.generic_context import build_generic_unhandled_agent_context
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProvider
from rs.llm.providers.event_llm_provider import EventLlmProvider
from rs.llm.providers.generic_llm_provider import GenericLlmProvider
from rs.llm.providers.map_llm_provider import MapLlmProvider
from rs.llm.providers.shop_purchase_llm_provider import ShopPurchaseLlmProvider
from rs.llm.subagent_validation_middleware import validate_proposed_command
from rs.llm.telemetry import build_decision_telemetry, write_decision_telemetry
from rs.llm.validator import validate_command
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.llm.integration.event_context import build_event_agent_context
from rs.llm.integration.map_context import build_map_agent_context
from rs.llm.integration.shop_purchase_context import build_shop_purchase_agent_context
from rs.llm.reward_subagent import (
    AstrolabeTransformSubagent,
    BossRewardSubagent,
    CombatRewardSubagent,
    RewardSessionResult,
)
from rs.machine.command import Command
from rs.machine.state import GameState


GraphRoute = Literal[
    "decide_event",
    "decide_shop",
    "decide_card_reward",
    "decide_map",
    "decide_generic",
    "decide_fallback",
]

PROTOCOL_META_COMMANDS = {"key", "click", "wait", "state"}


@dataclass
class GraphExecutionResult:
    handled: bool
    commands: list[str] | None = None
    final_state: GameState | None = None
    battle_session: BattleSessionResult | None = None
    reward_session: RewardSessionResult | None = None
    campfire_session: CampfireSessionResult | None = None


class AIPlayerGraphState(TypedDict, total=False):
    context_payload: Dict[str, Any]
    decision_context_payload: Dict[str, Any]
    handler_name: str
    route_name: GraphRoute
    run_id: str
    screen_type: str
    floor: int | None
    act: int | None
    distilled_run_summary: str
    recent_key_decisions: list[dict[str, Any]]
    retrieved_episodic_memories: str
    retrieved_semantic_memories: str
    langmem_status: str
    proposed_command: str | None
    confidence: float
    explanation: str
    metadata: Dict[str, Any]
    decision_valid: bool
    validation_code: str
    validation_message: str
    commands: list[str] | None


class AIPlayerGraph:
    """Unified LangGraph-driven LLM decision engine for a single run thread."""

    def __init__(
            self,
            config: LlmConfig | None = None,
            langmem_service: LangMemService | None = None,
            battle_subagent: BattleSubagent | None = None,
            campfire_subagent: CampfireSubagent | None = None,
            combat_reward_subagent: CombatRewardSubagent | None = None,
            boss_reward_subagent: BossRewardSubagent | None = None,
            astrolabe_transform_subagent: AstrolabeTransformSubagent | None = None,
    ):
        self._config = load_llm_config() if config is None else config
        self._langmem_service = get_langmem_service(self._config) if langmem_service is None else langmem_service
        self._checkpointer = InMemorySaver()
        self._compiled_graph: Any | None = None
        self._map_handler_config = PathHandlerConfig()
        self._event_provider = EventLlmProvider()
        self._shop_provider = ShopPurchaseLlmProvider()
        self._card_reward_provider = CardRewardLlmProvider()
        self._map_provider = MapLlmProvider()
        self._generic_provider = GenericLlmProvider()
        self._battle_subagent = BattleSubagent(langmem_service=self._langmem_service) if battle_subagent is None else battle_subagent
        self._campfire_subagent = (
            CampfireSubagent(langmem_service=self._langmem_service)
            if campfire_subagent is None else campfire_subagent
        )
        self._combat_reward_subagent = (
            CombatRewardSubagent(langmem_service=self._langmem_service)
            if combat_reward_subagent is None else combat_reward_subagent
        )
        self._boss_reward_subagent = (
            BossRewardSubagent(langmem_service=self._langmem_service)
            if boss_reward_subagent is None else boss_reward_subagent
        )
        self._astrolabe_transform_subagent = (
            AstrolabeTransformSubagent(langmem_service=self._langmem_service)
            if astrolabe_transform_subagent is None else astrolabe_transform_subagent
        )
        self._action_policy = ActionPolicyRegistry()
        self._last_card_reward_skip: dict[str, Any] | None = None
        self._last_card_reward_skip_consumed: bool = False
        self._action_executor = UnifiedActionExecutor(
            battle_runner=self,
            campfire_runner=self,
            combat_reward_runner=self,
            boss_reward_runner=self,
            astrolabe_runner=self,
            context_builder=self._build_context,
        )
        if battle_subagent is None:
            self._battle_subagent.register_tool(EnumerateLegalActionsTool())
            self._battle_subagent.register_tool(AnalyzeWithCalculatorTool())
            self._battle_subagent.register_tool(ValidateBattleCommandTool())
            self._battle_subagent.register_tool(ExecuteBattleCommandTool())
            self._battle_subagent.register_tool(RetrieveBattleExperienceTool(self._langmem_service))

    def is_enabled(self) -> bool:
        return self._config.enabled and self._config.ai_player_graph_enabled

    def can_handle(self, state: GameState) -> bool:
        if self._deterministic_single_command(state) is not None:
            return True
        return self._build_context(state) is not None

    def decide(self, state: GameState) -> list[str] | None:
        if not self.is_enabled():
            self._trace_raw_event(event_type="graph_disabled", summary="ai player graph disabled by config")
            return None

        deterministic_commands = self._deterministic_single_command(state)
        if deterministic_commands is not None:
            self._trace_raw_event(
                event_type="graph_decide_success",
                screen_type=state.screen_type(),
                summary=f"deterministic_single_command commands={deterministic_commands}",
            )
            return deterministic_commands

        context = self._build_context(state)
        if context is None:
            self._trace_raw_event(
                event_type="graph_unhandled_state",
                screen_type=state.screen_type(),
                summary="state not handled by ai player graph",
            )
            return None

        if context.handler_name in {
            "BattleHandler",
            "CampfireHandler",
            "CombatRewardHandler",
            "BossRewardHandler",
            "AstrolabeTransformHandler",
        }:
            self._trace_context_event(
                context,
                event_type="graph_unhandled_state",
                summary=f"{context.handler_name} requires runtime-aware execute() path",
            )
            return None

        thread_id = self._resolve_thread_id(context)
        self._trace_context_event(
            context,
            event_type="graph_context_built",
            thread_id=thread_id,
            summary=(
                f"handler={context.handler_name}, screen={context.screen_type}, "
                f"choices={len(context.choice_list)}, commands={len(context.available_commands)}"
            ),
        )

        if not self._config.is_handler_enabled(context.handler_name):
            self._trace_context_event(
                context,
                event_type="graph_handler_disabled",
                thread_id=thread_id,
                summary=f"handler {context.handler_name} disabled in config",
            )
            return None

        graph_input: Dict[str, Any] = {
            "context_payload": self._serialize_context(context),
        }
        started_at = time.perf_counter()
        output: Dict[str, Any] | None = None
        self._trace_context_event(
            context,
            event_type="graph_decide_start",
            thread_id=thread_id,
            summary=f"starting graph decision for {context.handler_name}",
        )

        for attempt in range(self._config.max_retries + 1):
            try:
                self._trace_context_event(
                    context,
                    event_type="graph_attempt_start",
                    thread_id=thread_id,
                    summary=f"attempt={attempt + 1}",
                    metadata={"attempt": attempt + 1},
                )
                output = self._invoke_with_timeout(graph_input, thread_id)
                break
            except FutureTimeoutError as exc:
                self._trace_context_event(
                    context,
                    event_type="graph_attempt_timeout",
                    thread_id=thread_id,
                    summary=f"attempt {attempt + 1} timed out",
                    metadata={"attempt": attempt + 1, "exception": type(exc).__name__, "message": str(exc)},
                )
                if attempt >= self._config.max_retries:
                    return None
            except Exception as exc:
                self._trace_context_event(
                    context,
                    event_type="graph_attempt_exception",
                    thread_id=thread_id,
                    summary=f"attempt {attempt + 1} raised {type(exc).__name__}",
                    metadata={"attempt": attempt + 1, "exception": type(exc).__name__, "message": str(exc)},
                )
                if attempt >= self._config.max_retries:
                    return None

        if output is None or not bool(output.get("decision_valid")):
            self._trace_output_event(
                context,
                output,
                event_type="graph_decide_invalid_output",
                thread_id=thread_id,
                summary="graph output missing or decision invalid",
            )
            return None

        commands = output.get("commands")
        if not isinstance(commands, list) or not commands:
            self._trace_output_event(
                context,
                output,
                event_type="graph_decide_no_commands",
                thread_id=thread_id,
                summary="graph output did not contain executable commands",
            )
            return None

        decision_context = self._deserialize_context(
            cast(dict[str, Any], output.get("decision_context_payload", graph_input["context_payload"]))
        )
        decision = AgentDecision(
            proposed_command=cast(str | None, output.get("proposed_command")),
            confidence=float(output.get("confidence", 0.0)),
            explanation=str(output.get("explanation", "")),
            required_tools_used=[],
            fallback_recommended=False,
            metadata=dict(output.get("metadata", {})),
        )

        if self._config.telemetry_enabled:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            telemetry = build_decision_telemetry(decision_context, decision, latency_ms)
            try:
                write_decision_telemetry(telemetry, self._config.telemetry_path)
            except Exception:
                pass

        self._trace_output_event(
            decision_context,
            output,
            event_type="graph_decide_success",
            thread_id=thread_id,
            summary=f"commands={self._safe_preview(commands, -1)}",
            metadata={"commands": [str(command) for command in commands]},
        )
        return [str(command) for command in commands]

    def execute(
            self,
            state: GameState,
            runtime: BattleRuntimeAdapter | None = None,
    ) -> GraphExecutionResult | None:
        if not self.is_enabled():
            return None

        followup_commands = self._forced_followup_commands_after_card_reward_skip(state)
        if followup_commands is not None:
            if runtime is None:
                return GraphExecutionResult(handled=True, commands=followup_commands, final_state=state)
            final_state = runtime.execute(followup_commands)
            return GraphExecutionResult(handled=True, commands=None, final_state=final_state)

        deterministic_commands = self._deterministic_single_command(state)
        if deterministic_commands is not None:
            if runtime is None:
                return GraphExecutionResult(handled=True, commands=deterministic_commands, final_state=state)
            final_state = runtime.execute(deterministic_commands)
            return GraphExecutionResult(handled=True, commands=None, final_state=final_state)

        context = self._build_context(state)
        if context is None:
            return None
        if not self._config.is_handler_enabled(context.handler_name):
            return None

        if runtime is None:
            commands = self.decide(state)
            if commands is None:
                return None
            return GraphExecutionResult(handled=True, commands=commands, final_state=state)

        session = self._action_executor.execute(state, runtime)
        return GraphExecutionResult(
            handled=session.handled,
            commands=session.commands,
            final_state=session.final_state,
            battle_session=session.battle_session,
            reward_session=session.reward_session,
            campfire_session=None,
        )

    def get_compiled_graph(self) -> Any:
        if self._compiled_graph is None:
            graph = StateGraph(AIPlayerGraphState)
            graph.add_node("ingest_game_state", self._ingest_game_state_node)
            graph.add_node("retrieve_long_term_memory", self._retrieve_long_term_memory_node)
            graph.add_node("route_decision_type", self._route_decision_type_node)
            graph.add_node("decide_event", self._decide_event_node)
            graph.add_node("decide_shop", self._decide_shop_node)
            graph.add_node("decide_card_reward", self._decide_card_reward_node)
            graph.add_node("decide_map", self._decide_map_node)
            graph.add_node("decide_generic", self._decide_generic_node)
            graph.add_node("decide_fallback", self._decide_fallback_node)
            graph.add_node("validate_decision", self._validate_decision_node)
            graph.add_node("commit_short_term_memory", self._commit_short_term_memory_node)
            graph.add_node("emit_commands", self._emit_commands_node)

            graph.add_edge(START, "ingest_game_state")
            graph.add_edge("ingest_game_state", "retrieve_long_term_memory")
            graph.add_edge("retrieve_long_term_memory", "route_decision_type")
            graph.add_conditional_edges(
                "route_decision_type",
                self._route_to_node,
                {
                    "decide_event": "decide_event",
                    "decide_shop": "decide_shop",
                    "decide_card_reward": "decide_card_reward",
                    "decide_map": "decide_map",
                    "decide_generic": "decide_generic",
                    "decide_fallback": "decide_fallback",
                },
            )
            graph.add_edge("decide_event", "validate_decision")
            graph.add_edge("decide_shop", "validate_decision")
            graph.add_edge("decide_card_reward", "validate_decision")
            graph.add_edge("decide_map", "validate_decision")
            graph.add_edge("decide_generic", "validate_decision")
            graph.add_edge("decide_fallback", "validate_decision")
            graph.add_edge("validate_decision", "commit_short_term_memory")
            graph.add_edge("commit_short_term_memory", "emit_commands")
            graph.add_edge("emit_commands", END)
            self._compiled_graph = graph.compile(checkpointer=self._checkpointer)
        return self._compiled_graph

    def _invoke_with_timeout(self, graph_input: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        graph = self.get_compiled_graph()
        config = {"configurable": {"thread_id": thread_id}}
        if self._config.timeout_ms < 0:
            return dict(graph.invoke(graph_input, config=config))

        timeout_seconds = max(0.001, self._config.timeout_ms / 1000.0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(graph.invoke, graph_input, config=config)
            return dict(future.result(timeout=timeout_seconds))

    def _trace_enabled(self) -> bool:
        return bool(self._config.graph_trace_enabled)

    def _safe_preview(self, value: Any, limit: int = -1) -> str:
        """Whitespace-normalized string preview; ``limit <= 0`` (e.g. ``-1``) means no truncation."""
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _run_id_from_context(self, context: AgentContext) -> str:
        run_id = str(context.extras.get("run_id", "")).strip()
        if run_id != "":
            return run_id
        return self._resolve_thread_id(context)

    def _trace_record(self, **kwargs: Any) -> None:
        if not self._trace_enabled():
            return
        record = build_graph_trace_record(**kwargs)
        try:
            write_graph_trace(record, self._config.graph_trace_path)
            mirror_graph_trace_to_run_log(record)
        except Exception:
            pass

    def _trace_raw_event(
            self,
            *,
            event_type: str,
            thread_id: str = "",
            run_id: str = "",
            handler_name: str = "",
            screen_type: str = "",
            node_name: str = "",
            route_name: str = "",
            decision_valid: bool | None = None,
            validation_code: str = "",
            proposed_command: str | None = None,
            confidence: float | None = None,
            summary: str = "",
            metadata: Dict[str, Any] | None = None,
    ) -> None:
        self._trace_record(
            thread_id=thread_id,
            run_id=run_id,
            handler_name=handler_name,
            screen_type=screen_type,
            event_type=event_type,
            node_name=node_name,
            route_name=route_name,
            decision_valid=decision_valid,
            validation_code=validation_code,
            proposed_command=proposed_command,
            confidence=confidence,
            summary=self._safe_preview(summary, -1),
            metadata=dict(metadata or {}),
        )

    def _trace_context_event(
            self,
            context: AgentContext,
            *,
            event_type: str,
            thread_id: str | None = None,
            node_name: str = "",
            route_name: str = "",
            decision_valid: bool | None = None,
            validation_code: str = "",
            proposed_command: str | None = None,
            confidence: float | None = None,
            summary: str = "",
            metadata: Dict[str, Any] | None = None,
    ) -> None:
        resolved_thread_id = self._resolve_thread_id(context) if thread_id is None else thread_id
        self._trace_raw_event(
            thread_id=resolved_thread_id,
            run_id=self._run_id_from_context(context),
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            event_type=event_type,
            node_name=node_name,
            route_name=route_name,
            decision_valid=decision_valid,
            validation_code=validation_code,
            proposed_command=proposed_command,
            confidence=confidence,
            summary=summary,
            metadata=metadata,
        )

    def _trace_state_event(
            self,
            state: AIPlayerGraphState,
            *,
            event_type: str,
            node_name: str,
            summary: str = "",
            metadata: Dict[str, Any] | None = None,
            route_name: str | None = None,
            decision_valid: bool | None = None,
            validation_code: str | None = None,
            proposed_command: str | None = None,
            confidence: float | None = None,
    ) -> None:
        payload = cast(dict[str, Any], state.get("decision_context_payload", state.get("context_payload", {})))
        if not payload:
            self._trace_raw_event(
                thread_id=str(state.get("run_id", "")),
                run_id=str(state.get("run_id", "")),
                handler_name=str(state.get("handler_name", "")),
                screen_type=str(state.get("screen_type", "")),
                event_type=event_type,
                node_name=node_name,
                route_name=str(route_name if route_name is not None else state.get("route_name", "")),
                decision_valid=decision_valid if decision_valid is not None else cast(bool | None, state.get("decision_valid")),
                validation_code=(
                    str(validation_code) if validation_code is not None else str(state.get("validation_code", ""))
                ),
                proposed_command=proposed_command if proposed_command is not None else cast(str | None, state.get("proposed_command")),
                confidence=confidence if confidence is not None else self._coerce_float(state.get("confidence")),
                summary=summary,
                metadata=metadata,
            )
            return

        context = self._deserialize_context(payload)
        self._trace_context_event(
            context,
            event_type=event_type,
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(context),
            node_name=node_name,
            route_name=str(route_name if route_name is not None else state.get("route_name", "")),
            decision_valid=decision_valid if decision_valid is not None else cast(bool | None, state.get("decision_valid")),
            validation_code=(
                str(validation_code) if validation_code is not None else str(state.get("validation_code", ""))
            ),
            proposed_command=proposed_command if proposed_command is not None else cast(str | None, state.get("proposed_command")),
            confidence=confidence if confidence is not None else self._coerce_float(state.get("confidence")),
            summary=summary,
            metadata=metadata,
        )

    def _trace_output_event(
            self,
            context: AgentContext,
            output: Dict[str, Any] | None,
            *,
            event_type: str,
            thread_id: str,
            summary: str,
            metadata: Dict[str, Any] | None = None,
    ) -> None:
        payload = output or {}
        merged_metadata = dict(metadata or {})
        if payload.get("commands") is not None:
            merged_metadata.setdefault("commands", payload.get("commands"))
        self._trace_context_event(
            context,
            event_type=event_type,
            thread_id=thread_id,
            route_name=str(payload.get("route_name", "")),
            decision_valid=cast(bool | None, payload.get("decision_valid")),
            validation_code=str(payload.get("validation_code", "")),
            proposed_command=cast(str | None, payload.get("proposed_command")),
            confidence=self._coerce_float(payload.get("confidence")),
            summary=summary,
            metadata=merged_metadata,
        )

    # def _is_single_choice_non_battle_bypass(self, state: GameState) -> bool:
    #     return (
    #         not is_battle_scope_state(state)
    #         and not self._is_runtime_subagent_scope(state)
    #         and state.has_command(Command.CHOOSE)
    #         and len(state.get_choice_list()) <= 1
    #     )

    def _deterministic_single_command(self, state: GameState) -> list[str] | None:
        """When exactly one protocol verb is available, return its command line without invoking the graph.

        If multiple verbs are listed (e.g. choose and skip), the player still has a real decision; return None.
        For ``choose``, requires exactly one entry in ``choice_list`` (emit ``choose 0``).
        """
        if state.screen_type() == ScreenType.CARD_REWARD.value:
            return None
        if self._is_runtime_subagent_scope(state):
            return None

        raw = state.json.get("available_commands")
        if not isinstance(raw, list):
            return None
        verbs = [str(item).strip() for item in raw if str(item).strip()]
        meaningful_verbs = [verb for verb in verbs if verb not in PROTOCOL_META_COMMANDS]
        if len(meaningful_verbs) != 1:
            return None
        verb = meaningful_verbs[0]
        if verb == Command.CHOOSE.value:
            if len(state.get_choice_list()) != 1:
                return None
            return ["choose 0"]
        return [verb]

    def _forced_followup_commands_after_card_reward_skip(self, state: GameState) -> list[str] | None:
        if not self._matches_last_card_reward_skip(state):
            return None

        available_commands = {
            str(command).strip().lower()
            for command in state.json.get("available_commands", [])
            if str(command).strip() != ""
        }
        if state.screen_type() == ScreenType.CARD_REWARD.value and Command.SKIP.value in available_commands:
            self._last_card_reward_skip_consumed = True
            return [Command.SKIP.value, "wait 30"]

        if (
                state.screen_type() == ScreenType.COMBAT_REWARD.value
                and Command.PROCEED.value in available_commands
                and self._is_combat_reward_card_only_row(state)
        ):
            self._last_card_reward_skip_consumed = True
            return [Command.PROCEED.value]

        return None

    def _matches_last_card_reward_skip(self, state: GameState) -> bool:
        if self._last_card_reward_skip is None or self._last_card_reward_skip_consumed:
            return False
        if state.screen_type() not in {ScreenType.CARD_REWARD.value, ScreenType.COMBAT_REWARD.value}:
            return False
        return self._last_card_reward_skip.get("run_id", "") == self._state_run_id(state)

    def _state_run_id(self, state: GameState) -> str:
        game_state = state.game_state()
        character_class = str(game_state.get("class", "unknown")).strip().lower()
        seed = str(game_state.get("seed", "unknown")).strip().lower()
        return f"{character_class}:{seed}"

    def _is_combat_reward_card_only_row(self, state: GameState) -> bool:
        if state.screen_type() != ScreenType.COMBAT_REWARD.value:
            return False
        rewards = state.screen_state().get("rewards", [])
        if not isinstance(rewards, list) or len(rewards) != 1:
            return False
        reward = rewards[0]
        if not isinstance(reward, dict):
            return False
        return str(reward.get("reward_type", "")).strip().upper() == "CARD"

    def _is_runtime_subagent_scope(self, state: GameState) -> bool:
        return (
            state.screen_type() == ScreenType.REST.value
            or
            state.screen_type() == ScreenType.COMBAT_REWARD.value
            or state.screen_type() == ScreenType.BOSS_REWARD.value
            or state.screen_type() == ScreenType.CHEST.value
            or state.screen_type() == ScreenType.GRID.value
            or state.screen_type() == ScreenType.HAND_SELECT.value
            or is_astrolabe_transform_state(state)
        )

    def _build_context(self, state: GameState) -> AgentContext | None:
        if state.screen_type() == ScreenType.HAND_SELECT.value:
            return build_generic_unhandled_agent_context(state, "HandSelectHandler")

        if state.screen_type() == ScreenType.GRID.value and not is_astrolabe_transform_state(state):
            return build_generic_unhandled_agent_context(state, "GridSelectHandler")

        if state.screen_type() == ScreenType.CHEST.value:
            return build_generic_unhandled_agent_context(state, "ChestHandler")

        if is_battle_scope_state(state):
            return build_battle_agent_context(state, "BattleHandler")

        if state.screen_type() == ScreenType.REST.value and state.has_command(Command.CHOOSE):
            return build_campfire_agent_context(state, "CampfireHandler")

        if state.screen_type() == ScreenType.COMBAT_REWARD.value and state.has_command(Command.CHOOSE):
            return build_combat_reward_agent_context(state, "CombatRewardHandler")

        if state.screen_type() == ScreenType.BOSS_REWARD.value and state.has_command(Command.CHOOSE):
            return build_boss_reward_agent_context(state, "BossRewardHandler")

        if is_astrolabe_transform_state(state) and state.has_command(Command.CHOOSE):
            return build_astrolabe_transform_agent_context(state, "AstrolabeTransformHandler")

        if state.screen_type() == ScreenType.EVENT.value and state.has_command(Command.CHOOSE):
            return build_event_agent_context(state, "EventHandler")

        if state.screen_type() == ScreenType.SHOP_SCREEN.value:
            return build_shop_purchase_agent_context(state, "ShopPurchaseHandler")

        if state.has_command(Command.CHOOSE) and state.screen_type() == ScreenType.CARD_REWARD.value and (
                state.game_state()["room_phase"] == "COMPLETE"
                or state.game_state()["room_phase"] == "EVENT"
                or state.game_state()["room_phase"] == "COMBAT"
        ):
            return build_card_reward_agent_context(state, "CardRewardHandler")

        if state.screen_type() == ScreenType.MAP.value and state.has_command(Command.CHOOSE):
            return build_map_agent_context(state, "MapHandler", self._map_handler_config)

        return build_generic_unhandled_agent_context(state, "GenericHandler")

    def _build_generic_context(self, state: GameState, handler_name: str) -> AgentContext:
        available_commands = state.json.get("available_commands")
        if not isinstance(available_commands, list):
            available_commands = []
        game_state = state.game_state()
        return AgentContext(
            handler_name=handler_name,
            screen_type=state.screen_type(),
            available_commands=[str(command) for command in available_commands],
            choice_list=state.get_choice_list().copy(),
            game_state={
                "floor": state.floor(),
                "act": game_state.get("act"),
                "room_phase": game_state.get("room_phase"),
                "room_type": game_state.get("room_type"),
                "character_class": game_state.get("class"),
            },
            extras={},
        )

    def _serialize_context(self, context: AgentContext) -> Dict[str, Any]:
        return asdict(context)

    def _deserialize_context(self, payload: Dict[str, Any]) -> AgentContext:
        return AgentContext(
            handler_name=str(payload.get("handler_name", "")),
            screen_type=str(payload.get("screen_type", "")),
            available_commands=[str(command) for command in payload.get("available_commands", [])],
            choice_list=[str(choice) for choice in payload.get("choice_list", [])],
            game_state=dict(payload.get("game_state", {})),
            extras=dict(payload.get("extras", {})),
        )

    def _resolve_thread_id(self, context: AgentContext) -> str:
        run_id = str(context.extras.get("run_id", "")).strip()
        if run_id != "":
            return run_id
        agent_identity = str(context.extras.get("agent_identity", "neo_primates")).strip().lower() or "neo_primates"
        character_class = str(context.game_state.get("character_class", "unknown")).strip().lower() or "unknown"
        return f"{agent_identity}:{character_class}"

    def _ingest_game_state_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="ingest_game_state",
            summary="building short-term run state",
        )
        context = self._deserialize_context(state["context_payload"])
        recent_key_decisions = self._trim_recent_decisions(state.get("recent_key_decisions", []))
        distilled_run_summary = self._build_distilled_run_summary(context, recent_key_decisions)
        result = {
            "handler_name": context.handler_name,
            "run_id": self._resolve_thread_id(context),
            "screen_type": context.screen_type,
            "floor": self._coerce_int(context.game_state.get("floor")),
            "act": self._coerce_int(context.game_state.get("act")),
            "recent_key_decisions": recent_key_decisions,
            "distilled_run_summary": distilled_run_summary,
            "decision_context_payload": state.get("context_payload", {}),
            "retrieved_episodic_memories": "none",
            "retrieved_semantic_memories": "none",
            "langmem_status": self._langmem_service.status(),
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "",
            "metadata": {},
            "decision_valid": False,
            "validation_code": "not_decided",
            "validation_message": "not_decided",
            "commands": None,
        }
        self._trace_context_event(
            context,
            event_type="node_exit",
            thread_id=result["run_id"],
            node_name="ingest_game_state",
            summary=f"act={result['act']}, floor={result['floor']}, recent_count={len(recent_key_decisions)}",
            metadata={"recent_key_decisions": list(recent_key_decisions)},
        )
        return result

    def _retrieve_long_term_memory_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="retrieve_long_term_memory",
            summary="loading langmem context",
        )
        decision_context = self._build_decision_context_from_state(state)
        payload = self._langmem_service.build_context_memory(decision_context)
        result = {
            "decision_context_payload": self._serialize_context(decision_context),
            "retrieved_episodic_memories": payload.get("retrieved_episodic_memories", "none"),
            "retrieved_semantic_memories": payload.get("retrieved_semantic_memories", "none"),
            "langmem_status": payload.get("langmem_status", self._langmem_service.status()),
        }
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="retrieve_long_term_memory",
            route_name=str(state.get("route_name", "")),
            summary=(
                f"langmem_status={result['langmem_status']}, episodic_len="
                f"{len(str(result['retrieved_episodic_memories']))}, semantic_len="
                f"{len(str(result['retrieved_semantic_memories']))}"
            ),
            metadata={
                "langmem_status": result["langmem_status"],
                "episodic_preview": self._safe_preview(result["retrieved_episodic_memories"], -1),
                "semantic_preview": self._safe_preview(result["retrieved_semantic_memories"], -1),
            },
        )
        return result

    def _route_decision_type_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="route_decision_type",
            summary="routing graph by handler name",
        )
        route_map: dict[str, GraphRoute] = {
            "EventHandler": "decide_event",
            "ShopPurchaseHandler": "decide_shop",
            "CardRewardHandler": "decide_card_reward",
            "MapHandler": "decide_map",
            "ChestHandler": "decide_generic",
            "GridSelectHandler": "decide_generic",
            "HandSelectHandler": "decide_generic",
            "GenericHandler": "decide_generic",
        }
        result = {"route_name": route_map.get(str(state.get("handler_name", "")), "decide_fallback")}
        self._trace_state_event(
            state,
            event_type="node_exit",
            node_name="route_decision_type",
            route_name=result["route_name"],
            summary=f"resolved route {result['route_name']}",
        )
        return result

    def _route_to_node(self, state: AIPlayerGraphState) -> GraphRoute:
        return cast(GraphRoute, state.get("route_name", "decide_fallback"))

    def _decide_event_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(state, event_type="node_enter", node_name="decide_event", summary="requesting event proposal")
        decision_context = self._build_decision_context_from_state(state)
        proposal, proposal_validated, validation_feedback = self._propose_with_validation_retry(decision_context, self._event_provider)
        result = self._proposal_result_after_retry(
            decision_context,
            proposal,
            proposal_validated,
            validation_feedback,
            failure_code="event_validation_failed_after_retries",
        )
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="decide_event",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposal.proposed_command if proposal is not None else None,
            confidence=proposal.confidence if proposal is not None else 0.0,
            summary=(
                f"proposal={(proposal.proposed_command if proposal is not None else None)}, "
                f"confidence={(proposal.confidence if proposal is not None else 0.0):.2f}, "
                f"explanation={self._safe_preview((proposal.explanation if proposal is not None else ''), -1)}"
            ),
            metadata={"provider_metadata": dict(proposal.metadata) if proposal is not None else {}},
        )
        return result

    def _decide_shop_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(state, event_type="node_enter", node_name="decide_shop", summary="requesting shop proposal")
        decision_context = self._build_decision_context_from_state(state)
        proposal, proposal_validated, validation_feedback = self._propose_with_validation_retry(decision_context, self._shop_provider)
        result = self._proposal_result_after_retry(
            decision_context,
            proposal,
            proposal_validated,
            validation_feedback,
            failure_code="shop_validation_failed_after_retries",
        )
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="decide_shop",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposal.proposed_command if proposal is not None else None,
            confidence=proposal.confidence if proposal is not None else 0.0,
            summary=(
                f"proposal={(proposal.proposed_command if proposal is not None else None)}, "
                f"confidence={(proposal.confidence if proposal is not None else 0.0):.2f}, "
                f"explanation={self._safe_preview((proposal.explanation if proposal is not None else ''), -1)}"
            ),
            metadata={"provider_metadata": dict(proposal.metadata) if proposal is not None else {}},
        )
        return result

    def _decide_card_reward_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="decide_card_reward",
            summary="requesting card reward proposal",
        )
        decision_context = self._build_decision_context_from_state(state)
        proposal, proposal_validated, validation_feedback = self._propose_with_validation_retry(decision_context, self._card_reward_provider)
        result = self._proposal_result_after_retry(
            decision_context,
            proposal,
            proposal_validated,
            validation_feedback,
            failure_code="card_reward_validation_failed_after_retries",
        )
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="decide_card_reward",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposal.proposed_command if proposal is not None else None,
            confidence=proposal.confidence if proposal is not None else 0.0,
            summary=(
                f"proposal={(proposal.proposed_command if proposal is not None else None)}, "
                f"confidence={(proposal.confidence if proposal is not None else 0.0):.2f}, "
                f"explanation={self._safe_preview((proposal.explanation if proposal is not None else ''), -1)}"
            ),
            metadata={"provider_metadata": dict(proposal.metadata) if proposal is not None else {}},
        )
        return result

    def _decide_map_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(state, event_type="node_enter", node_name="decide_map", summary="requesting map proposal")
        decision_context = self._build_decision_context_from_state(state)
        proposal, proposal_validated, validation_feedback = self._propose_with_validation_retry(decision_context, self._map_provider)
        result = self._proposal_result_after_retry(
            decision_context,
            proposal,
            proposal_validated,
            validation_feedback,
            failure_code="map_validation_failed_after_retries",
        )
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="decide_map",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposal.proposed_command if proposal is not None else None,
            confidence=proposal.confidence if proposal is not None else 0.0,
            summary=(
                f"proposal={(proposal.proposed_command if proposal is not None else None)}, "
                f"confidence={(proposal.confidence if proposal is not None else 0.0):.2f}, "
                f"explanation={self._safe_preview((proposal.explanation if proposal is not None else ''), -1)}"
            ),
            metadata={"provider_metadata": dict(proposal.metadata) if proposal is not None else {}},
        )
        return result

    def _decide_generic_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(state, event_type="node_enter", node_name="decide_generic", summary="requesting generic proposal")
        decision_context = self._build_decision_context_from_state(state)
        proposal, proposal_validated, validation_feedback = self._propose_with_validation_retry(decision_context, self._generic_provider)
        result = self._proposal_result_after_retry(
            decision_context,
            proposal,
            proposal_validated,
            validation_feedback,
            failure_code="generic_validation_failed_after_retries",
        )

        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="decide_generic",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposal.proposed_command if proposal is not None else None,
            confidence=proposal.confidence if proposal is not None else 0.0,
            summary=(
                f"proposal={(proposal.proposed_command if proposal is not None else None)}, "
                f"confidence={(proposal.confidence if proposal is not None else 0.0):.2f}, "
                f"explanation={self._safe_preview((proposal.explanation if proposal is not None else ''), -1)}"
            ),
            metadata={"provider_metadata": dict(proposal.metadata) if proposal is not None else {}},
        )
        return result

    def _propose_with_optional_feedback(
            self,
            provider: Any,
            decision_context: AgentContext,
            validation_feedback: Dict[str, Any] | None,
    ) -> Any:
        try:
            return provider.propose(
                decision_context,
                validation_feedback=validation_feedback,
            )
        except TypeError:
            return provider.propose(decision_context)

    def _propose_with_validation_retry(
            self,
            decision_context: AgentContext,
            provider: Any,
            *,
            max_attempts: int = 3,
    ) -> tuple[Any, bool, Dict[str, Any] | None]:
        validation_feedback: Dict[str, Any] | None = None
        proposal = None
        proposal_validated = False

        for _ in range(max_attempts):
            proposal = self._propose_with_optional_feedback(provider, decision_context, validation_feedback)
            if proposal is None:
                break
            proposed_command = proposal.proposed_command
            if proposed_command is None:
                validation_feedback = {
                    "code": "empty_command",
                    "message": (
                        "proposed_command cannot be null. Return exactly one concrete, legal command "
                        "for the current state."
                    ),
                    "allowed_commands": [str(command) for command in decision_context.available_commands],
                    "choice_list": [str(choice) for choice in decision_context.choice_list],
                }
                continue

            validation, feedback = validate_proposed_command(decision_context, proposed_command)
            if validation.is_valid:
                proposal_validated = True
                break
            validation_feedback = feedback

        return proposal, proposal_validated, validation_feedback

    def _proposal_result_after_retry(
            self,
            decision_context: AgentContext,
            proposal: Any,
            proposal_validated: bool,
            validation_feedback: Dict[str, Any] | None,
            *,
            failure_code: str,
    ) -> Dict[str, Any]:
        if proposal is None:
            return self._proposal_update(
                decision_context,
                None,
                0.0,
                "provider_unavailable",
                {"validation_error": "provider_unavailable"},
            )

        if not proposal_validated and validation_feedback is not None:
            metadata = dict(getattr(proposal, "metadata", {}))
            metadata["validation_feedback"] = dict(validation_feedback)
            metadata["validation_error"] = failure_code
            return self._proposal_update(
                decision_context,
                None,
                0.0,
                failure_code,
                metadata,
            )

        return self._proposal_update(
            decision_context,
            proposal.proposed_command,
            proposal.confidence,
            proposal.explanation,
            proposal.metadata,
        )

    def _decide_fallback_node(self, _state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            _state,
            event_type="node_enter",
            node_name="decide_fallback",
            summary="falling back because no supported route matched",
        )
        result = {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "unsupported_decision_type",
            "metadata": {"validation_error": "unsupported_decision_type"},
        }
        self._trace_state_event(
            _state,
            event_type="node_exit",
            node_name="decide_fallback",
            summary="fallback produced no command",
            metadata=result["metadata"],
        )
        return result

    def _validate_decision_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="validate_decision",
            summary="validating proposed command",
        )
        decision_context = self._build_decision_context_from_state(state)
        metadata = dict(state.get("metadata", {}))
        proposed_command = state.get("proposed_command")
        confidence = float(state.get("confidence", 0.0))

        if proposed_command is None:
            metadata["validation_error"] = "empty_command"
            result = {
                "decision_valid": False,
                "validation_code": "empty_command",
                "validation_message": "command is empty",
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }
            self._trace_context_event(
                decision_context,
                event_type="node_exit",
                thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
                node_name="validate_decision",
                route_name=str(state.get("route_name", "")),
                decision_valid=False,
                validation_code="empty_command",
                proposed_command=None,
                confidence=confidence,
                summary="validation failed: command is empty",
                metadata=metadata,
            )
            return result

        if confidence < self._config.confidence_threshold:
            metadata["validation_error"] = "low_confidence"
            result = {
                "decision_valid": False,
                "validation_code": "low_confidence",
                "validation_message": "confidence below threshold",
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }
            self._trace_context_event(
                decision_context,
                event_type="node_exit",
                thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
                node_name="validate_decision",
                route_name=str(state.get("route_name", "")),
                decision_valid=False,
                validation_code="low_confidence",
                proposed_command=proposed_command,
                confidence=confidence,
                summary="validation failed: confidence below threshold",
                metadata=metadata,
            )
            return result

        validation = validate_command(decision_context, proposed_command)
        metadata["validation_error"] = validation.code
        if not validation.is_valid:
            metadata["validation_message"] = validation.message
            result = {
                "decision_valid": False,
                "validation_code": validation.code,
                "validation_message": validation.message,
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }
            self._trace_context_event(
                decision_context,
                event_type="node_exit",
                thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
                node_name="validate_decision",
                route_name=str(state.get("route_name", "")),
                decision_valid=False,
                validation_code=validation.code,
                proposed_command=proposed_command,
                confidence=confidence,
                summary=f"validation failed: {self._safe_preview(validation.message, -1)}",
                metadata=metadata,
            )
            return result

        resolution = self._action_policy.resolve(decision_context, proposed_command)
        commands = resolution.commands
        if commands is None:
            metadata["validation_error"] = resolution.validation.code
            result = {
                "decision_valid": False,
                "validation_code": resolution.validation.code,
                "validation_message": resolution.validation.message,
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }
            self._trace_context_event(
                decision_context,
                event_type="node_exit",
                thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
                node_name="validate_decision",
                route_name=str(state.get("route_name", "")),
                decision_valid=False,
                validation_code=resolution.validation.code,
                proposed_command=proposed_command,
                confidence=confidence,
                summary=f"validation failed: {self._safe_preview(resolution.validation.message, -1)}",
                metadata=metadata,
            )
            return result

        result = {
            "decision_valid": True,
            "validation_code": validation.code,
            "validation_message": validation.message,
            "commands": commands,
            "metadata": metadata,
            "decision_context_payload": self._serialize_context(decision_context),
        }
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="validate_decision",
            route_name=str(state.get("route_name", "")),
            decision_valid=True,
            validation_code=validation.code,
            proposed_command=proposed_command,
            confidence=confidence,
            summary=f"validation ok, commands={self._safe_preview(commands, -1)}",
            metadata={"commands": list(commands), **metadata},
        )
        return result

    def _commit_short_term_memory_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="commit_short_term_memory",
            summary="updating short-term memory",
        )
        if not bool(state.get("decision_valid")):
            self._trace_state_event(
                state,
                event_type="node_exit",
                node_name="commit_short_term_memory",
                summary="skipped because decision is invalid",
            )
            return {}

        decision_context = self._deserialize_context(cast(dict[str, Any], state.get("decision_context_payload", state["context_payload"])))
        proposed_command = state.get("proposed_command")
        if proposed_command is None:
            self._trace_context_event(
                decision_context,
                event_type="node_exit",
                thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
                node_name="commit_short_term_memory",
                route_name=str(state.get("route_name", "")),
                summary="skipped because proposed command is empty",
            )
            return {}

        decision = AgentDecision(
            proposed_command=proposed_command,
            confidence=float(state.get("confidence", 0.0)),
            explanation=str(state.get("explanation", "")),
            required_tools_used=[],
            fallback_recommended=False,
            metadata=dict(state.get("metadata", {})),
        )
        self._langmem_service.record_accepted_decision(decision_context, decision)
        if decision_context.handler_name == "CardRewardHandler":
            if proposed_command == Command.SKIP.value:
                self._last_card_reward_skip = {
                    "run_id": self._run_id_from_context(decision_context),
                    "floor": self._coerce_int(decision_context.game_state.get("floor")),
                }
                self._last_card_reward_skip_consumed = False
            else:
                self._last_card_reward_skip = None
                self._last_card_reward_skip_consumed = False

        recent_key_decisions = self._trim_recent_decisions(state.get("recent_key_decisions", []))
        recent_key_decisions.append({
            "handler_name": decision_context.handler_name,
            "floor": self._coerce_int(decision_context.game_state.get("floor")),
            "act": self._coerce_int(decision_context.game_state.get("act")),
            "proposed_command": proposed_command,
            "confidence": round(float(state.get("confidence", 0.0)), 2),
            "explanation": str(state.get("explanation", "")),
        })
        recent_key_decisions = self._trim_recent_decisions(recent_key_decisions)

        distilled_run_summary = self._build_distilled_run_summary(decision_context, recent_key_decisions)
        result = {
            "recent_key_decisions": recent_key_decisions,
            "distilled_run_summary": distilled_run_summary,
        }
        self._trace_context_event(
            decision_context,
            event_type="node_exit",
            thread_id=str(state.get("run_id", "")) or self._resolve_thread_id(decision_context),
            node_name="commit_short_term_memory",
            route_name=str(state.get("route_name", "")),
            proposed_command=proposed_command,
            confidence=float(state.get("confidence", 0.0)),
            summary=f"recorded decision, recent_count={len(recent_key_decisions)}",
            metadata={
                "recent_count": len(recent_key_decisions),
                "recent_preview": self._safe_preview(self._format_recent_decisions(recent_key_decisions), -1),
            },
        )
        return result

    def _emit_commands_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        self._trace_state_event(
            state,
            event_type="node_enter",
            node_name="emit_commands",
            summary="emitting final commands",
        )
        result = {"commands": state.get("commands")}
        self._trace_state_event(
            state,
            event_type="node_exit",
            node_name="emit_commands",
            summary=f"commands={self._safe_preview(result['commands'], -1)}",
            metadata={"commands": result["commands"]},
        )
        return result

    def _proposal_update(
            self,
            decision_context: AgentContext,
            proposed_command: str | None,
            confidence: float,
            explanation: str,
            metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "decision_context_payload": self._serialize_context(decision_context),
            "proposed_command": proposed_command,
            "confidence": confidence,
            "explanation": explanation,
            "metadata": dict(metadata),
        }

    def _build_decision_context_from_state(self, state: AIPlayerGraphState) -> AgentContext:
        context = self._deserialize_context(state["context_payload"])
        extras = dict(context.extras)
        extras["run_memory_summary"] = str(state.get("distilled_run_summary", extras.get("run_memory_summary", "")))
        extras["recent_llm_decisions"] = self._format_recent_decisions(state.get("recent_key_decisions", []))
        extras["retrieved_episodic_memories"] = state.get(
            "retrieved_episodic_memories",
            extras.get("retrieved_episodic_memories", "none"),
        )
        extras["retrieved_semantic_memories"] = state.get(
            "retrieved_semantic_memories",
            extras.get("retrieved_semantic_memories", "none"),
        )
        extras["langmem_status"] = state.get("langmem_status", extras.get("langmem_status", "disabled_by_config"))
        return AgentContext(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=list(context.available_commands),
            choice_list=list(context.choice_list),
            game_state=dict(context.game_state),
            extras=extras,
        )

    def _build_distilled_run_summary(
            self,
            context: AgentContext,
            recent_key_decisions: list[dict[str, Any]],
    ) -> str:
        base_summary = str(context.extras.get("run_memory_summary", "")).strip()
        recent_summary = self._format_recent_decisions(recent_key_decisions)
        if recent_summary == "none":
            return base_summary
        return f"{base_summary} | recent={recent_summary}".strip()

    def _format_recent_decisions(self, recent_key_decisions: list[dict[str, Any]]) -> str:
        trimmed = self._trim_recent_decisions(recent_key_decisions)
        if not trimmed:
            return "none"

        parts: list[str] = []
        for entry in trimmed:
            location = []
            act = self._coerce_int(entry.get("act"))
            floor = self._coerce_int(entry.get("floor"))
            if act is not None:
                location.append(f"A{act}")
            if floor is not None:
                location.append(f"F{floor}")
            location_prefix = f"{' '.join(location)} " if location else ""
            parts.append(
                f"{location_prefix}{entry.get('handler_name', 'unknown')} -> "
                f"{entry.get('proposed_command', 'none')} "
                f"({float(entry.get('confidence', 0.0)):.2f}, {entry.get('explanation', '')})"
            )
        return " | ".join(parts)

    def _trim_recent_decisions(self, recent_key_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = [dict(entry) for entry in recent_key_decisions if isinstance(entry, dict)]
        return trimmed[-5:]

    def run(self, state: GameState, runtime: BattleRuntimeAdapter) -> ActionExecutionResult:
        context = self._build_context(state)
        if context is None:
            return ActionExecutionResult(handled=False, final_state=runtime.current_state())

        if context.handler_name == "BattleHandler":
            session_result = self._battle_subagent.run(state, runtime)
            return ActionExecutionResult(
                handled=session_result.handled,
                final_state=session_result.final_state,
                battle_session=session_result,
                steps=session_result.steps,
                summary=session_result.summary,
            )

        if context.handler_name == "CampfireHandler":
            campfire_session = self._campfire_subagent.run(state, runtime)
            return ActionExecutionResult(
                handled=campfire_session.handled,
                final_state=campfire_session.final_state,
                steps=campfire_session.steps,
                summary=campfire_session.summary,
            )

        if context.handler_name == "CombatRewardHandler":
            reward_session = self._combat_reward_subagent.run(state, runtime)
            return ActionExecutionResult(
                handled=reward_session.handled,
                final_state=reward_session.final_state,
                reward_session=reward_session,
                steps=reward_session.steps,
                summary=reward_session.summary,
            )

        if context.handler_name == "BossRewardHandler":
            reward_session = self._boss_reward_subagent.run(state, runtime)
            return ActionExecutionResult(
                handled=reward_session.handled,
                final_state=reward_session.final_state,
                reward_session=reward_session,
                steps=reward_session.steps,
                summary=reward_session.summary,
            )

        if context.handler_name == "AstrolabeTransformHandler":
            reward_session = self._astrolabe_transform_subagent.run(state, runtime)
            return ActionExecutionResult(
                handled=reward_session.handled,
                final_state=reward_session.final_state,
                reward_session=reward_session,
                steps=reward_session.steps,
                summary=reward_session.summary,
            )

        commands = self.decide(runtime.current_state())
        if commands is None:
            return ActionExecutionResult(handled=False, final_state=runtime.current_state())
        final_state = runtime.execute(commands)
        return ActionExecutionResult(
            handled=True,
            final_state=final_state,
            commands=None,
            steps=1,
            summary="graph_decide_then_execute",
        )

    def _coerce_int(self, value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None
