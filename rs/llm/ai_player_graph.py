from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
import time
from typing import Any, Dict, Literal, TypedDict, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from rs.game.path import PathHandlerConfig
from rs.game.screen_type import ScreenType
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig, load_llm_config
from rs.llm.langmem_service import LangMemService, get_langmem_service
from rs.llm.providers.card_reward_llm_provider import CardRewardLlmProvider
from rs.llm.providers.event_llm_provider import EventLlmProvider
from rs.llm.providers.map_llm_provider import MapLlmProvider
from rs.llm.providers.shop_purchase_llm_provider import ShopPurchaseLlmProvider
from rs.llm.telemetry import build_decision_telemetry, write_decision_telemetry
from rs.llm.validator import validate_command
from rs.llm.integration.card_reward_context import build_card_reward_agent_context
from rs.llm.integration.event_context import build_event_agent_context
from rs.llm.integration.map_context import build_map_agent_context
from rs.llm.integration.shop_purchase_context import build_shop_purchase_agent_context
from rs.machine.command import Command
from rs.machine.state import GameState


GraphRoute = Literal[
    "decide_event",
    "decide_shop",
    "decide_card_reward",
    "decide_map",
    "decide_fallback",
]


class AIPlayerGraphState(TypedDict, total=False):
    context_payload: Dict[str, Any]
    decision_context_payload: Dict[str, Any]
    handler_name: str
    route_name: GraphRoute
    run_id: str
    strategy_name: str
    screen_type: str
    floor: int | None
    act: int | None
    distilled_run_summary: str
    recent_key_decisions: list[dict[str, Any]]
    current_priorities: list[str]
    risk_flags: list[str]
    deck_direction: str
    run_hypotheses: list[str]
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

    def is_enabled(self) -> bool:
        return self._config.enabled and self._config.ai_player_graph_enabled

    def can_handle(self, state: GameState) -> bool:
        return self._build_context(state) is not None

    def decide(self, state: GameState) -> list[str] | None:
        if not self.is_enabled():
            return None

        context = self._build_context(state)
        if context is None or not self._config.is_handler_enabled(context.handler_name):
            return None

        graph_input: Dict[str, Any] = {
            "context_payload": self._serialize_context(context),
        }
        thread_id = self._resolve_thread_id(context)
        started_at = time.perf_counter()
        output: Dict[str, Any] | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                output = self._invoke_with_timeout(graph_input, thread_id)
                break
            except FutureTimeoutError:
                if attempt >= self._config.max_retries:
                    return None
            except Exception:
                if attempt >= self._config.max_retries:
                    return None

        if output is None or not bool(output.get("decision_valid")):
            return None

        commands = output.get("commands")
        if not isinstance(commands, list) or not commands:
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

        return [str(command) for command in commands]

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
                    "decide_fallback": "decide_fallback",
                },
            )
            graph.add_edge("decide_event", "validate_decision")
            graph.add_edge("decide_shop", "validate_decision")
            graph.add_edge("decide_card_reward", "validate_decision")
            graph.add_edge("decide_map", "validate_decision")
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

    def _build_context(self, state: GameState) -> AgentContext | None:
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

        return None

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
        strategy_name = str(context.extras.get("strategy_name", "unknown")).strip().lower() or "unknown"
        character_class = str(context.game_state.get("character_class", "unknown")).strip().lower() or "unknown"
        return f"{strategy_name}:{character_class}"

    def _ingest_game_state_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        context = self._deserialize_context(state["context_payload"])
        recent_key_decisions = self._trim_recent_decisions(state.get("recent_key_decisions", []))
        current_priorities = self._build_current_priorities(context)
        risk_flags = self._build_risk_flags(context)
        deck_direction = self._build_deck_direction(context)
        run_hypotheses = self._build_run_hypotheses(context, current_priorities, risk_flags, deck_direction)
        distilled_run_summary = self._build_distilled_run_summary(
            context,
            recent_key_decisions,
            current_priorities,
            risk_flags,
            deck_direction,
        )
        return {
            "handler_name": context.handler_name,
            "run_id": self._resolve_thread_id(context),
            "strategy_name": str(context.extras.get("strategy_name", "unknown")),
            "screen_type": context.screen_type,
            "floor": self._coerce_int(context.game_state.get("floor")),
            "act": self._coerce_int(context.game_state.get("act")),
            "recent_key_decisions": recent_key_decisions,
            "current_priorities": current_priorities,
            "risk_flags": risk_flags,
            "deck_direction": deck_direction,
            "run_hypotheses": run_hypotheses,
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

    def _retrieve_long_term_memory_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        payload = self._langmem_service.build_context_memory(decision_context)
        return {
            "decision_context_payload": self._serialize_context(decision_context),
            "retrieved_episodic_memories": payload.get("retrieved_episodic_memories", "none"),
            "retrieved_semantic_memories": payload.get("retrieved_semantic_memories", "none"),
            "langmem_status": payload.get("langmem_status", self._langmem_service.status()),
        }

    def _route_decision_type_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        route_map: dict[str, GraphRoute] = {
            "EventHandler": "decide_event",
            "ShopPurchaseHandler": "decide_shop",
            "CardRewardHandler": "decide_card_reward",
            "MapHandler": "decide_map",
        }
        return {"route_name": route_map.get(str(state.get("handler_name", "")), "decide_fallback")}

    def _route_to_node(self, state: AIPlayerGraphState) -> GraphRoute:
        return cast(GraphRoute, state.get("route_name", "decide_fallback"))

    def _decide_event_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        proposal = self._event_provider.propose(decision_context)
        return self._proposal_update(decision_context, proposal.proposed_command, proposal.confidence, proposal.explanation, proposal.metadata)

    def _decide_shop_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        proposal = self._shop_provider.propose(decision_context)
        return self._proposal_update(decision_context, proposal.proposed_command, proposal.confidence, proposal.explanation, proposal.metadata)

    def _decide_card_reward_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        proposal = self._card_reward_provider.propose(decision_context)
        return self._proposal_update(decision_context, proposal.proposed_command, proposal.confidence, proposal.explanation, proposal.metadata)

    def _decide_map_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        proposal = self._map_provider.propose(decision_context)
        return self._proposal_update(decision_context, proposal.proposed_command, proposal.confidence, proposal.explanation, proposal.metadata)

    def _decide_fallback_node(self, _state: AIPlayerGraphState) -> Dict[str, Any]:
        return {
            "proposed_command": None,
            "confidence": 0.0,
            "explanation": "unsupported_decision_type",
            "metadata": {"validation_error": "unsupported_decision_type"},
        }

    def _validate_decision_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        decision_context = self._build_decision_context_from_state(state)
        metadata = dict(state.get("metadata", {}))
        proposed_command = state.get("proposed_command")
        confidence = float(state.get("confidence", 0.0))

        if proposed_command is None:
            metadata["validation_error"] = "empty_command"
            return {
                "decision_valid": False,
                "validation_code": "empty_command",
                "validation_message": "command is empty",
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }

        if confidence < self._config.confidence_threshold:
            metadata["validation_error"] = "low_confidence"
            return {
                "decision_valid": False,
                "validation_code": "low_confidence",
                "validation_message": "confidence below threshold",
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }

        validation = validate_command(decision_context, proposed_command)
        metadata["validation_error"] = validation.code
        if not validation.is_valid:
            metadata["validation_message"] = validation.message
            return {
                "decision_valid": False,
                "validation_code": validation.code,
                "validation_message": validation.message,
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }

        commands = self._commands_for_context(decision_context, proposed_command)
        if commands is None:
            metadata["validation_error"] = "unsupported_command_mapping"
            return {
                "decision_valid": False,
                "validation_code": "unsupported_command_mapping",
                "validation_message": "no execution mapping for validated command",
                "commands": None,
                "metadata": metadata,
                "decision_context_payload": self._serialize_context(decision_context),
            }

        return {
            "decision_valid": True,
            "validation_code": validation.code,
            "validation_message": validation.message,
            "commands": commands,
            "metadata": metadata,
            "decision_context_payload": self._serialize_context(decision_context),
        }

    def _commit_short_term_memory_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        if not bool(state.get("decision_valid")):
            return {}

        decision_context = self._deserialize_context(cast(dict[str, Any], state.get("decision_context_payload", state["context_payload"])))
        proposed_command = state.get("proposed_command")
        if proposed_command is None:
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

        distilled_run_summary = self._build_distilled_run_summary(
            decision_context,
            recent_key_decisions,
            cast(list[str], state.get("current_priorities", [])),
            cast(list[str], state.get("risk_flags", [])),
            str(state.get("deck_direction", "unknown")),
        )
        return {
            "recent_key_decisions": recent_key_decisions,
            "distilled_run_summary": distilled_run_summary,
        }

    def _emit_commands_node(self, state: AIPlayerGraphState) -> Dict[str, Any]:
        return {"commands": state.get("commands")}

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
        extras["current_priorities"] = list(state.get("current_priorities", []))
        extras["risk_flags"] = list(state.get("risk_flags", []))
        extras["deck_direction"] = str(state.get("deck_direction", "unknown"))
        extras["run_hypotheses"] = list(state.get("run_hypotheses", []))
        return AgentContext(
            handler_name=context.handler_name,
            screen_type=context.screen_type,
            available_commands=list(context.available_commands),
            choice_list=list(context.choice_list),
            game_state=dict(context.game_state),
            extras=extras,
        )

    def _build_current_priorities(self, context: AgentContext) -> list[str]:
        priorities: list[str] = []
        hp = self._coerce_int(context.game_state.get("current_hp"))
        max_hp = self._coerce_int(context.game_state.get("max_hp"))
        gold = self._coerce_int(context.game_state.get("gold"))
        if hp is not None and max_hp is not None and max_hp > 0 and hp / max_hp <= 0.4:
            priorities.append("survive")
        if context.handler_name == "ShopPurchaseHandler" and gold is not None and gold >= 150:
            priorities.append("spend_gold_well")
        if context.handler_name == "CardRewardHandler":
            priorities.append("tighten_deck_direction")
        if context.handler_name == "MapHandler":
            priorities.append("plan_safe_route")
        if context.handler_name == "EventHandler":
            priorities.append("avoid_bad_event_traps")
        if not priorities:
            priorities.append("stay_consistent")
        return priorities

    def _build_risk_flags(self, context: AgentContext) -> list[str]:
        flags: list[str] = []
        hp = self._coerce_int(context.game_state.get("current_hp"))
        max_hp = self._coerce_int(context.game_state.get("max_hp"))
        gold = self._coerce_int(context.game_state.get("gold"))
        if hp is not None and max_hp is not None and max_hp > 0:
            hp_ratio = hp / max_hp
            if hp_ratio <= 0.35:
                flags.append("low_hp")
            elif hp_ratio <= 0.55:
                flags.append("moderate_hp")
        if context.handler_name == "ShopPurchaseHandler" and gold is not None and gold < 75:
            flags.append("low_gold")
        if context.handler_name == "CardRewardHandler" and bool(context.extras.get("potions_full", False)):
            flags.append("potions_full")
        return flags

    def _build_deck_direction(self, context: AgentContext) -> str:
        deck_profile = context.extras.get("deck_profile")
        if not isinstance(deck_profile, dict) or not deck_profile:
            return "unknown"

        weighted_features: list[tuple[str, float]] = []
        for key, value in deck_profile.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            weighted_features.append((str(key), score))

        if not weighted_features:
            return "unknown"
        weighted_features.sort(key=lambda item: item[1], reverse=True)
        return weighted_features[0][0]

    def _build_run_hypotheses(
            self,
            context: AgentContext,
            current_priorities: list[str],
            risk_flags: list[str],
            deck_direction: str,
    ) -> list[str]:
        hypotheses: list[str] = []
        if deck_direction != "unknown":
            hypotheses.append(f"deck leans {deck_direction}")
        if "survive" in current_priorities:
            hypotheses.append("run needs immediate stabilization")
        if "plan_safe_route" in current_priorities:
            hypotheses.append("pathing should avoid risky elites unless payoff is clear")
        if "low_hp" in risk_flags:
            hypotheses.append("preserve hp over greedy upside")
        if not hypotheses:
            hypotheses.append(f"maintain {context.handler_name} consistency")
        return hypotheses[:4]

    def _build_distilled_run_summary(
            self,
            context: AgentContext,
            recent_key_decisions: list[dict[str, Any]],
            current_priorities: list[str],
            risk_flags: list[str],
            deck_direction: str,
    ) -> str:
        base_summary = str(context.extras.get("run_memory_summary", "")).strip()
        priority_text = ", ".join(current_priorities) if current_priorities else "none"
        risk_text = ", ".join(risk_flags) if risk_flags else "stable"
        recent_summary = self._format_recent_decisions(recent_key_decisions)
        return (
            f"{base_summary} | deck_direction={deck_direction} | priorities={priority_text} "
            f"| risks={risk_text} | recent={recent_summary}"
        ).strip()

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

    def _commands_for_context(self, context: AgentContext, proposed_command: str) -> list[str] | None:
        if context.handler_name == "EventHandler":
            return [proposed_command, "wait 30"]

        if context.handler_name == "ShopPurchaseHandler":
            if proposed_command == "return":
                return ["return", "proceed"]
            return [proposed_command, "wait 30"]

        if context.handler_name == "CardRewardHandler":
            return [proposed_command, "wait 30"]

        if context.handler_name == "MapHandler":
            return [proposed_command]

        return None

    def _coerce_int(self, value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None
