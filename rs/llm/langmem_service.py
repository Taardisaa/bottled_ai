from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any, Callable
import uuid
from urllib import error, request

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.base import SearchItem
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

from rs.helper.logger import log_to_run
from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig
from rs.utils.config import config as llm_runtime_config


MemoryManagerFactory = Callable[[tuple[str, ...], InMemoryStore], Any]
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_COMPAT_PATCH_MARKER = "_compatibility_tool_call_bind_tools_patched"

_CONFIDENCE_RE = re.compile(
    r"""
      \(?                                   # optional opening paren
      (?:with\s+)?                           # optional "with "
      [Cc]onf(?:idence)?                     # "conf" or "confidence" or "Confidence"
      (?:\s*[:=]\s*|\s+)                     # separator: colon, equals, or space
      \d+\.\d+                               # decimal score
      \)?                                    # optional closing paren
    """,
    re.VERBOSE,
)

_HANDLER_NAME_RE = re.compile(
    r"\b(?:Battle|CardReward|CombatReward|Event|Map|GridSelect|Shop|Campfire)Handler\b"
    r"|\bRunFinalizer\b"
)

_SYSTEM_LABEL_RE = re.compile(
    r"\b(?:"
    r"guardrail_no_submission"
    r"|battle_subagent_no_progress_guardrail"
    r"|submit_battle_commands"
    r"|deterministic_card_handoff_when_only_card_row_remains"
    r"|deterministic_card_handoff_with_relic_selection"
    r"|deterministic_card_handoff"
    r"|all_cards_selected_confirm"
    r")\b"
)

_META_STATS_RE = re.compile(
    r"\(based\s+on\s+\d+\+?\s+instances?\s+across\s+\d+\+?\s+memory\s+records?\)"
)

_STS_EVENT_REFLECTION_INSTRUCTIONS = """You are extracting strategic lessons for a Slay the Spire AI agent.

The input is a session summary from one in-game event (battle, campfire, or reward). Your job is to distill
actionable strategic rules from it and store them for future reference.

Rules for what to store:
- Write only in plain game terms: card names, enemy names, floor ranges, HP values, room types (ELITE/BOSS/MONSTER).
- Answer any "Review:" questions in the summary by forming a concrete strategic rule.
- Generalise from specific plays to reusable situational rules. Examples of GOOD rules:
  "Against Jaw Worm, finish it quickly — it buffs itself over time, so longer fights cost more HP."
  "When an enemy is about to deal lethal damage, prioritise block cards or potions to survive."
  "In multi-enemy fights, focus fire on one enemy to reduce total incoming damage per turn."
- NEVER store: confidence scores, handler names, session IDs, guardrail names, or any internal system label.
  Handler names are internal labels like BattleHandler, CombatRewardHandler, MapHandler, CardRewardHandler —
  do NOT repeat them. If the input contains these labels, ignore them.
  BAD: "BattleHandler chose play 2 0 with confidence 0.85" — never write this.
- Consolidate: if a rule already exists and this observation reinforces it, strengthen or refine the existing
  memory rather than adding a duplicate.
- If an action was clearly suboptimal, record what should be done differently. Suboptimal actions include both:
  (a) taking unnecessary damage when blocking would have prevented it, AND
  (b) spending a turn blocking when finishing the enemy would have prevented more total damage."""

_STS_RETROSPECTIVE_INSTRUCTIONS = """You are extracting high-level strategic lessons for a Slay the Spire AI agent
by reviewing an entire completed run.

The input is a run narrative: the outcome (won/died), the floor reached, enemies encountered, and a sample of key
decisions made across the run. Your job is to identify what went wrong or right at a strategic level.

Rules for what to store:
- Write only in plain game terms: card names, enemy types, floor ranges, HP thresholds, relic names, act numbers.
- Focus on run-arc patterns: deck composition mismatches, missed upgrade opportunities, poor path choices, resource
  mismanagement across multiple floors.
- Good examples:
  "Dying before act 2 is often caused by fights lasting too many turns — a mix of damage and block cards in
  early rewards helps."
  "Skipping elite fights in act 1 preserves HP but weakens deck scaling into act 2 — consider the tradeoff."
  "Fights that dragged past 5+ turns caused the most HP loss across the run."
- NEVER store: confidence scores, handler names (BattleHandler, MapHandler, CombatRewardHandler, etc.),
  session IDs, command syntax details, or guardrail system labels. If the input mentions these labels,
  ignore them entirely.
  BAD: "BattleHandler chose play 3 0 with confidence 0.25 on floor 5" — never write this.
- Consolidate existing memories: if a pattern is already recorded, strengthen or update rather than duplicate.
- Store what to do DIFFERENTLY next run, not just what happened."""


def _wrap_bound_tool_runnable(bound: Any) -> Any:
    def _invoke(input_value: Any, config: Any = None) -> Any:
        try:
            response = bound.invoke(input_value, config=config)
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            response = bound.invoke(input_value)
        if isinstance(response, AIMessage):
            return _repair_ai_message_tool_calls(response)
        return response

    async def _ainvoke(input_value: Any, config: Any = None) -> Any:
        try:
            response = await bound.ainvoke(input_value, config=config)
        except TypeError as exc:
            if "config" not in str(exc):
                raise
            response = await bound.ainvoke(input_value)
        if isinstance(response, AIMessage):
            return _repair_ai_message_tool_calls(response)
        return response

    return RunnableLambda(_invoke, afunc=_ainvoke)


def _patch_bind_tools_method(owner: Any) -> None:
    bind_tools = getattr(owner, "bind_tools", None)
    if bind_tools is None or getattr(bind_tools, _COMPAT_PATCH_MARKER, False):
        return

    @wraps(bind_tools)
    def _patched_bind_tools(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = bind_tools(self, *args, **kwargs)
        return _wrap_bound_tool_runnable(bound)

    setattr(_patched_bind_tools, _COMPAT_PATCH_MARKER, True)
    setattr(owner, "bind_tools", _patched_bind_tools)


def install_compatibility_tool_call_patch() -> None:
    """Monkey patch common chat-model bind_tools paths to repair markup tool calls."""
    _patch_bind_tools_method(BaseChatModel)
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI
    except Exception:
        return
    _patch_bind_tools_method(BaseChatOpenAI)


install_compatibility_tool_call_patch()


class CompatibilityToolCallChatModel(BaseChatModel):
    """Wrap a chat model and repair tool calls emitted as plain-text markup."""

    model: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return f"compatibility_tool_call::{getattr(self.model, '_llm_type', 'unknown')}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        identifying = getattr(self.model, "_identifying_params", {})
        if isinstance(identifying, dict):
            return dict(identifying)
        return {"wrapped_model": str(identifying)}

    def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            **kwargs: Any,
    ) -> ChatResult:
        response = self.model.invoke(messages, stop=stop, **kwargs)
        if not isinstance(response, AIMessage):
            raise TypeError(f"Expected AIMessage from wrapped model, got {type(response).__name__}")
        repaired = _repair_ai_message_tool_calls(response)
        return ChatResult(generations=[ChatGeneration(message=repaired)])

    async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            **kwargs: Any,
    ) -> ChatResult:
        response = await self.model.ainvoke(messages, stop=stop, **kwargs)
        if not isinstance(response, AIMessage):
            raise TypeError(f"Expected AIMessage from wrapped model, got {type(response).__name__}")
        repaired = _repair_ai_message_tool_calls(response)
        return ChatResult(generations=[ChatGeneration(message=repaired)])

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> Any:
        bound = self.model.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return _wrap_bound_tool_runnable(bound)


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [str(text) for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    namespace: tuple[str, ...]
    memory_type: str
    content: str
    source_run_id: str
    handler_name: str
    created_at_utc: str
    updated_at_utc: str
    tags: tuple[str, ...]
    kind: str = "Memory"

    def to_store_value(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_run_id": self.source_run_id,
            "handler_name": self.handler_name,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "tags": list(self.tags),
        }


class LangMemRepository:
    def __init__(self, sqlite_path: str):
        self._path = Path(sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS langmem_records (
                    memory_id TEXT PRIMARY KEY,
                    namespace_json TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    handler_name TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    kind TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_langmem_namespace
                ON langmem_records(namespace_json, memory_type, updated_at_utc)
                """
            )
            connection.commit()

    def load_all(self) -> list[MemoryRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, namespace_json, memory_type, content, source_run_id,
                       handler_name, created_at_utc, updated_at_utc, tags_json, kind
                FROM langmem_records
                ORDER BY created_at_utc ASC
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def upsert(self, record: MemoryRecord) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO langmem_records (
                    memory_id, namespace_json, memory_type, content, source_run_id,
                    handler_name, created_at_utc, updated_at_utc, tags_json, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    namespace_json=excluded.namespace_json,
                    memory_type=excluded.memory_type,
                    content=excluded.content,
                    source_run_id=excluded.source_run_id,
                    handler_name=excluded.handler_name,
                    created_at_utc=excluded.created_at_utc,
                    updated_at_utc=excluded.updated_at_utc,
                    tags_json=excluded.tags_json,
                    kind=excluded.kind
                """,
                (
                    record.memory_id,
                    json.dumps(list(record.namespace)),
                    record.memory_type,
                    record.content,
                    record.source_run_id,
                    record.handler_name,
                    record.created_at_utc,
                    record.updated_at_utc,
                    json.dumps(list(record.tags)),
                    record.kind,
                ),
            )
            connection.commit()

    def delete(self, memory_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM langmem_records WHERE memory_id = ?", (memory_id,))
            connection.commit()

    def list_namespace(self, namespace: tuple[str, ...], memory_type: str) -> list[MemoryRecord]:
        namespace_json = json.dumps(list(namespace))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, namespace_json, memory_type, content, source_run_id,
                       handler_name, created_at_utc, updated_at_utc, tags_json, kind
                FROM langmem_records
                WHERE namespace_json = ? AND memory_type = ?
                ORDER BY updated_at_utc DESC, created_at_utc DESC
                """,
                (namespace_json, memory_type),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=str(row["memory_id"]),
            namespace=tuple(json.loads(row["namespace_json"])),
            memory_type=str(row["memory_type"]),
            content=str(row["content"]),
            source_run_id=str(row["source_run_id"]),
            handler_name=str(row["handler_name"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
            tags=tuple(json.loads(row["tags_json"])),
            kind=str(row["kind"]),
        )


class LangMemService:
    def __init__(
            self,
            config: LlmConfig,
            repository: LangMemRepository | None = None,
            embeddings_client: Any | None = None,
            reflection_manager_factory: MemoryManagerFactory | None = None,
    ):
        self._config = config
        self._repository = repository
        self._embeddings_client = embeddings_client
        self._reflection_manager_factory = reflection_manager_factory
        self._store: InMemoryStore | None = None
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, config.langmem_max_reflection_workers),
            thread_name_prefix="langmem",
        )
        self._reflection_buffers: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._deferred_reflections: list[tuple[AgentContext, list[str], str]] = []
        self._reflections_paused = False
        self._status = "disabled_by_config"
        self._status_detail = ""
        self._initialize()

    def _initialize(self) -> None:
        if not self._config.langmem_enabled:
            return

        try:
            embeddings_client = self._build_embeddings_client()
            health_vector = embeddings_client.embed_query("langmem health check")
            if not isinstance(health_vector, list) or not health_vector:
                raise ValueError("empty_embedding_vector")
            dims = len(health_vector)
            self._embeddings_client = embeddings_client
            self._store = InMemoryStore(index={
                "embed": embeddings_client,
                "dims": dims,
                "fields": ["content"],
            })
            if self._repository is None:
                self._repository = LangMemRepository(self._config.langmem_sqlite_path)
            self._hydrate_from_repository()
            self._status = "ready"
        except Exception as e:
            self._status = "embeddings_unavailable"
            self._status_detail = str(e)
            self._store = None
            if self._config.langmem_fail_fast_init:
                raise RuntimeError(f"LangMem initialization failed: {self.status()}") from e

    def _build_embeddings_client(self) -> Embeddings:
        if self._embeddings_client is not None:
            return self._embeddings_client

        base_url = str(self._config.langmem_embeddings_base_url or "").strip()
        api_key = self._config.langmem_embeddings_api_key or llm_runtime_config.llm_api_key or "local-embeddings"
        if base_url:
            self._verify_embeddings_model_available(base_url, api_key, self._config.langmem_embeddings_model)
            return OpenAIEmbeddings(
                model=self._config.langmem_embeddings_model,
                base_url=base_url,
                api_key=api_key,
                tiktoken_enabled=False,
            )
        return self._build_local_embeddings_client(self._config.langmem_embeddings_model)

    def _build_local_embeddings_client(self, model_name: str) -> Embeddings:
        return LocalSentenceTransformerEmbeddings(model_name)

    def _verify_embeddings_model_available(self, base_url: str, api_key: str, model_name: str) -> None:
        models_url = base_url.rstrip("/") + "/models"
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        http_request = request.Request(models_url, headers=headers, method="GET")
        try:
            with request.urlopen(http_request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"embeddings_models_check_failed:{exc.code}:{error_body}") from exc
        except Exception as exc:
            raise ValueError(f"embeddings_models_check_failed:{exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("embeddings_models_check_failed:missing_models_data")

        available_models = {
            str(item.get("id", "")).strip()
            for item in data
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        if model_name not in available_models:
            available_preview = ", ".join(sorted(available_models)[:10]) or "none"
            raise ValueError(
                f"embeddings_model_not_available:{model_name}; available_models={available_preview}"
            )

    def _hydrate_from_repository(self) -> None:
        if self._store is None or self._repository is None:
            return

        for record in self._repository.load_all():
            self._store.put(
                record.namespace,
                record.memory_id,
                _build_store_value(record),
                index=["content"],
            )

    def is_ready(self) -> bool:
        return self._status == "ready" and self._store is not None

    def status(self) -> str:
        if self._status_detail:
            return f"{self._status}:{self._status_detail}"
        return self._status

    def build_context_memory(self, context: AgentContext) -> dict[str, str]:
        import time as _time
        if not self.is_ready():
            return {
                "retrieved_episodic_memories": "none",
                "retrieved_semantic_memories": "none",
                "langmem_status": self.status(),
            }

        _t0 = _time.perf_counter()
        query = self._build_query_text(context)
        episodic_namespace = self._run_namespace(context)
        semantic_namespace = self._semantic_namespace(context)
        retrospective_namespace = self._retrospective_namespace(context)

        inject_episodic = self._config.langmem_inject_episodic_memories
        inject_semantic = self._config.langmem_inject_semantic_memories
        with self._lock:
            episodic_items = self._store.search(episodic_namespace, query=query, limit=self._config.langmem_top_k) if inject_episodic else []
            semantic_items = self._store.search(semantic_namespace, query=query, limit=self._config.langmem_top_k) if inject_semantic else []
            retrospective_items = self._store.search(
                retrospective_namespace, query=query, limit=self._config.langmem_top_k
            ) if inject_semantic else []
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        log_to_run(f"[TIMING] LangMem.build_context_memory handler={context.handler_name} took {_elapsed_ms:.0f}ms")

        retrieved_episodic_memories = self._format_search_results(episodic_items, "EP")
        semantic_str = self._format_search_results(semantic_items, "SEM")
        retro_str = self._format_search_results(retrospective_items, "RETRO")
        if retro_str != "none" and semantic_str != "none":
            retrieved_semantic_memories = f"{semantic_str} | {retro_str}"
        elif retro_str != "none":
            retrieved_semantic_memories = retro_str
        else:
            retrieved_semantic_memories = semantic_str
        langmem_status = self.status()
        log_to_run(
            "LangMem retrieval | "
            f"handler={context.handler_name} | "
            f"run_namespace={episodic_namespace} | "
            f"semantic_namespace={semantic_namespace} | "
            f"episodic={retrieved_episodic_memories} | "
            f"semantic={retrieved_semantic_memories} | "
            f"status={langmem_status}"
        )
        return {
            "retrieved_episodic_memories": retrieved_episodic_memories,
            "retrieved_semantic_memories": retrieved_semantic_memories,
            "langmem_status": langmem_status,
        }

    def record_accepted_decision(self, context: AgentContext, decision: AgentDecision) -> None:
        if not self.is_ready():
            return
        if decision.proposed_command is None or decision.fallback_recommended:
            return
        if decision.confidence < self._config.langmem_min_record_confidence:
            return

        record = self._build_episodic_record(context, decision)
        self._store_record(record)

        run_id = self._resolve_run_id(context)
        buf_key = (run_id, context.handler_name)
        self._reflection_buffers[buf_key].append(record.content)
        if len(self._reflection_buffers[buf_key]) >= self._config.langmem_reflection_batch_size:
            batch = list(self._reflection_buffers[buf_key])
            self._reflection_buffers[buf_key].clear()
            self._submit_or_defer_reflection(context, batch, "decision_batch")

    def record_custom_memory(
            self,
            context: AgentContext,
            content: str,
            *,
            tags: tuple[str, ...] = (),
            reflect: bool = False,
    ) -> None:
        if not self.is_ready():
            return

        normalized_content = str(content).strip()
        if normalized_content == "":
            return

        now = _utc_now()
        record = MemoryRecord(
            memory_id=uuid.uuid4().hex,
            namespace=self._run_namespace(context),
            memory_type="episodic",
            content=normalized_content,
            source_run_id=self._resolve_run_id(context),
            handler_name=context.handler_name,
            created_at_utc=now,
            updated_at_utc=now,
            tags=("custom_memory", context.handler_name, *tags),
        )
        self._store_record(record)

        if reflect:
            self._submit_or_defer_reflection(context, [normalized_content], "custom_memory")

    def finalize_run(self, context: AgentContext, payload: dict[str, Any]) -> None:
        if not self.is_ready():
            return

        self.resume_reflections()

        run_id = self._resolve_run_id(context)
        batch = []
        for key in list(self._reflection_buffers):
            if key[0] == run_id:
                batch.extend(self._reflection_buffers[key])
                self._reflection_buffers[key].clear()
        run_summary = self._build_run_finalization_summary(payload)
        if run_summary:
            batch.append(run_summary)
        if batch:
            self._executor.submit(self._reflect_batch, context, batch, "run_finalization")
        self._executor.submit(self._submit_retrospective_reflection, context, payload)

    def _submit_retrospective_reflection(self, context: AgentContext, payload: dict[str, Any]) -> None:
        if not self.is_ready() or self._repository is None:
            return
        run_namespace = self._run_namespace(context)
        episodic_records = self._repository.list_namespace(run_namespace, "episodic")
        if not episodic_records:
            return
        # Sample: first 5 + last 20 to capture early decisions and run-end arc
        sample = episodic_records[:5]
        if len(episodic_records) > 5:
            sample = sample + episodic_records[max(5, len(episodic_records) - 20):]
        lines = [record.content for record in sample]
        run_summary = self._build_run_finalization_summary(payload)
        run_memory_summary = str(payload.get("run_memory_summary", "")).strip()
        narrative_parts = [run_summary]
        if run_memory_summary:
            narrative_parts.append(f"Run summary: {run_memory_summary}")
        narrative_parts.append("Key decisions this run:")
        narrative_parts.extend(lines)
        narrative = "\n".join(narrative_parts)
        self._reflect_batch(context, [narrative], "run_retrospective")

    def pause_reflections(self) -> None:
        with self._lock:
            self._reflections_paused = True

    def resume_reflections(self) -> None:
        with self._lock:
            self._reflections_paused = False
            pending = list(self._deferred_reflections)
            self._deferred_reflections.clear()
        for context, batch, trigger in pending:
            self._executor.submit(self._reflect_batch, context, batch, trigger)

    def _submit_or_defer_reflection(
            self, context: AgentContext, batch: list[str], trigger: str,
    ) -> None:
        with self._lock:
            if self._reflections_paused:
                self._deferred_reflections.append((context, batch, trigger))
                return
        self._executor.submit(self._reflect_batch, context, batch, trigger)

    def shutdown(self, wait: bool = True) -> None:
        self.resume_reflections()
        self._executor.shutdown(wait=wait)

    def __deepcopy__(self, memo: dict[int, Any]) -> LangMemService:
        memo[id(self)] = self
        return self

    def _store_record(self, record: MemoryRecord) -> None:
        if self._store is None:
            return

        with self._lock:
            if self._repository is not None:
                self._repository.upsert(record)
            self._store.put(record.namespace, record.memory_id, record.to_store_value(), index=["content"])

    def _build_episodic_record(self, context: AgentContext, decision: AgentDecision) -> MemoryRecord:
        now = _utc_now()
        floor = context.game_state.get("floor", "?")
        act = context.game_state.get("act", "?")
        if context.handler_name == "BattleHandler":
            content = self._build_battle_episodic_content(context, decision, act, floor)
        else:
            reason = _TOOL_CALL_BLOCK_RE.sub("", decision.explanation).strip() or "agent_step"
            content = (
                f"A{act} F{floor} {context.screen_type} chose {decision.proposed_command} "
                f"because {reason}."
            )
        return MemoryRecord(
            memory_id=uuid.uuid4().hex,
            namespace=self._run_namespace(context),
            memory_type="episodic",
            content=content,
            source_run_id=self._resolve_run_id(context),
            handler_name=context.handler_name,
            created_at_utc=now,
            updated_at_utc=now,
            tags=("accepted_decision", context.handler_name),
        )

    @staticmethod
    def _build_battle_episodic_content(
            context: AgentContext, decision: AgentDecision, act: Any, floor: Any
    ) -> str:
        room_type = context.game_state.get("room_type", "MONSTER")
        turn = context.game_state.get("turn", "?")
        current_hp = context.game_state.get("current_hp", "?")
        max_hp = context.game_state.get("max_hp", "?")
        energy = context.extras.get("player_energy", "?")
        monsters = context.extras.get("monster_summaries", [])
        if monsters:
            m = monsters[0]
            monster_desc = (
                f"{m.get('name', 'enemy')} HP {m.get('current_hp', '?')}/{m.get('max_hp', '?')} "
                f"intent={m.get('intent', '?')}"
            )
        else:
            monster_desc = "no enemy"
        reason = _TOOL_CALL_BLOCK_RE.sub("", decision.explanation).strip() or "battle_subagent_step"
        return (
            f"A{act} F{floor} {room_type} T{turn} | HP {current_hp}/{max_hp} E{energy} | "
            f"{monster_desc} | chose {decision.proposed_command} because {reason}."
        )

    def _reflect_batch(self, context: AgentContext, batch: list[str], trigger: str) -> None:
        import time as _time
        if not batch or not self.is_ready():
            return

        _t0 = _time.perf_counter()
        manager = self._build_reflection_manager(context, trigger=trigger)
        messages = [HumanMessage(content="\n".join(batch))]
        final_puts: list[dict[str, Any]] = []
        for _ in range(3):
            try:
                final_puts = manager.invoke({"messages": messages})
            except Exception:
                final_puts = []
                continue
            if final_puts:
                break
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        log_to_run(
            f"[TIMING] LangMem._reflect_batch trigger={trigger} handler={context.handler_name} "
            f"took {_elapsed_ms:.0f}ms puts={len(final_puts)}"
        )
        if not final_puts:
            return

        source_run_id = self._resolve_run_id(context)
        for put in final_puts:
            namespace = tuple(put["namespace"])
            key = str(put["key"])
            value = dict(put["value"])
            content = self._sanitize_reflection_content(_coerce_memory_content(value))
            now = _utc_now()
            record = MemoryRecord(
                memory_id=key,
                namespace=namespace,
                memory_type="semantic",
                content=content,
                source_run_id=source_run_id,
                handler_name=context.handler_name,
                created_at_utc=value.get("created_at_utc", now),
                updated_at_utc=now,
                tags=("semantic", trigger, context.handler_name),
                kind=str(value.get("kind", "Memory")),
            )
            raw_content = value.get("content")
            if isinstance(raw_content, dict) and "content" in raw_content:
                raw_content = dict(raw_content)
                raw_content["content"] = self._sanitize_reflection_content(str(raw_content["content"]))
            elif isinstance(raw_content, str):
                raw_content = self._sanitize_reflection_content(raw_content)
            semantic_value = _build_store_value(record, raw_content=raw_content)
            semantic_value["trigger"] = trigger
            with self._lock:
                if self._repository is not None:
                    self._repository.upsert(record)
                if self._store is not None:
                    self._store.put(namespace, record.memory_id, semantic_value, index=["content"])
            if trigger in ("run_retrospective", "run_finalization"):
                self._prune_retrospective_namespace(namespace)
            else:
                self._prune_semantic_namespace(namespace)

    def _build_reflection_manager(self, context: AgentContext, trigger: str = "") -> Any:
        is_retrospective = trigger in ("run_retrospective", "run_finalization")
        if is_retrospective:
            namespace = self._retrospective_namespace(context)
            instructions = _STS_RETROSPECTIVE_INSTRUCTIONS
        else:
            namespace = self._semantic_namespace(context)
            instructions = _STS_EVENT_REFLECTION_INSTRUCTIONS

        if self._reflection_manager_factory is not None:
            return self._reflection_manager_factory(namespace, self._store)

        chat_base_url = llm_runtime_config.llm_base_url or llm_runtime_config.openai_base_url or None
        chat_api_key = llm_runtime_config.llm_api_key or llm_runtime_config.openai_key or "langmem-reflection"
        reasoning_kwargs: dict[str, Any] = {}
        if llm_runtime_config.llm_base_url:
            effort = "high" if llm_runtime_config.llm_enable_thinking else "none"
            reasoning_kwargs = {"extra_body": {"reasoning_effort": effort}}
        chat_model = ChatOpenAI(
            model=llm_runtime_config.fast_llm_model,
            base_url=chat_base_url,
            api_key=chat_api_key,
            temperature=0.6,
            model_kwargs=reasoning_kwargs,
        )
        compatible_chat_model = CompatibilityToolCallChatModel(model=chat_model)
        return create_memory_store_manager(
            compatible_chat_model,
            store=self._store,
            namespace=namespace,
            instructions=instructions,
            enable_deletes=False,
            query_model=compatible_chat_model,
            query_limit=self._config.langmem_top_k,
        )

    def _prune_semantic_namespace(self, namespace: tuple[str, ...]) -> None:
        if self._repository is None:
            return
        semantic_records = self._repository.list_namespace(namespace, "semantic")
        for record in semantic_records[self._config.langmem_max_semantic_memories_per_namespace:]:
            self._repository.delete(record.memory_id)
            if self._store is not None:
                self._store.delete(namespace, record.memory_id)

    def _prune_retrospective_namespace(self, namespace: tuple[str, ...]) -> None:
        if self._repository is None:
            return
        retro_records = self._repository.list_namespace(namespace, "semantic")
        for record in retro_records[self._config.langmem_max_retrospective_memories:]:
            self._repository.delete(record.memory_id)
            if self._store is not None:
                self._store.delete(namespace, record.memory_id)

    def _run_namespace(self, context: AgentContext) -> tuple[str, ...]:
        agent_identity = self._resolve_agent_identity(context)
        character_class, seed = self._resolve_character_and_seed(context)
        return ("run", agent_identity, character_class, seed)

    def _semantic_namespace(self, context: AgentContext) -> tuple[str, ...]:
        agent_identity = self._resolve_agent_identity(context)
        character_class, _ = self._resolve_character_and_seed(context)
        return ("semantic", agent_identity, character_class, context.handler_name)

    def _retrospective_namespace(self, context: AgentContext) -> tuple[str, ...]:
        agent_identity = self._resolve_agent_identity(context)
        character_class, _ = self._resolve_character_and_seed(context)
        return ("retrospective", agent_identity, character_class)

    @staticmethod
    def _resolve_run_id(context: AgentContext) -> str:
        raw = str(context.extras.get("run_id", "unknown:unknown")).strip()
        return raw or "unknown:unknown"

    def _resolve_character_and_seed(self, context: AgentContext) -> tuple[str, str]:
        run_id = self._resolve_run_id(context)
        if ":" in run_id:
            character_class, seed = run_id.split(":", 1)
            return character_class.strip() or "unknown_class", seed.strip() or "unknown_seed"

        character_class = str(context.game_state.get("character_class", "unknown_class")).strip()
        return character_class or "unknown_class", "unknown_seed"

    @staticmethod
    def _resolve_agent_identity(context: AgentContext) -> str:
        agent_identity = str(context.extras.get("agent_identity", "neo_primates")).strip()
        return agent_identity or "neo_primates"

    @staticmethod
    def _build_query_text(context: AgentContext) -> str:
        return (
            f"{context.handler_name} on {context.screen_type}. "
            f"Act {context.game_state.get('act', 'unknown')} floor {context.game_state.get('floor', 'unknown')}. "
            f"Choices: {context.choice_list}. "
            f"Run summary: {context.extras.get('run_memory_summary', '')}. "
            f"Relics: {context.extras.get('relic_names', [])}. "
            f"Deck profile: {context.extras.get('deck_profile', {})}."
        )

    @staticmethod
    def _sanitize_reflection_content(content: str) -> str:
        text = _CONFIDENCE_RE.sub("", content)
        text = _HANDLER_NAME_RE.sub("", text)
        text = _SYSTEM_LABEL_RE.sub("", text)
        text = _META_STATS_RE.sub("", text)
        text = re.sub(r"\(\s*\)", "", text)       # empty parens
        text = re.sub(r",\s*,", ",", text)        # double commas
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # space before punctuation
        text = re.sub(r"  +", " ", text)           # multiple spaces
        return text.strip()

    def _format_search_results(self, items: list[SearchItem], source_tag: str) -> str:
        if not items:
            return "none"

        min_score = self._config.langmem_min_similarity_score
        items = [item for item in items if (item.score or 0.0) >= min_score]
        if not items:
            return "none"

        lines = []
        for item in items:
            score = item.score if item.score is not None else 0.0
            content = _coerce_memory_content(item.value)
            handler_name = str(item.value.get("handler_name", item.namespace[-1] if item.namespace else "memory"))
            lines.append(f"[{source_tag}|match={score:.2f}|{handler_name}] {content}")
        return " | ".join(lines)

    @staticmethod
    def _build_run_finalization_summary(payload: dict[str, Any]) -> str:
        floor = payload.get("floor", "unknown")
        score = payload.get("score", "unknown")
        bosses = ", ".join(payload.get("bosses", [])) or "none"
        elites = ", ".join(payload.get("elites", [])) or "none"
        victory = payload.get("victory")
        outcome = "won" if victory is True else ("died" if victory is False else "ended")
        return (
            f"Run {outcome} on floor {floor} with score {score}. "
            f"Bosses seen: {bosses}. Elites seen: {elites}. "
            f"Summary: {payload.get('run_memory_summary', '')}"
        )


_langmem_service: LangMemService | None = None


def get_langmem_service(config: LlmConfig | None = None) -> LangMemService:
    global _langmem_service
    if _langmem_service is None:
        from rs.llm.config import load_llm_config

        _langmem_service = LangMemService(load_llm_config() if config is None else config)
    return _langmem_service


def shutdown_langmem_service(wait: bool = True) -> None:
    global _langmem_service
    if _langmem_service is not None:
        _langmem_service.shutdown(wait=wait)
        _langmem_service = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_store_value(record: MemoryRecord, raw_content: Any | None = None) -> dict[str, Any]:
    stored_content = raw_content
    if stored_content is None:
        if record.memory_type == "semantic" and record.kind == "Memory":
            stored_content = {"content": record.content}
        else:
            stored_content = record.content
    return {
        "kind": record.kind,
        "content": stored_content,
        "memory_type": record.memory_type,
        "source_run_id": record.source_run_id,
        "handler_name": record.handler_name,
        "created_at_utc": record.created_at_utc,
        "updated_at_utc": record.updated_at_utc,
        "tags": list(record.tags),
    }


def _repair_ai_message_tool_calls(message: AIMessage) -> AIMessage:
    if message.tool_calls:
        return message

    parsed_tool_calls = _parse_tool_calls_from_content(message.content)
    if not parsed_tool_calls:
        return message

    return AIMessage(
        content=message.content,
        additional_kwargs=dict(message.additional_kwargs),
        response_metadata=dict(message.response_metadata),
        name=message.name,
        id=message.id,
        tool_calls=parsed_tool_calls,
        invalid_tool_calls=list(message.invalid_tool_calls),
        usage_metadata=message.usage_metadata,
    )


def repair_ai_message_tool_calls(message: AIMessage) -> AIMessage:
    """Public compatibility hook for providers that emit tool calls as text markup."""
    return _repair_ai_message_tool_calls(message)


def _parse_tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content.strip():
        return []

    tool_calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(content):
        payload_text = match.group(1).strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        name = str(payload.get("name", "")).strip()
        if not name:
            continue

        args = payload.get("arguments", payload.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}

        tool_calls.append({
            "name": name,
            "args": args,
            "id": str(payload.get("id") or f"compat_tool_{uuid.uuid4().hex[:12]}"),
            "type": "tool_call",
        })
    return tool_calls


def _coerce_memory_content(value: dict[str, Any]) -> str:
    content = value.get("content", "")
    if isinstance(content, dict):
        inner = content.get("content", content)
        return str(inner)
    return str(content)
