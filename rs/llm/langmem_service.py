from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Callable
import uuid
from urllib import error, request

from langchain_core.messages import HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.store.base import SearchItem
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.llm.config import LlmConfig
from rs.utils.config import config as llm_runtime_config


MemoryManagerFactory = Callable[[tuple[str, ...], InMemoryStore], Any]


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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langmem")
        self._reflection_buffers: dict[str, list[str]] = defaultdict(list)
        self._status = "disabled_by_config"
        self._status_detail = ""
        self._initialize()

    def _initialize(self) -> None:
        if not self._config.langmem_enabled:
            return

        try:
            if self._repository is None:
                self._repository = LangMemRepository(self._config.langmem_sqlite_path)
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
            self._hydrate_from_repository()
            self._status = "ready"
        except Exception as e:
            self._status = "embeddings_unavailable"
            self._status_detail = str(e)
            self._store = None

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
            self._store.put(record.namespace, record.memory_id, record.to_store_value(), index=["content"])

    def is_ready(self) -> bool:
        return self._status == "ready" and self._store is not None

    def status(self) -> str:
        if self._status_detail:
            return f"{self._status}:{self._status_detail}"
        return self._status

    def build_context_memory(self, context: AgentContext) -> dict[str, str]:
        if not self.is_ready():
            return {
                "retrieved_episodic_memories": "none",
                "retrieved_semantic_memories": "none",
                "langmem_status": self.status(),
            }

        query = self._build_query_text(context)
        episodic_namespace = self._run_namespace(context)
        semantic_namespace = self._semantic_namespace(context)

        with self._lock:
            episodic_items = self._store.search(episodic_namespace, query=query, limit=self._config.langmem_top_k)
            semantic_items = self._store.search(semantic_namespace, query=query, limit=self._config.langmem_top_k)

        return {
            "retrieved_episodic_memories": self._format_search_results(episodic_items, "EP"),
            "retrieved_semantic_memories": self._format_search_results(semantic_items, "SEM"),
            "langmem_status": self.status(),
        }

    def record_accepted_decision(self, context: AgentContext, decision: AgentDecision) -> None:
        if not self.is_ready():
            return
        if decision.proposed_command is None or decision.fallback_recommended:
            return

        record = self._build_episodic_record(context, decision)
        self._store_record(record)

        run_id = self._resolve_run_id(context)
        self._reflection_buffers[run_id].append(record.content)
        if len(self._reflection_buffers[run_id]) >= self._config.langmem_reflection_batch_size:
            batch = list(self._reflection_buffers[run_id])
            self._reflection_buffers[run_id].clear()
            self._executor.submit(self._reflect_batch, context, batch, "decision_batch")

    def finalize_run(self, context: AgentContext, payload: dict[str, Any]) -> None:
        if not self.is_ready():
            return

        run_id = self._resolve_run_id(context)
        batch = list(self._reflection_buffers.get(run_id, []))
        self._reflection_buffers[run_id].clear()
        run_summary = self._build_run_finalization_summary(payload)
        if run_summary:
            batch.append(run_summary)
        if batch:
            self._executor.submit(self._reflect_batch, context, batch, "run_finalization")

    def shutdown(self, wait: bool = True) -> None:
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
        content = (
            f"A{act} F{floor} {context.handler_name} chose {decision.proposed_command} "
            f"with confidence {decision.confidence:.2f} because {decision.explanation}."
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

    def _reflect_batch(self, context: AgentContext, batch: list[str], trigger: str) -> None:
        if not batch or not self.is_ready():
            return

        manager = self._build_reflection_manager(context)
        messages = [HumanMessage(content="\n".join(batch))]
        try:
            final_puts = manager.invoke({"messages": messages})
        except Exception:
            return

        source_run_id = self._resolve_run_id(context)
        for put in final_puts:
            namespace = tuple(put["namespace"])
            key = str(put["key"])
            value = dict(put["value"])
            content = _coerce_memory_content(value)
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
            semantic_value = record.to_store_value()
            semantic_value["trigger"] = trigger
            with self._lock:
                if self._repository is not None:
                    self._repository.upsert(record)
                if self._store is not None:
                    self._store.put(namespace, record.memory_id, semantic_value, index=["content"])
            self._prune_semantic_namespace(namespace)

    def _build_reflection_manager(self, context: AgentContext) -> Any:
        if self._reflection_manager_factory is not None:
            return self._reflection_manager_factory(self._semantic_namespace(context), self._store)

        chat_base_url = llm_runtime_config.llm_base_url or llm_runtime_config.openai_base_url or None
        chat_api_key = llm_runtime_config.llm_api_key or llm_runtime_config.openai_key or "langmem-reflection"
        chat_model = ChatOpenAI(
            model=llm_runtime_config.fast_llm_model,
            base_url=chat_base_url,
            api_key=chat_api_key,
            temperature=0.0,
        )
        return create_memory_store_manager(
            chat_model,
            store=self._store,
            namespace=self._semantic_namespace(context),
            enable_deletes=False,
            query_model=chat_model,
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

    def _run_namespace(self, context: AgentContext) -> tuple[str, ...]:
        strategy_name = self._resolve_strategy_name(context)
        character_class, seed = self._resolve_character_and_seed(context)
        return ("run", strategy_name, character_class, seed)

    def _semantic_namespace(self, context: AgentContext) -> tuple[str, ...]:
        strategy_name = self._resolve_strategy_name(context)
        character_class, _ = self._resolve_character_and_seed(context)
        return ("semantic", strategy_name, character_class, context.handler_name)

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
    def _resolve_strategy_name(context: AgentContext) -> str:
        strategy_name = str(context.extras.get("strategy_name", "unknown_strategy")).strip()
        return strategy_name or "unknown_strategy"

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
    def _format_search_results(items: list[SearchItem], source_tag: str) -> str:
        if not items:
            return "none"

        lines = []
        for item in items[:3]:
            score = item.score if item.score is not None else 0.0
            content = _coerce_memory_content(item.value)
            handler_name = str(item.value.get("handler_name", item.namespace[-1] if item.namespace else "memory"))
            lines.append(f"[{source_tag}|match={score:.2f}|{handler_name}] {content}")
        return " | ".join(lines)

    @staticmethod
    def _build_run_finalization_summary(payload: dict[str, Any]) -> str:
        floor = payload.get("floor", "unknown")
        victory = bool(payload.get("victory", False))
        score = payload.get("score", "unknown")
        bosses = ", ".join(payload.get("bosses", [])) or "none"
        elites = ", ".join(payload.get("elites", [])) or "none"
        return (
            f"Run ended on floor {floor} with score {score}. "
            f"Victory={victory}. Bosses seen: {bosses}. Elites seen: {elites}. "
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


def _coerce_memory_content(value: dict[str, Any]) -> str:
    content = value.get("content", "")
    if isinstance(content, dict):
        inner = content.get("content", content)
        return str(inner)
    return str(content)
