from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from rs.helper.logger import log_to_run
from rs.utils.path_utils import resolve_from_repo_root


@dataclass
class GraphTraceRecord:
    timestamp_utc: str
    thread_id: str
    run_id: str
    handler_name: str
    screen_type: str
    event_type: str
    node_name: str
    route_name: str
    decision_valid: bool | None
    validation_code: str
    proposed_command: str | None
    confidence: float | None
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_graph_trace_record(
        *,
        thread_id: str = "",
        run_id: str = "",
        handler_name: str = "",
        screen_type: str = "",
        event_type: str,
        node_name: str = "",
        route_name: str = "",
        decision_valid: bool | None = None,
        validation_code: str = "",
        proposed_command: str | None = None,
        confidence: float | None = None,
        summary: str = "",
        metadata: Dict[str, Any] | None = None,
) -> GraphTraceRecord:
    return GraphTraceRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
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
        summary=summary,
        metadata=dict(metadata or {}),
    )


def write_graph_trace(record: GraphTraceRecord, trace_path: str) -> None:
    output_path = resolve_from_repo_root(trace_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")


def mirror_graph_trace_to_run_log(record: GraphTraceRecord) -> None:
    parts = ["[AIPlayerGraphTrace]", record.event_type]
    if record.node_name:
        parts.append(record.node_name)
    if record.route_name:
        parts.append(f"route={record.route_name}")
    if record.summary:
        parts.append(record.summary)
    log_to_run(" | ".join(parts))
