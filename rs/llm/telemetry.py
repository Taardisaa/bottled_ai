from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from rs.llm.agents.base_agent import AgentContext, AgentDecision
from rs.utils.hash_utils import sha256_from_json_payload
from rs.utils.path_utils import resolve_from_repo_root


@dataclass
class DecisionTelemetry:
    timestamp_utc: str
    state_snapshot_hash: str
    handler_name: str
    screen_type: str
    tool_calls_used: List[str]
    proposed_command: str | None
    validation_result: str
    fallback_used: bool
    latency_ms: int
    estimated_cost_usd: float | None
    token_in: int | None
    token_out: int | None
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_state_snapshot_hash(context: AgentContext) -> str:
    """Compute stable state hash for telemetry grouping.

    Args:
        context: Decision context containing state and high-level fields.

    Returns:
        str: SHA256 hash of normalized state payload.
    """
    payload = {
        "handler_name": context.handler_name,
        "screen_type": context.screen_type,
        "available_commands": context.available_commands,
        "choice_list": context.choice_list,
        "game_state": context.game_state,
        "extras": context.extras,
    }
    return sha256_from_json_payload(payload)


def build_decision_telemetry(
    context: AgentContext,
    decision: AgentDecision,
    latency_ms: int,
) -> DecisionTelemetry:
    """Build structured decision telemetry from context and advisor output.

    Args:
        context: Decision context.
        decision: Agent decision object.
        latency_ms: Wall-clock decision latency in milliseconds.

    Returns:
        DecisionTelemetry: Structured telemetry row for JSONL persistence.
    """
    metadata = decision.metadata.copy()
    token_in = metadata.pop("token_in", None)
    token_out = metadata.pop("token_out", None)
    estimated_cost_usd = metadata.pop("estimated_cost_usd", None)
    validation_result = str(metadata.get("validation_error", "ok"))

    return DecisionTelemetry(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        state_snapshot_hash=compute_state_snapshot_hash(context),
        handler_name=context.handler_name,
        screen_type=context.screen_type,
        tool_calls_used=decision.required_tools_used,
        proposed_command=decision.proposed_command,
        validation_result=validation_result,
        fallback_used=decision.fallback_recommended,
        latency_ms=latency_ms,
        estimated_cost_usd=float(estimated_cost_usd) if estimated_cost_usd is not None else None,
        token_in=int(token_in) if token_in is not None else None,
        token_out=int(token_out) if token_out is not None else None,
        confidence=decision.confidence,
        metadata=metadata,
    )


def write_decision_telemetry(telemetry: DecisionTelemetry, telemetry_path: str) -> None:
    """Append decision telemetry as JSON line to a telemetry file.

    Args:
        telemetry: Structured telemetry object.
        telemetry_path: Relative or absolute path for JSONL file.

    Returns:
        None.
    """
    output_path = resolve_from_repo_root(telemetry_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(asdict(telemetry), sort_keys=True) + "\n")
