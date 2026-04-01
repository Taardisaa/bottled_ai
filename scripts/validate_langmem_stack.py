from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.llm.agents.base_agent import AgentContext
from rs.llm.config import load_llm_config
from rs.llm.langmem_service import LangMemRepository, LangMemService
from rs.utils.config import config as llm_runtime_config


def _build_openai_client() -> OpenAI:
    base_url = llm_runtime_config.llm_base_url or llm_runtime_config.openai_base_url
    api_key = llm_runtime_config.llm_api_key or llm_runtime_config.openai_key
    if not base_url:
        raise ValueError("Missing LLM/OpenAI base URL (LLM_BASE_URL or OPENAI_BASE_URL).")
    if not api_key:
        raise ValueError("Missing LLM/OpenAI API key (LLM_API_KEY or OPENAI_API_KEY).")
    return OpenAI(base_url=base_url, api_key=api_key)


def check_langmem_ready_and_retrieval() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.sqlite3"
        cfg = replace(load_llm_config(), langmem_enabled=True, langmem_sqlite_path=str(db_path))
        service = LangMemService(config=cfg)
        try:
            if not service.is_ready():
                return False, f"[LANGMEM_INIT] LangMem not ready: {service.status()}"

            token = f"LMEM_TOKEN_{uuid.uuid4().hex[:12]}"
            context = AgentContext(
                handler_name="EventHandler",
                screen_type="EVENT",
                available_commands=["choose 0"],
                choice_list=["option a"],
                game_state={"floor": 3, "act": 1, "character_class": "IRONCLAD"},
                extras={"run_id": f"ironclad:{uuid.uuid4().hex}", "agent_identity": "validator"},
            )
            service.record_custom_memory(
                context,
                f"User preference token is {token}.",
                tags=("validate_langmem_stack",),
                reflect=False,
            )
            payload = service.build_context_memory(context)
            episodic = str(payload.get("retrieved_episodic_memories", ""))
            if token not in episodic:
                return False, "[LANGMEM_RETRIEVAL] LangMem retrieval missing inserted episodic token."
        finally:
            service.shutdown(wait=True)

        repo = LangMemRepository(str(db_path))
        records = repo.load_all()
        if not records:
            return False, "[LANGMEM_SQLITE] SQLite has no LangMem rows after custom memory write."
        return True, "LangMem ready + episodic write/read + SQLite persistence are OK."


def check_native_tool_calling(model: str, temperature: float) -> tuple[bool, str]:
    client = _build_openai_client()
    token = f"TCALL_{uuid.uuid4().hex[:10]}"
    memories: list[str] = []
    tools = [
        {
            "type": "function",
            "function": {
                "name": "manage_memory",
                "description": "Persist a memory string.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search stored memories by query and return matching rows.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    messages: list[dict[str, object]] = [
        {"role": "user", "content": f"Remember this exact token in memory: {token}. Use memory tools."}
    ]

    # Step 1: model must issue manage_memory tool call.
    response1 = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools,
        tool_choice="required",
    )
    msg1 = response1.choices[0].message
    tool_calls1 = msg1.tool_calls or []
    if not tool_calls1:
        return False, "[NATIVE_TOOL_PROTOCOL] No tool_calls on remember step."
    call1 = tool_calls1[0]
    if call1.function.name != "manage_memory":
        return False, f"[NATIVE_TOOL_PROTOCOL] Expected manage_memory, got {call1.function.name}."
    args1 = call1.function.arguments or "{}"
    try:
        parsed1 = json.loads(args1)
    except Exception:
        return False, f"[NATIVE_TOOL_PROTOCOL] Invalid JSON in manage_memory args: {args1}"
    content = str(parsed1.get("content", ""))
    if token not in content:
        return False, "[NATIVE_TOOL_PROTOCOL] manage_memory args missing expected token."
    memories.append(content)

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call1.id,
                    "type": "function",
                    "function": {"name": call1.function.name, "arguments": args1},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call1.id,
            "content": json.dumps({"stored": True, "count": len(memories)}),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "What token did I ask you to remember? Use memory tools to look it up.",
        }
    )

    # Step 2: model must issue search_memory tool call.
    response2 = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools,
        tool_choice="required",
    )
    msg2 = response2.choices[0].message
    tool_calls2 = msg2.tool_calls or []
    if not tool_calls2:
        return False, "[NATIVE_TOOL_PROTOCOL] No tool_calls on recall step."
    call2 = tool_calls2[0]
    if call2.function.name != "search_memory":
        return False, f"[NATIVE_TOOL_PROTOCOL] Expected search_memory, got {call2.function.name}."
    args2 = call2.function.arguments or "{}"
    try:
        parsed2 = json.loads(args2)
    except Exception:
        return False, f"[NATIVE_TOOL_PROTOCOL] Invalid JSON in search_memory args: {args2}"
    query = str(parsed2.get("query", "")).strip()
    if query == "":
        return False, "[NATIVE_TOOL_PROTOCOL] search_memory query is empty."
    hits = [m for m in memories if query.lower() in m.lower()]
    if not hits:
        return False, "[NATIVE_TOOL_PROTOCOL] search_memory produced no matching hits."

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call2.id,
                    "type": "function",
                    "function": {"name": call2.function.name, "arguments": args2},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call2.id,
            "content": json.dumps({"hits": hits}),
        }
    )
    messages.append(
        {"role": "user", "content": "Now answer with only the exact token you found."}
    )
    response3 = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    final_text = (response3.choices[0].message.content or "").strip()
    if token not in final_text:
        return False, f"[NATIVE_TOOL_RECALL] Final answer did not include expected token. got={final_text!r}"
    return True, "Native tool-calling + store/retrieve/answer flow is OK."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate memory tools end-to-end (custom flow + native tool-calling).",
        epilog=(
            "Runbook: python scripts/validate_langmem_stack.py --model Qwen/Qwen3-32B --temperature 0.6\n"
            "Failure classes: [LANGMEM_INIT|LANGMEM_RETRIEVAL|LANGMEM_SQLITE|NATIVE_TOOL_PROTOCOL|NATIVE_TOOL_RECALL]"
        ),
    )
    parser.add_argument("--model", default=llm_runtime_config.fast_llm_model, help="Model id for tool-calling check.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for tool-call check.")
    args = parser.parse_args()

    failures: list[str] = []
    print("[1/2] Checking LangMem readiness + persistence/retrieval...")
    ok, msg = check_langmem_ready_and_retrieval()
    print("  -", msg)
    if not ok:
        failures.append(msg)

    print("[2/2] Checking native tool-calling on remote endpoint...")
    try:
        ok, msg = check_native_tool_calling(args.model, args.temperature)
    except Exception as exc:
        ok = False
        msg = f"[NATIVE_TOOL_PROTOCOL] Native tool-calling request failed: {exc}"
    print("  -", msg)
    if not ok:
        failures.append(msg)

    if failures:
        print("\nValidation FAILED:")
        for item in failures:
            print(" -", item)
        return 1

    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
