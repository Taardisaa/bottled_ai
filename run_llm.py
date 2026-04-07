from __future__ import annotations

import argparse
import json

from litellm import completion

from rs.utils.config import load_config, Config
from rs.utils import llm_utils
from rs.utils.llm_utils import (
    _call_litellm_quietly,
    _litellm_completion_kwargs,
    _litellm_routed_model_id,
    run_llm_preflight_check,
)


CHECKS = ("all", "availability", "tools", "reasoning")


def _mask_key(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


# ---------------------------------------------------------------------------
# Check 1: Availability
# ---------------------------------------------------------------------------

def check_availability(model: str) -> tuple[bool, str]:
    result = run_llm_preflight_check(model, probe_message="Reply with exactly: OK")

    print(f"  routed_model: {result.routed_model}")
    print(f"  provider: {result.provider}")
    print(f"  endpoint: {result.endpoint}")
    print(f"  max_tokens: {result.max_tokens}")

    if not result.available:
        print(f"  status: FAILED")
        print(f"  error: {result.error}")
        return False, result.error

    print(f"  response_model: {result.response_model}")
    print(f"  response_preview: {result.response_preview}")
    print(f"  total_tokens: {result.total_tokens}")
    print(f"  status: OK")
    return True, ""


# ---------------------------------------------------------------------------
# Check 2: Tool call support
# ---------------------------------------------------------------------------

_TOOL_DEFINITION = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]


def check_tool_call_support(model: str) -> tuple[bool, str]:
    try:
        kwargs = _litellm_completion_kwargs(model, 0.0)
        raw = _call_litellm_quietly(
            completion,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=_TOOL_DEFINITION,
            tool_choice="required",
            **kwargs,
        )

        if hasattr(raw, "model_dump"):
            resp = raw.model_dump()
        elif isinstance(raw, dict):
            resp = raw
        else:
            msg = f"unexpected response type: {type(raw)!r}"
            print(f"  status: FAILED")
            print(f"  error: {msg}")
            return False, msg

        message = resp.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            print(f"  status: FAILED")
            print(f"  error: model returned no tool calls")
            # Show what the model returned instead
            content = message.get("content", "")
            if content:
                preview = content[:200] + ("..." if len(content) > 200 else "")
                print(f"  response_content: {preview}")
            return False, "model returned no tool calls"

        first_call = tool_calls[0]
        func = first_call.get("function", {})
        func_name = func.get("name", "")
        func_args_raw = func.get("arguments", "")

        print(f"  tool_name: {func_name}")
        print(f"  arguments: {func_args_raw}")

        if func_name != "get_weather":
            msg = f"expected tool name 'get_weather', got '{func_name}'"
            print(f"  status: FAILED")
            print(f"  error: {msg}")
            return False, msg

        try:
            parsed_args = json.loads(func_args_raw) if isinstance(func_args_raw, str) else func_args_raw
        except json.JSONDecodeError as e:
            msg = f"tool arguments not valid JSON: {e}"
            print(f"  status: FAILED")
            print(f"  error: {msg}")
            return False, msg

        print(f"  parsed_args: {parsed_args}")
        print(f"  status: OK")
        return True, ""

    except Exception as e:
        msg = str(e)
        print(f"  status: FAILED")
        print(f"  error: {msg}")
        return False, msg


# ---------------------------------------------------------------------------
# Check 3: Reasoning toggle
# Uses reasoning_effort ("none" / "high") via extra_body.
# Detects reasoning via the reasoning_content field in the response message.
# ---------------------------------------------------------------------------

_REASONING_PROMPT = "What is 17 * 23?"


def _extract_message(raw: object) -> dict:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return {}
    return raw.get("choices", [{}])[0].get("message", {}) or {}


def _do_reasoning_call(model: str, effort: str) -> dict:
    kwargs = _litellm_completion_kwargs(model, 0.6)
    extra_body = kwargs.pop("extra_body", {}) or {}
    extra_body["reasoning_effort"] = effort
    kwargs["extra_body"] = extra_body

    raw = _call_litellm_quietly(
        completion,
        messages=[{"role": "user", "content": _REASONING_PROMPT}],
        **kwargs,
    )
    return _extract_message(raw)


def _get_reasoning(message: dict) -> str | None:
    reasoning = message.get("reasoning_content")
    if reasoning:
        return str(reasoning)
    psf = message.get("provider_specific_fields") or {}
    for key in ("reasoning_content", "reasoning"):
        val = psf.get(key)
        if val:
            return str(val)
    return None


def check_reasoning_toggle(model: str) -> tuple[bool, str]:
    try:
        print("  effort=high: calling...")
        high_msg = _do_reasoning_call(model, "high")
        high_reasoning = _get_reasoning(high_msg)
        high_content = str(high_msg.get("content", "") or "")
        if high_reasoning:
            print(f"  effort=high: reasoning_content present ({len(high_reasoning)} chars)")
        else:
            print(f"  effort=high: no reasoning_content")
        print(f"  effort=high content preview: {high_content[:120]}")

        print("  effort=none: calling...")
        none_msg = _do_reasoning_call(model, "none")
        none_reasoning = _get_reasoning(none_msg)
        none_content = str(none_msg.get("content", "") or "")
        if none_reasoning:
            print(f"  effort=none: reasoning_content present ({len(none_reasoning)} chars)")
        else:
            print(f"  effort=none: no reasoning_content")
        print(f"  effort=none content preview: {none_content[:120]}")

        if high_reasoning and not none_reasoning:
            print(f"  assessment: reasoning toggle is working")
            print(f"  status: OK")
            return True, ""
        elif not high_reasoning and not none_reasoning:
            print(f"  assessment: model does not produce reasoning_content")
            print(f"  status: NOT SUPPORTED")
            return True, "no reasoning_content observed"
        elif high_reasoning and none_reasoning:
            print(f"  assessment: model always produces reasoning_content (toggle has no effect)")
            print(f"  status: ALWAYS ON")
            return True, "toggle has no effect"
        else:
            print(f"  assessment: toggle appears inverted (none has reasoning, high does not)")
            print(f"  status: INVERTED")
            return False, "toggle inverted"

    except Exception as e:
        msg = str(e)
        print(f"  status: FAILED")
        print(f"  error: {msg}")
        return False, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="LLM diagnostic checks for the configured endpoint.")
    parser.add_argument("--model", default=None, help="Model override (defaults to fast_llm_model from config).")
    parser.add_argument("--prompt", default=None, help="Custom prompt for the availability check.")
    parser.add_argument(
        "--check",
        choices=CHECKS,
        default="all",
        help="Which check to run (default: all).",
    )
    args = parser.parse_args()

    current_config = load_config()
    llm_utils.config = current_config

    model = args.model or current_config.fast_llm_model
    endpoint = current_config.llm_base_url.strip()
    api_key = current_config.llm_api_key.strip()

    if not endpoint:
        print("LLM diagnostic: LLM_BASE_URL is missing in .env")
        return 1

    print("LLM diagnostic checks")
    print(f"- model: {model}")
    print(f"- endpoint: {endpoint}")
    print(f"- api_key: {_mask_key(api_key)}")
    print(f"- llm_enable_thinking: {current_config.llm_enable_thinking}")
    print()

    run_check = args.check
    checks_to_run: list[tuple[str, str]] = []
    if run_check in ("all", "availability"):
        checks_to_run.append(("availability", "Availability"))
    if run_check in ("all", "tools"):
        checks_to_run.append(("tools", "Tool call support"))
    if run_check in ("all", "reasoning"):
        checks_to_run.append(("reasoning", "Reasoning toggle"))

    total = len(checks_to_run)
    results: list[tuple[str, bool, str]] = []

    for i, (check_id, check_label) in enumerate(checks_to_run, 1):
        print(f"[{i}/{total}] {check_label}")

        if check_id == "availability":
            ok, err = check_availability(model)
        elif check_id == "tools":
            ok, err = check_tool_call_support(model)
        elif check_id == "reasoning":
            ok, err = check_reasoning_toggle(model)
        else:
            ok, err = False, "unknown check"

        results.append((check_label, ok, err))
        print()

    passed = sum(1 for _, ok, _ in results if ok)
    failed_names = [name for name, ok, _ in results if not ok]

    print(f"Summary: {passed}/{total} checks passed")
    if failed_names:
        print(f"Failed: {', '.join(failed_names)}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
