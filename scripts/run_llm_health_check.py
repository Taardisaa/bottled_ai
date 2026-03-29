from __future__ import annotations

import argparse
import json
import sys
from urllib import error, request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.utils.config import load_config


DEFAULT_PROMPT = r"Return the answer of 12*12 in \boxed{}."
DEFAULT_EXPECTED = r"\boxed{144}"


def _extract_content(response: object) -> str:
    if not isinstance(response, dict):
        return ""

    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    return "" if content is None else str(content).strip()


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct LLM health check against the configured endpoint.")
    parser.add_argument("--model", default=None, help="Model override. Defaults to fast_llm_model from config.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the LLM.")
    parser.add_argument(
        "--expected",
        default=DEFAULT_EXPECTED,
        help="Expected substring in the LLM response.",
    )
    args = parser.parse_args()

    current_config = load_config()
    model = args.model or current_config.fast_llm_model
    endpoint = current_config.llm_base_url.strip()
    api_key = current_config.llm_api_key.strip()

    if not endpoint:
        print("LLM health check")
        print("- status: failed")
        print("- error: LLM_BASE_URL is not configured")
        return 1

    print("LLM health check")
    print(f"- model: {model}")
    print(f"- endpoint: {endpoint}")
    print(f"- prompt: {args.prompt}")
    print(f"- expected: {args.expected}")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _chat_completions_url(endpoint),
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=30) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print("- status: failed")
        print(f"- error: HTTP {exc.code}: {error_body}")
        return 1
    except Exception as exc:
        print("- status: failed")
        print(f"- error: {exc}")
        return 1

    response_text = _extract_content(raw_response)
    success = args.expected in response_text

    print(f"- response: {response_text or '<empty>'}")
    print(f"- status: {'ok' if success else 'unexpected-response'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
