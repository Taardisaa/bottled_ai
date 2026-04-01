from __future__ import annotations

import argparse
import json

from rs.utils.config import load_config
from rs.utils import llm_utils


DEFAULT_PROMPT = "Reply with exactly: LLM test OK"


def _mask_key(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-shot LLM test call using .env credentials.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text for the test call.")
    args = parser.parse_args()

    current_config = load_config()
    llm_utils.config = current_config

    endpoint = current_config.llm_base_url.strip()
    api_key = current_config.llm_api_key.strip()
    model = args.model or current_config.fast_llm_model

    if not endpoint:
        print("LLM test failed: LLM_BASE_URL is missing in .env")
        return 1
    if not api_key:
        print("LLM test failed: LLM_API_KEY is missing in .env")
        return 1

    print("Running one-shot LLM test")
    print(f"- model: {model}")
    print(f"- llm_base_url: {endpoint}")
    print(f"- llm_api_key: {_mask_key(api_key)}")

    response, total_tokens = llm_utils.ask_llm_once(
        message=args.prompt,
        model=model,
        struct=None,
        temperature=0.6,
        enable_cache=False,
    )

    if response is None:
        print("LLM test failed: no response returned")
        return 1

    print(f"- total_tokens: {total_tokens}")
    if isinstance(response, (dict, list)):
        print("- response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(f"- response: {response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
