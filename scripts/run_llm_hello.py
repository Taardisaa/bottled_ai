from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rs.utils.config import Config, load_config


def _mask_key(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _print_config(config: Config) -> None:
    print("Loaded LLM config:")
    print(f"- fast_llm_model: {config.fast_llm_model}")
    print(f"- llm_base_url: {config.llm_base_url or '<not set>'}")
    print(f"- llm_api_key: {_mask_key(config.llm_api_key)}")
    print(f"- openrouter_base_url: {config.openrouter_base_url}")


def main() -> int:
    try:
        from rs.utils import llm_utils
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}")
        print("Install dependencies first, for example: python -m pip install -r requirements.txt")
        return 2

    current_config = load_config()
    llm_utils.config = current_config
    _print_config(current_config)

    response, total_tokens = llm_utils.ask_llm_once(
        message="Hello!",
        model=current_config.fast_llm_model,
        struct=None,
        temperature=1.0,
        enable_cache=False,
    )

    print("\nLLM call result:")
    print(f"- total_tokens: {total_tokens}")
    if response is None:
        print("- response: <none>")
        return 1

    print("- response:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
