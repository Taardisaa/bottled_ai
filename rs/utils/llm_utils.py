from contextlib import redirect_stdout
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Optional, Any, Union, Tuple, List

from collections.abc import Sequence
import litellm
from litellm import (
    completion,
    batch_completion,
    decode as litellm_decode,
    encode as litellm_encode,
    get_max_tokens as litellm_get_max_tokens,
    token_counter as litellm_token_counter,
)

# Avoid repeated "Provider List" / help banners when LiteLLM cannot infer a provider.
litellm.suppress_debug_info = True
from langchain.tools import BaseTool
from pydantic import BaseModel
from typing import Dict, Type
import tiktoken

from rs.utils.config import config
from rs.utils.common_utils import set_env

from loguru import logger

# Accept either a Pydantic BaseModel subclass or a TypedDict / dict class for structured output.
LLMAcceptStructParam = Union[Type[BaseModel], Dict] 

# Possible types when storing LLM responses to cache. Can be instances of BaseModel or dict.
LLMStoredResponse = Union[BaseModel, Dict]

# Possible types when loading LLM responses from cache. Same as stored types.
LLMLoadedResponse = Union[BaseModel, Dict]


# Model token limits (input context window)
MODEL_TOKEN_LIMITS = {
    "gpt-5": 272000,
    "gpt-5-mini": 272000,
    # Local MLX deployment configuration.
    "qwen-mlx": 125600,
}


_TOKENIZER_PROVIDER_PREFIXES = {
    "openai",
    "openrouter",
    "anthropic",
    "azure",
    "hosted_vllm",
    "vertex_ai",
    "bedrock",
    "ollama",
}

SECOND_LAYER_THINKING_TOKEN_LIMIT = 1024
SECOND_LAYER_STRUCT_CONVERT_MAX_RETRIES = 3
SECOND_LAYER_FIRST_PASS_RESTARTS = 1
_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LLMPreflightResult:
    available: bool
    requested_model: str
    routed_model: str
    normalized_model: str
    provider: str
    endpoint: str
    max_tokens: int
    response_model: str = ""
    response_preview: str = ""
    total_tokens: int = 0
    error: str = ""


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("openrouter/")


def _has_custom_base_url() -> bool:
    return bool(config.llm_base_url.strip())


def _normalize_model_for_litellm(model: str) -> str:
    normalized = str(model).strip()
    if not _has_custom_base_url():
        return normalized

    if _is_openrouter_model(normalized):
        return normalized
    return normalized


def _finalize_model_before_request(model: str) -> str:
    normalized = str(model).strip()
    if not _has_custom_base_url():
        return normalized
    if _is_openrouter_model(normalized):
        return normalized
    if normalized.startswith("openai/"):
        return normalized.split("/", 1)[1]
    return normalized


def _model_has_litellm_provider_prefix(model: str) -> bool:
    m = str(model).strip()
    return any(m.startswith(f"{p}/") for p in _TOKENIZER_PROVIDER_PREFIXES)


def _litellm_routed_model_id(request_model: str) -> str:
    """Model id passed to litellm.completion (adds openai/ for custom endpoints when needed)."""
    finalized = _finalize_model_before_request(request_model)
    routed = _normalize_model_for_litellm(finalized)
    if (
            _has_custom_base_url()
            and not _is_openrouter_model(request_model)
            and not _model_has_litellm_provider_prefix(routed)
    ):
        return f"openai/{routed}"
    return routed


def _provider_name_for_model(model: str) -> str:
    if _has_custom_base_url():
        return "custom-openai-compatible"
    if _is_openrouter_model(model):
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    return "openai-compatible"


def _endpoint_for_model(model: str) -> str:
    if _has_custom_base_url():
        return config.llm_base_url
    if _is_openrouter_model(model):
        return config.openrouter_base_url
    if config.openai_base_url and not model.startswith("claude"):
        return config.openai_base_url
    return "provider-default"


def _safe_preview(value: Any, limit: int = 120) -> str:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        preview = json.dumps(value, ensure_ascii=False)
    else:
        preview = str(value)

    preview = " ".join(preview.split())
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def _llm_output_preview(value: Any, limit: int = 2000) -> str:
    if value is None:
        return "None"
    if isinstance(value, BaseModel):
        text = json.dumps(value.model_dump(), ensure_ascii=False)
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\r", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _normalize_model_for_tokenizer(model: str) -> str:
    normalized = str(model).strip()
    if normalized == "":
        return normalized

    parts = normalized.split("/")
    while len(parts) > 1 and parts[0].strip().lower() in _TOKENIZER_PROVIDER_PREFIXES:
        parts = parts[1:]

    return "/".join(part.strip() for part in parts if part.strip())


def _tokenizer_model_candidates(model: str) -> list[str]:
    normalized = _normalize_model_for_tokenizer(model)
    candidates: list[str] = []
    for candidate in [normalized, normalized.lower()]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if "/" in normalized:
        tail = normalized.rsplit("/", 1)[-1].strip()
        for candidate in [tail, tail.lower()]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _fallback_encoding_name_for_model(model: str) -> str:
    normalized = _normalize_model_for_tokenizer(model).lower()
    modern_families = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "qwen",
        "deepseek",
        "llama",
        "claude",
        "gemini",
        "mistral",
    )
    if any(normalized.startswith(prefix) or f"/{prefix}" in normalized for prefix in modern_families):
        return "o200k_base"
    return "cl100k_base"


def _get_tiktoken_encoding(model: str, log_fallback: bool = True):
    for candidate in _tokenizer_model_candidates(model):
        try:
            return tiktoken.encoding_for_model(candidate)
        except KeyError:
            continue

    fallback_encoding_name = _fallback_encoding_name_for_model(model)
    try:
        encoding = tiktoken.get_encoding(fallback_encoding_name)
    except KeyError:
        fallback_encoding_name = "cl100k_base"
        encoding = tiktoken.get_encoding(fallback_encoding_name)

    if log_fallback:
        normalized = _normalize_model_for_tokenizer(model)
        if normalized != str(model).strip():
            logger.debug(
                f"Model {model} normalized to {normalized} for token counting; "
                f"using fallback {fallback_encoding_name} encoding"
            )
        else:
            logger.warning(
                f"Model {model} not found in tiktoken, using {fallback_encoding_name} encoding"
            )
    return encoding


def _normalized_model_token_limit_key(model: str) -> str:
    for candidate in _tokenizer_model_candidates(model):
        lowered = candidate.lower()
        if lowered in MODEL_TOKEN_LIMITS:
            return lowered
    return _normalize_model_for_tokenizer(model).lower()


def _get_litellm_token_count(text: str, model: str) -> int:
    normalized_model = _normalize_model_for_tokenizer(model)
    token_count = _call_litellm_quietly(litellm_token_counter, model=normalized_model, text=text)
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError(f"LiteLLM token counter returned invalid value: {token_count!r}")
    return token_count


def _log_tokenizer_fallback(model: str, source: str, error: Exception) -> None:
    normalized_key = _normalized_model_token_limit_key(model)
    if source == "LiteLLM max-token lookup" and normalized_key in MODEL_TOKEN_LIMITS:
        return

    normalized = _normalize_model_for_tokenizer(model)
    message = f"{source} tokenizer lookup failed for {model}: {error}"
    if normalized != str(model).strip():
        logger.debug(message)
    else:
        logger.debug(message)


def _encode_text_with_model_tokenizer(text: str, model: str) -> tuple[list[int], str]:
    normalized_model = _normalize_model_for_tokenizer(model)
    try:
        token_ids = _call_litellm_quietly(litellm_encode, model=normalized_model, text=text)
        if isinstance(token_ids, list):
            return token_ids, normalized_model
        raise ValueError(f"LiteLLM encode returned invalid value: {token_ids!r}")
    except Exception as e:
        _log_tokenizer_fallback(model, "LiteLLM", e)

    encoding = _get_tiktoken_encoding(model, log_fallback=False)
    return encoding.encode(text), "__tiktoken__"


def _decode_tokens_with_model_tokenizer(token_ids: list[int], model: str, codec_id: str) -> str:
    if codec_id != "__tiktoken__":
        return _call_litellm_quietly(litellm_decode, model=codec_id, tokens=token_ids)

    encoding = _get_tiktoken_encoding(model, log_fallback=False)
    return encoding.decode(token_ids)


def _litellm_completion_kwargs(model: str, temperature: float, **extra_kwargs: Any) -> Dict[str, Any]:
    enable_thinking = extra_kwargs.pop("enable_thinking", None)
    kwargs: Dict[str, Any] = {
        "model": _litellm_routed_model_id(model),
        "temperature": temperature,
    }
    if _has_custom_base_url():
        kwargs["api_base"] = config.llm_base_url
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key
    elif _is_openrouter_model(model):
        kwargs["api_base"] = config.openrouter_base_url
        kwargs["api_key"] = config.openrouter_key or config.openai_key
    elif config.openai_base_url and not model.startswith("claude"):
        kwargs["api_base"] = config.openai_base_url
        kwargs["api_key"] = config.openai_key
    for key, value in extra_kwargs.items():
        if value is not None:
            kwargs[key] = value

    if _should_send_qwen_thinking_toggle(model):
        effective_enable_thinking = config.llm_enable_thinking if enable_thinking is None else bool(enable_thinking)
        kwargs["extra_body"] = _merge_extra_body(
            kwargs.get("extra_body"),
            {"chat_template_kwargs": {"enable_thinking": effective_enable_thinking}},
        )
    elif _has_custom_base_url() and not _is_qwen3_model(model):
        effective = config.llm_enable_thinking if enable_thinking is None else bool(enable_thinking)
        kwargs["extra_body"] = _merge_extra_body(
            kwargs.get("extra_body"),
            {"reasoning_effort": "high" if effective else "none"},
        )
    return kwargs


def _is_qwen3_model(model: str) -> bool:
    normalized = _normalize_model_for_tokenizer(model).lower()
    return "qwen3" in normalized or "qwen/qwen3" in normalized


def _should_send_qwen_thinking_toggle(model: str) -> bool:
    normalized = _normalize_model_for_tokenizer(model).lower()
    return _has_custom_base_url() and "qwen" in normalized and not _is_qwen3_model(model)


def _apply_qwen3_thinking_token(content: str, model: str, enable_thinking: bool) -> str:
    """For Qwen3 models, append /think or /no_think token to the message content."""
    if not _is_qwen3_model(model):
        return content
    token = "/think" if enable_thinking else "/no_think"
    return f"{content} {token}"


def _merge_extra_body(existing: Any, additions: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    for key, value in additions.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _call_litellm_quietly(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Discard Python-level stdout from LiteLLM and related libraries during calls."""
    with open(os.devnull, "w", encoding="utf-8") as dev_null, redirect_stdout(dev_null):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if function is completion and _has_custom_base_url():
                retried_kwargs = _fallback_completion_kwargs_for_model_error(kwargs, error)
                if retried_kwargs is not None:
                    return function(*args, **retried_kwargs)
            raise


def _fallback_completion_kwargs_for_model_error(
        kwargs: Dict[str, Any],
        error: Exception,
) -> Dict[str, Any] | None:
    model = str(kwargs.get("model", "")).strip()
    if model == "":
        return None
    message = str(error).lower()

    # Some proxies validate model names before provider stripping and only accept
    # bare ids (e.g. qwen-mlx), so retry once without openai/ prefix.
    if "invalid model name passed in model=openai/" in message and model.startswith("openai/"):
        retry_kwargs = dict(kwargs)
        retry_kwargs["model"] = model.split("/", 1)[1]
        logger.warning(f"Retrying completion with raw model id: {retry_kwargs['model']}")
        return retry_kwargs

    return None


# TODO: Current impl only supports OpenAI and Anthropic models. Add more providers as needed.
def _ensure_api_key_for_model(model: str) -> bool:
    """Ensure the appropriate API key is set for the given model.

    Returns:
        True if the API key is available, False otherwise.
    """
    if _has_custom_base_url():
        set_env("OPENAI_BASE_URL", config.llm_base_url)
        set_env("OPENAI_API_BASE", config.llm_base_url)
        if config.llm_api_key:
            set_env("OPENAI_API_KEY", config.llm_api_key)
        return True

    # Determine provider from model name
    if _is_openrouter_model(model):
        config_key = config.openrouter_key or config.openai_key
        if not config_key:
            logger.error(f"OpenRouter key for model '{model}' is not set. Check OPENROUTER_API_KEY in your .env.")
            return False

        # Keep OpenAI-compatible env vars for downstream compatibility.
        set_env("OPENAI_API_KEY", config_key)
        set_env("OPENAI_BASE_URL", config.openrouter_base_url)
        set_env("OPENAI_API_BASE", config.openrouter_base_url)
        return True

    if model.startswith("claude"):
        env_var = "ANTHROPIC_API_KEY"
        config_key = config.anthropic_key
    else:
        # Default to OpenAI for gpt-*, o1-*, etc.
        env_var = "OPENAI_API_KEY"
        config_key = config.openai_key

    # Ensure OpenAI-compatible base URL overrides from OpenRouter do not leak
    # into non-OpenRouter calls in the same process.
    # Preserve env vars when a custom OpenAI base URL is configured.
    if not _is_openrouter_model(model) and not config.openai_base_url:
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)

    if not config_key:
        logger.error(f"API key for model '{model}' is not set. Check your .env file.")
        return False

    # Always sync runtime env var to the selected provider key.
    # This prevents stale keys from earlier calls (e.g. openrouter -> openai)
    # from leaking across requests in the same process.
    set_env(env_var, config_key)

    return True


# TODO: Expand to a more generic approach.
def get_model_token_limit(model: str) -> int:
    """Get the token limit for a specific model.
    
    Args:
        model (str): The model name.
        
    Returns:
        int: The token limit for the model. Defaults to 8192 if unknown.
    """
    normalized_model = _normalize_model_for_tokenizer(model)
    try:
        max_tokens = _call_litellm_quietly(litellm_get_max_tokens, normalized_model)
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    except Exception as e:
        _log_tokenizer_fallback(model, "LiteLLM max-token lookup", e)

    normalized_key = _normalized_model_token_limit_key(model)
    if normalized_key in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[normalized_key]

    if normalized_key.startswith("gpt-5"):
        return MODEL_TOKEN_LIMITS["gpt-5"]

    return 125600


def count_tokens(text: str, model) -> int:
    """Count the number of tokens in a text string.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The model to use for tokenization. Defaults to "gpt-4".
        
    Returns:
        int: The number of tokens in the text.
    """
    litellm_error: Exception | None = None
    try:
        return _get_litellm_token_count(text=text, model=str(model))
    except Exception as e:
        litellm_error = e
        _log_tokenizer_fallback(str(model), "LiteLLM token counting", litellm_error)

    try:
        encoding = _get_tiktoken_encoding(str(model), log_fallback=False)
        return len(encoding.encode(text))
    except Exception as tiktoken_error:
        logger.warning(
            f"Failed to count tokens for model {model} with LiteLLM and tiktoken; "
            f"using character-count estimate. LiteLLM error: {litellm_error}; "
            f"tiktoken error: {tiktoken_error}"
        )
        # Rough estimation: ~4 characters per token
        return len(text) // 4


def truncate_message(message: str, 
                     model: str = config.fast_llm_model, 
                     max_tokens: Optional[int] = None, 
                     reserve_tokens: int = 2000) -> Tuple[str, int]:
    """Truncate a message to fit within the model's token limit.
    
    Args:
        message (str): The message to truncate.
        model (str): The model name. Defaults to the fast LLM model from config.
        max_tokens (Optional[int]): Maximum tokens allowed. If None, uses model's limit minus reserve.
        reserve_tokens (int): Number of tokens to reserve for response. Defaults to 2000.
        
    Returns:
        str: The truncated message.
        int: The number of **remaining** tokens for LLM output after truncation.
    """
    if max_tokens is None:
        max_tokens = get_model_token_limit(model) - reserve_tokens
    
    current_tokens = count_tokens(message, model)
    
    if current_tokens <= max_tokens:
        return message, max_tokens - current_tokens
    
    logger.warning(f"Message has {current_tokens} tokens, exceeding limit of {max_tokens}. Truncating...")

    try:
        tokens, codec_id = _encode_text_with_model_tokenizer(message, model)
        truncated_tokens = tokens[:max_tokens]
        truncated_message = _decode_tokens_with_model_tokenizer(truncated_tokens, model, codec_id)
    except Exception as e:
        logger.debug(f"Tokenizer-based truncation failed for model {model}: {e}. Falling back to binary search.")
        low = 0
        high = len(message)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = message[:mid]
            candidate_tokens = count_tokens(candidate, model)
            if candidate_tokens <= max_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        truncated_message = best
        truncated_tokens = None

    final_token_count = count_tokens(truncated_message, model)
    logger.info(f"Message truncated from {current_tokens} to {final_token_count} tokens")
    return truncated_message, max_tokens - final_token_count


def _is_llm_load_cacheable(struct: Optional[Any] = None) -> bool:
    """Check if an LLM query can load from cache based on config and struct type."""
    if not config.get_load_cache_option("llm_query"):
        return False

    if struct is not None:
        is_dict_type = (struct is dict or
                       (hasattr(struct, '__origin__') and struct.__origin__ is dict) or
                       (hasattr(struct, '__dict__') and hasattr(struct, '__annotations__')))
        has_json_methods = hasattr(struct, 'to_json') and hasattr(struct, 'from_json')
        if not (is_dict_type or has_json_methods):
            logger.error("Struct provided does not support caching. Must be dict, TypedDict, or have to_json/from_json methods.")
            return False

    return True


def _is_llm_store_cacheable(struct: Optional[Any] = None) -> bool:
    """Check if an LLM query can store to cache based on config and struct type."""
    if not config.get_store_cache_option("llm_query"):
        return False

    if struct is not None:
        is_dict_type = (struct is dict or
                       (hasattr(struct, '__origin__') and struct.__origin__ is dict) or
                       (hasattr(struct, '__dict__') and hasattr(struct, '__annotations__')))
        has_json_methods = hasattr(struct, 'to_json') and hasattr(struct, 'from_json')
        if not (is_dict_type or has_json_methods):
            logger.error("Struct provided does not support caching. Must be dict, TypedDict, or have to_json/from_json methods.")
            return False

    return True


# Backward compatibility alias
_is_llm_cacheable = _is_llm_load_cacheable


def _reconstruct_llm_response(cached: Any, struct: Optional[LLMAcceptStructParam] = None) -> Optional[LLMLoadedResponse]:
    """Reconstruct the appropriate type from a cached LLM response."""
    if cached is None:
        return None

    # Already the right type (Pydantic model)
    if isinstance(cached, BaseModel):
        return cached

    # Dict from cache — reconstruct if struct is a Pydantic model
    if isinstance(cached, dict) and struct is not None:
        if isinstance(struct, type) and issubclass(struct, BaseModel):
            try:
                return struct.model_validate(cached)
            except Exception:
                return cached

    return cached if isinstance(cached, dict) else None


def _extract_litellm_content(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, dict):
        return None

    choices = response.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return None

    if "parsed" in message:
        return message.get("parsed")
    return message.get("content")


def run_llm_preflight_check(
        model: Optional[str] = None,
        probe_message: str = "Reply with exactly: OK") -> LLMPreflightResult:
    requested_model = model or config.fast_llm_model
    routed_model = _litellm_routed_model_id(requested_model)
    normalized_model = _normalize_model_for_tokenizer(requested_model)
    provider = _provider_name_for_model(requested_model)
    endpoint = _endpoint_for_model(requested_model)
    max_tokens = get_model_token_limit(requested_model)

    if not _ensure_api_key_for_model(requested_model):
        return LLMPreflightResult(
            available=False,
            requested_model=requested_model,
            routed_model=routed_model,
            normalized_model=normalized_model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=max_tokens,
            error="Missing or invalid API configuration for selected model.",
        )

    try:
        completion_kwargs = _litellm_completion_kwargs(
            requested_model,
            _normalize_temperature_for_model(requested_model, 0.0),
            max_tokens=16,
        )
        raw_response = _call_litellm_quietly(
            completion,
            messages=[{"role": "user", "content": probe_message}],
            **completion_kwargs,
        )
        if hasattr(raw_response, "model_dump"):
            response_dict = raw_response.model_dump()
        elif isinstance(raw_response, dict):
            response_dict = raw_response
        else:
            raise TypeError(f"Unexpected LiteLLM response type: {type(raw_response)!r}")

        usage = response_dict.get("usage", {})
        content = _extract_litellm_content(response_dict)
        response_model = str(response_dict.get("model", "") or routed_model)

        return LLMPreflightResult(
            available=True,
            requested_model=requested_model,
            routed_model=routed_model,
            normalized_model=normalized_model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=max_tokens,
            response_model=response_model,
            response_preview=_safe_preview(content),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
    except Exception as e:
        return LLMPreflightResult(
            available=False,
            requested_model=requested_model,
            routed_model=routed_model,
            normalized_model=normalized_model,
            provider=provider,
            endpoint=endpoint,
            max_tokens=max_tokens,
            error=str(e),
        )


def _coerce_structured_response(content: Any, struct: LLMAcceptStructParam) -> Optional[LLMLoadedResponse]:
    if isinstance(struct, type) and issubclass(struct, BaseModel):
        if isinstance(content, struct):
            return content

        parsed_payload = content
        if isinstance(content, str):
            try:
                parsed_payload = json.loads(content)
            except Exception:
                return None

        if not isinstance(parsed_payload, dict):
            return None

        # Reject dicts whose keys have zero overlap with the schema properties.
        # Without this check, a dict with completely unrelated keys (e.g. the
        # LLM using "tool_call" instead of "tool_name") silently passes
        # validation when all schema fields have defaults, producing an
        # all-defaults object that discards the LLM's actual intent.
        if parsed_payload:
            schema_props = struct.model_json_schema().get("properties", {})
            if schema_props and not (set(parsed_payload.keys()) & set(schema_props.keys())):
                return None

        try:
            return struct.model_validate(parsed_payload)
        except Exception:
            return None

    # dict / TypedDict-like output
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            return None
    return None


def _extract_json_object_from_text(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:idx + 1]
                try:
                    loaded = json.loads(candidate)
                except Exception:
                    return None
                return loaded if isinstance(loaded, dict) else None
    return None


def _coerce_structured_from_field_lines(
        text: str,
        struct: LLMAcceptStructParam,
) -> Optional[LLMLoadedResponse]:
    if not isinstance(text, str):
        return None
    if not (isinstance(struct, type) and issubclass(struct, BaseModel)):
        return None

    properties = struct.model_json_schema().get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return None

    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        field_name, raw_value = line.split(":", 1)
        field_name = field_name.strip()
        if field_name not in properties:
            continue
        value_text = raw_value.strip()
        if value_text == "":
            payload[field_name] = ""
            continue
        lowered = value_text.lower()
        if lowered == "null":
            payload[field_name] = None
            continue
        if lowered == "true":
            payload[field_name] = True
            continue
        if lowered == "false":
            payload[field_name] = False
            continue
        try:
            payload[field_name] = json.loads(value_text)
            continue
        except Exception:
            payload[field_name] = value_text

    if not payload:
        return None
    try:
        return struct.model_validate(payload)
    except Exception:
        return None


def _try_coerce_structured_from_text(
        text: Any,
        struct: LLMAcceptStructParam,
) -> Optional[LLMLoadedResponse]:
    if text is None:
        return None
    if not isinstance(text, str):
        return _coerce_structured_response(text, struct)

    direct = _coerce_structured_response(text, struct)
    if direct is not None:
        return direct

    embedded_json = _extract_json_object_from_text(text)
    if embedded_json is not None:
        coerced = _coerce_structured_response(embedded_json, struct)
        if coerced is not None:
            return coerced

    return _coerce_structured_from_field_lines(text, struct)


def _parse_litellm_response(response: Any,
                           struct: Optional[LLMAcceptStructParam]) -> Tuple[Optional[LLMLoadedResponse], int]:
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif isinstance(response, dict):
        response_dict = response
    else:
        return None, 0

    usage = response_dict.get("usage", {})
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    if struct is None:
        return response_dict, total_tokens

    content = _extract_litellm_content(response_dict)
    coerced = _coerce_structured_response(content, struct)
    return coerced, total_tokens


def _normalize_temperature_for_model(model: str, temperature: float) -> float:
    if model in ['gpt-5', 'gpt-5-mini'] and temperature != 1.0:
        logger.warning(f"Model {model} does not support temperature other than 1.0. Setting temperature to 1.0.")
        return 1.0
    return temperature


def _build_struct_convert_prompt(raw_response_text: str, struct: LLMAcceptStructParam) -> str:
    if isinstance(struct, type) and issubclass(struct, BaseModel):
        schema = json.dumps(struct.model_json_schema(), sort_keys=True)
        properties = struct.model_json_schema().get("properties", {})
        field_lines: list[str] = []
        example_payload: dict[str, Any] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            field_type = field_schema.get("type")
            if field_type is None and isinstance(field_schema.get("anyOf"), list):
                any_of_types = [
                    option.get("type")
                    for option in field_schema["anyOf"]
                    if isinstance(option, dict) and option.get("type")
                ]
                field_type = " or ".join(str(value) for value in any_of_types)
            field_lines.append(f"- {field_name}: {field_type or 'value'}")
            if "default" in field_schema:
                example_payload[field_name] = field_schema["default"]
            elif field_type == "string":
                example_payload[field_name] = ""
            elif field_type == "number":
                example_payload[field_name] = 0
            elif field_type == "integer":
                example_payload[field_name] = 0
            elif field_type == "boolean":
                example_payload[field_name] = False
            else:
                example_payload[field_name] = None

        fields_text = "\n".join(field_lines) if field_lines else "- follow the schema exactly"
        example_text = json.dumps(example_payload, ensure_ascii=False, sort_keys=True)
        return (
            "Convert the source model output below into a final JSON decision object.\n"
            "Output requirements:\n"
            f"{fields_text}\n"
            "Rules:\n"
            "- Return exactly one JSON object.\n"
            "- Do not include markdown fences or extra commentary.\n"
            "- Do not add extra keys.\n"
            "- If the source is uncertain or non-committal, use null where appropriate.\n"
            f"Example JSON shape:\n{example_text}\n"
            f"JSON Schema:\n{schema}\n"
            "Source model output:\n"
            f"{raw_response_text}"
        )

    return (
        "Convert the source model output below into a valid JSON object.\n"
        "Rules:\n"
        "- Return exactly one JSON object.\n"
        "- Do not include markdown fences or extra commentary.\n"
        "Source model output:\n"
        f"{raw_response_text}"
    )


def _truncate_think_blocks_for_second_layer(
        raw_response_text: str,
        model: str,
        max_think_tokens: int = SECOND_LAYER_THINKING_TOKEN_LIMIT,
) -> str:
    def _replace(match: re.Match[str]) -> str:
        think_content = match.group(1)
        think_tokens = count_tokens(think_content, model)
        if think_tokens <= max_think_tokens:
            return match.group(0)

        truncated_content, _ = truncate_message(
            think_content,
            model=model,
            max_tokens=max_think_tokens,
            reserve_tokens=0,
        )
        logger.info(
            f"Second-layer conversion truncated <think> block from {think_tokens} to "
            f"{count_tokens(truncated_content, model)} tokens"
        )
        return f"<think>{truncated_content}\n...[truncated thinking]...\n</think>"

    return _THINK_BLOCK_PATTERN.sub(_replace, raw_response_text)


def _ask_llm_once_two_layer(
        message: str,
        model: str,
        struct: LLMAcceptStructParam,
        temperature: float,
) -> Tuple[Optional[LLMLoadedResponse], int]:
    total_tokens = 0
    first_temperature = _normalize_temperature_for_model(model, temperature)
    first_kwargs = _litellm_completion_kwargs(
        model,
        first_temperature,
        enable_thinking=config.llm_enable_thinking,
    )
    conversion_model = config.fast_llm_model
    if not _ensure_api_key_for_model(conversion_model):
        return None, 0

    second_temperature = _normalize_temperature_for_model(conversion_model, 0.0)
    second_kwargs = _litellm_completion_kwargs(
        conversion_model,
        second_temperature,
        enable_thinking=False,
        response_format=struct,
    )

    for first_pass_attempt in range(SECOND_LAYER_FIRST_PASS_RESTARTS + 1):
        first_request_content = _apply_qwen3_thinking_token(message, model, config.llm_enable_thinking)
        first_raw = _call_litellm_quietly(
            completion,
            messages=[{"role": "user", "content": first_request_content}],
            **first_kwargs,
        )

        if hasattr(first_raw, "model_dump"):
            first_response_dict = getattr(first_raw, "model_dump")()
        else:
            first_response_dict = first_raw
        if not isinstance(first_response_dict, dict):
            logger.error("Unexpected LiteLLM first-stage response type")
            return None, total_tokens

        first_tokens = int(first_response_dict.get("usage", {}).get("total_tokens", 0) or 0)
        total_tokens += first_tokens
        first_content = _extract_litellm_content(first_response_dict)
        if first_content is None:
            logger.error("Unexpected LiteLLM response content for first-stage unstructured call")
            if first_pass_attempt < SECOND_LAYER_FIRST_PASS_RESTARTS:
                logger.warning(
                    f"Retrying first-stage generation ({first_pass_attempt + 2}/{SECOND_LAYER_FIRST_PASS_RESTARTS + 1}) "
                    "due to empty first-stage content"
                )
                continue
            return None, total_tokens

        if isinstance(first_content, (dict, list)):
            first_content_text = json.dumps(first_content, ensure_ascii=False)
        else:
            first_content_text = str(first_content)
        logger.info(
            "LLM first-pass output preview: "
            + _llm_output_preview(first_content_text)
        )

        # Fast-path: if first-pass output already contains parseable structured data
        # (JSON or key-value fields), skip second-stage conversion entirely.
        if "<think>" not in first_content_text:
            first_direct = _try_coerce_structured_from_text(first_content_text, struct)
            if first_direct is not None:
                logger.info("Two-layer fast-path accepted first-pass structured output")
                return first_direct, total_tokens

        first_content_text = _truncate_think_blocks_for_second_layer(first_content_text, model)
        second_prompt = _build_struct_convert_prompt(first_content_text, struct)

        for attempt in range(SECOND_LAYER_STRUCT_CONVERT_MAX_RETRIES):
            second_request_content = _apply_qwen3_thinking_token(second_prompt, conversion_model, False)
            second_raw = _call_litellm_quietly(
                completion,
                messages=[{"role": "user", "content": second_request_content}],
                **second_kwargs,
            )
            second_parsed, second_tokens = _parse_litellm_response(second_raw, struct)
            total_tokens += second_tokens
            if second_parsed is not None:
                return second_parsed, total_tokens
            second_content = _extract_litellm_content(second_raw)
            second_fallback = _try_coerce_structured_from_text(second_content, struct)
            if second_fallback is not None:
                logger.info("Two-layer fallback recovered structured output from second-pass text")
                return second_fallback, total_tokens
            logger.warning(
                "Second-stage conversion parse failed "
                f"(attempt {attempt + 1}/{SECOND_LAYER_STRUCT_CONVERT_MAX_RETRIES})"
            )

        if first_pass_attempt < SECOND_LAYER_FIRST_PASS_RESTARTS:
            logger.warning(
                "Second-stage conversion exhausted; rerunning first-stage generation "
                f"({first_pass_attempt + 2}/{SECOND_LAYER_FIRST_PASS_RESTARTS + 1})"
            )

    logger.error("Unexpected LiteLLM response type or failed structured parse in second-stage conversion")
    return None, total_tokens


def ask_llm_once(message: str,
                 model: str = "gpt-5-mini",
                 struct: Optional[LLMAcceptStructParam] = None,
                 temperature: float = 0.0,
                 cache_namespace: Optional[str] = None,
                 enable_cache: bool = True,
                 two_layer_struct_convert: Optional[bool] = None) -> Tuple[Optional[LLMLoadedResponse], int]:
    """Ask a question to the LLM and get the response.

    Args:
        message (str): The message to ask.
        model (str, optional): The model to use. Defaults to "gpt-5-mini".
        struct (Optional[LLMAcceptStructParam], optional): subclass of BaseModel or TypedDict class. Defaults to None.
        temperature (float, optional): The temperature for the response. Defaults to 0.0.
        cache_namespace (Optional[str], optional): When provided, cache entries are isolated
            under this namespace (stored in a subdirectory). Used by the checker agent to
            isolate final patch checking results per config profile. Defaults to None.
        enable_cache (bool, optional): Additional gate for caching. When False, bypasses cache
            entirely (no load or store). AND-ed with other cache checks. Defaults to True.
        two_layer_struct_convert (Optional[bool], optional): When True and `struct` is provided,
            process each message through `ask_llm_once` with two-layer conversion.
            When None, uses config.llm_two_layer_struct_convert. Defaults to None.

    Returns:
        A tuple of (response from the LLM, total tokens used).
            Returns (None, 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return None, 0

        temperature = _normalize_temperature_for_model(model, temperature)
        use_two_layer_struct_convert = (
            config.llm_two_layer_struct_convert
            if two_layer_struct_convert is None
            else bool(two_layer_struct_convert)
        )

        # Truncate message if it exceeds token limit
        original_message = message
        message, remain_token = truncate_message(message, model=model)

        # Try to get cached response (use original message for hash to avoid cache misses)
        use_cache_load = _is_llm_load_cacheable(struct) and enable_cache
        if use_cache_load:
            ensure_cache_registry()
            cached_response = _reconstruct_llm_response(
                CacheRegistry.get("llm_query").load(
                    original_message, model, temperature, struct,
                    cache_namespace=cache_namespace),
                struct
            )
            if cached_response is not None:
                return cached_response, 0  # Cached responses don't cost tokens

        _t0_llm = time.perf_counter()
        if use_two_layer_struct_convert and struct is not None:
            response_to_return, total_tokens = _ask_llm_once_two_layer(
                message=message,
                model=model,
                struct=struct,
                temperature=temperature,
            )
        else:
            response_format = struct if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel) else None
            completion_kwargs = _litellm_completion_kwargs(
                model,
                temperature,
                response_format=response_format,
                enable_thinking=config.llm_enable_thinking,
            )

            request_content = _apply_qwen3_thinking_token(message, model, config.llm_enable_thinking)
            raw_response = _call_litellm_quietly(
                completion,
                messages=[{"role": "user", "content": request_content}],
                **completion_kwargs,
            )
            response_to_return, total_tokens = _parse_litellm_response(raw_response, struct)
        _llm_elapsed_ms = (time.perf_counter() - _t0_llm) * 1000
        logger.info(f"[TIMING] ask_llm_once model={model} ns={cache_namespace} took {_llm_elapsed_ms:.0f}ms tokens={total_tokens}")
        logger.info(f"LLM tokens used: {total_tokens}.")
        if response_to_return is None:
            logger.error("Unexpected LiteLLM response type or failed structured parse")
            return None, total_tokens
        logger.info(
            "LLM output preview: "
            + _llm_output_preview(response_to_return)
        )

        # Store response in cache if storing is enabled
        use_cache_store = _is_llm_store_cacheable(struct) and enable_cache
        if use_cache_store:
            if not CacheRegistry.get("llm_query").store(
                    response_to_return, original_message, model, temperature, struct,
                    cache_namespace=cache_namespace):
                logger.error("Failed to store LLM response in cache.")

        return response_to_return, total_tokens
    except Exception as e:
        logger.error(f"Failed to get response from LLM: {e} at {e.__traceback__.tb_lineno if e.__traceback__ else 'unknown line'}")
        return None, 0


def ask_llm_multi(messages: Sequence[Optional[str]], model: str = "gpt-5-mini",
                  struct: Optional[LLMAcceptStructParam] = None,
                  temperature: float = 0.0,
                  cache_namespace: Optional[str] = None,
                  enable_cache: bool = True,
                  two_layer_struct_convert: Optional[bool] = None) -> Tuple[List[Optional[LLMLoadedResponse]], int]:
    """Ask multiple questions to the LLM and get responses in batch.

    Args:
        messages (list[Optional[str]]): List of messages to ask.
        model (str, optional): The model to use. Defaults to "gpt-5-mini".
        struct (Optional[LLMAcceptStructParam], optional): subclass of BaseModel or TypedDict class. Defaults to None.
        temperature (float, optional): The temperature for the response. Defaults to 0.0.
        cache_namespace (Optional[str], optional): When provided, cache entries are isolated
            under this namespace (stored in a subdirectory). Defaults to None.
        enable_cache (bool, optional): Additional gate for caching. When False, bypasses cache
            entirely (no load or store). AND-ed with other cache checks. Defaults to True.
        two_layer_struct_convert (Optional[bool], optional): When True and `struct` is provided,
            first call the target model without structured output and then run a
            second conversion pass into the requested schema. When None, uses
            config.llm_two_layer_struct_convert. Defaults to None.

    Returns:
        A tuple of (list of responses from the LLM in the same order as input messages, total tokens used).
            Returns ([None] * len(messages), 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return [None] * len(messages), 0

        temperature = _normalize_temperature_for_model(model, temperature)
        use_two_layer_struct_convert = (
            config.llm_two_layer_struct_convert
            if two_layer_struct_convert is None
            else bool(two_layer_struct_convert)
        )

        if use_two_layer_struct_convert and struct is not None:
            responses: List[Optional[LLMLoadedResponse]] = [None] * len(messages)
            total_tokens = 0
            for i, message in enumerate(messages):
                if message is None:
                    continue
                response, token_count = ask_llm_once(
                    message=message,
                    model=model,
                    struct=struct,
                    temperature=temperature,
                    cache_namespace=cache_namespace,
                    enable_cache=enable_cache,
                    two_layer_struct_convert=True,
                )
                responses[i] = response
                total_tokens += token_count
            return responses, total_tokens

        responses: List[Optional[LLMLoadedResponse]] = [None] * len(messages)
        uncached_indices: List[int] = []
        uncached_messages: List[str] = []
        use_cache_load = _is_llm_load_cacheable(struct) and enable_cache

        for i, message in enumerate(messages):
            if message is None:
                continue

            if use_cache_load:
                ensure_cache_registry()
                cached_response = _reconstruct_llm_response(
                    CacheRegistry.get("llm_query").load(
                        message, model, temperature, struct,
                        cache_namespace=cache_namespace),
                    struct
                )
                if cached_response is not None:
                    responses[i] = cached_response
                    continue

            uncached_indices.append(i)
            uncached_messages.append(message)

        if not uncached_messages:
            non_none_count = sum(1 for msg in messages if msg is not None)
            logger.info(f"All {non_none_count} non-None messages found in cache.")
            return responses, 0

        truncated_messages = [truncate_message(msg, model=model)[0] for msg in uncached_messages]
        batch_payload = [
            [{"role": "user", "content": _apply_qwen3_thinking_token(msg, model, config.llm_enable_thinking)}]
            for msg in truncated_messages
        ]

        response_format = struct if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel) else None
        completion_kwargs = _litellm_completion_kwargs(
            model,
            temperature,
            response_format=response_format,
            enable_thinking=config.llm_enable_thinking,
        )

        total_tokens = 0
        parsed_responses: List[Optional[LLMLoadedResponse]] = []
        try:
            batch_raw = _call_litellm_quietly(batch_completion, messages=batch_payload, **completion_kwargs)
            if not isinstance(batch_raw, list) or len(batch_raw) != len(batch_payload):
                raise ValueError("Unexpected batch response shape")

            for raw in batch_raw:
                parsed, token_count = _parse_litellm_response(raw, struct)
                parsed_responses.append(parsed)
                total_tokens += token_count
        except Exception as e:
            logger.warning(f"LiteLLM batch failed ({e}); falling back to sequential completions.")
            parsed_responses = []
            total_tokens = 0
            for msg in truncated_messages:
                request_content = _apply_qwen3_thinking_token(msg, model, config.llm_enable_thinking)
                raw = _call_litellm_quietly(
                    completion,
                    messages=[{"role": "user", "content": request_content}],
                    **completion_kwargs,
                )
                parsed, token_count = _parse_litellm_response(raw, struct)
                parsed_responses.append(parsed)
                total_tokens += token_count

        use_cache_store = _is_llm_store_cacheable(struct) and enable_cache
        for i, parsed in enumerate(parsed_responses):
            original_index = uncached_indices[i]
            original_message = messages[original_index]
            responses[original_index] = parsed

            if use_cache_store and original_message is not None and parsed is not None:
                if not CacheRegistry.get("llm_query").store(
                        parsed, original_message, model, temperature, struct,
                        cache_namespace=cache_namespace):
                    logger.error("Failed to store LLM response in cache.")

        return responses, total_tokens
    except Exception as e:
        logger.error(f"Failed to get batch responses from LLM: {e}")
        return [None] * len(messages), 0


def generate_tool_docs(tools: Sequence[BaseTool]) -> str:
    """Generate tool documentation for system prompts."""
    return "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
