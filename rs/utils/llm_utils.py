from pathlib import Path
from typing import Optional, Any, Union, Tuple, List
import os
import json
from litellm import (
    completion,
    batch_completion,
    decode as litellm_decode,
    encode as litellm_encode,
    get_max_tokens as litellm_get_max_tokens,
    token_counter as litellm_token_counter,
)
from langchain.tools import BaseTool
from collections.abc import Sequence
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


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("openrouter/")


def _has_custom_base_url() -> bool:
    return bool(config.llm_base_url.strip())


def _normalize_model_for_litellm(model: str) -> str:
    if not _has_custom_base_url():
        return model

    if model.startswith("hosted_vllm/") or model.startswith("openai/") or _is_openrouter_model(model):
        return model

    return f"hosted_vllm/{model}"


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
    token_count = litellm_token_counter(model=normalized_model, text=text)
    if not isinstance(token_count, int) or token_count < 0:
        raise ValueError(f"LiteLLM token counter returned invalid value: {token_count!r}")
    return token_count


def _log_tokenizer_fallback(model: str, source: str, error: Exception) -> None:
    normalized = _normalize_model_for_tokenizer(model)
    message = f"{source} tokenizer lookup failed for {model}: {error}"
    if normalized != str(model).strip():
        logger.debug(message)
    else:
        logger.debug(message)


def _encode_text_with_model_tokenizer(text: str, model: str) -> tuple[list[int], str]:
    normalized_model = _normalize_model_for_tokenizer(model)
    try:
        token_ids = litellm_encode(model=normalized_model, text=text)
        if isinstance(token_ids, list):
            return token_ids, normalized_model
        raise ValueError(f"LiteLLM encode returned invalid value: {token_ids!r}")
    except Exception as e:
        _log_tokenizer_fallback(model, "LiteLLM", e)

    encoding = _get_tiktoken_encoding(model, log_fallback=False)
    return encoding.encode(text), "__tiktoken__"


def _decode_tokens_with_model_tokenizer(token_ids: list[int], model: str, codec_id: str) -> str:
    if codec_id != "__tiktoken__":
        return litellm_decode(model=codec_id, tokens=token_ids)

    encoding = _get_tiktoken_encoding(model, log_fallback=False)
    return encoding.decode(token_ids)


def _litellm_completion_kwargs(model: str, temperature: float, **extra_kwargs: Any) -> Dict[str, Any]:
    normalized_model = _normalize_model_for_litellm(model)
    kwargs: Dict[str, Any] = {
        "model": normalized_model,
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
    return kwargs


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
        max_tokens = litellm_get_max_tokens(normalized_model)
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    except Exception as e:
        _log_tokenizer_fallback(model, "LiteLLM max-token lookup", e)

    normalized_key = _normalized_model_token_limit_key(model)
    if normalized_key in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[normalized_key]

    if normalized_key.startswith("gpt-5"):
        return MODEL_TOKEN_LIMITS["gpt-5"]

    return 8192


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
        return (
            "Convert the following model output into a JSON object that strictly matches this JSON Schema. "
            "Return only the JSON object and no extra text.\n"
            f"Schema:\n{schema}\n"
            "Model output to convert:\n"
            f"{raw_response_text}"
        )

    return (
        "Convert the following model output into a valid JSON object. "
        "Return only the JSON object and no extra text.\n"
        "Model output to convert:\n"
        f"{raw_response_text}"
    )


def _ask_llm_once_two_layer(
        message: str,
        model: str,
        struct: LLMAcceptStructParam,
        temperature: float,
) -> Tuple[Optional[LLMLoadedResponse], int]:
    first_temperature = _normalize_temperature_for_model(model, temperature)
    first_kwargs = _litellm_completion_kwargs(model, first_temperature)
    first_raw = completion(
        messages=[{"role": "user", "content": message}],
        **first_kwargs,
    )

    if hasattr(first_raw, "model_dump"):
        first_response_dict = getattr(first_raw, "model_dump")()
    else:
        first_response_dict = first_raw
    if not isinstance(first_response_dict, dict):
        logger.error("Unexpected LiteLLM first-stage response type")
        return None, 0

    first_tokens = int(first_response_dict.get("usage", {}).get("total_tokens", 0) or 0)
    first_content = _extract_litellm_content(first_response_dict)
    if first_content is None:
        logger.error("Unexpected LiteLLM response content for first-stage unstructured call")
        return None, first_tokens

    if isinstance(first_content, (dict, list)):
        first_content_text = json.dumps(first_content, ensure_ascii=False)
    else:
        first_content_text = str(first_content)

    conversion_model = config.fast_llm_model
    if not _ensure_api_key_for_model(conversion_model):
        return None, first_tokens

    second_temperature = _normalize_temperature_for_model(conversion_model, temperature)
    second_prompt = _build_struct_convert_prompt(first_content_text, struct)
    second_kwargs = _litellm_completion_kwargs(
        conversion_model,
        second_temperature,
        response_format=struct,
    )

    second_raw = completion(
        messages=[{"role": "user", "content": second_prompt}],
        **second_kwargs,
    )
    second_parsed, second_tokens = _parse_litellm_response(second_raw, struct)
    if second_parsed is None:
        logger.error("Unexpected LiteLLM response type or failed structured parse in second-stage conversion")
        return None, first_tokens + second_tokens

    return second_parsed, first_tokens + second_tokens


def ask_llm_once(message: str,
                 model: str = "gpt-5-mini",
                 struct: Optional[LLMAcceptStructParam] = None,
                 temperature: float = 0.0,
                 cache_namespace: Optional[str] = None,
                 enable_cache: bool = True,
                 two_layer_struct_convert: bool = False) -> Tuple[Optional[LLMLoadedResponse], int]:
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
        two_layer_struct_convert (bool, optional): When True and `struct` is provided,
            process each message through `ask_llm_once` with two-layer conversion.
            Defaults to True.

    Returns:
        A tuple of (response from the LLM, total tokens used).
            Returns (None, 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return None, 0

        temperature = _normalize_temperature_for_model(model, temperature)

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

        if two_layer_struct_convert and struct is not None:
            response_to_return, total_tokens = _ask_llm_once_two_layer(
                message=message,
                model=model,
                struct=struct,
                temperature=temperature,
            )
        else:
            response_format = struct if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel) else None
            completion_kwargs = _litellm_completion_kwargs(model, temperature, response_format=response_format)

            raw_response = completion(
                messages=[{"role": "user", "content": message}],
                **completion_kwargs,
            )
            response_to_return, total_tokens = _parse_litellm_response(raw_response, struct)
        logger.info(f"LLM tokens used: {total_tokens}.")
        if response_to_return is None:
            logger.error("Unexpected LiteLLM response type or failed structured parse")
            return None, total_tokens

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
                  two_layer_struct_convert: bool = False) -> Tuple[List[Optional[LLMLoadedResponse]], int]:
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
        two_layer_struct_convert (bool, optional): When True and `struct` is provided,
            first call the target model without structured output and then run a
            second conversion pass into the requested schema. Defaults to False.

    Returns:
        A tuple of (list of responses from the LLM in the same order as input messages, total tokens used).
            Returns ([None] * len(messages), 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return [None] * len(messages), 0

        temperature = _normalize_temperature_for_model(model, temperature)

        if two_layer_struct_convert and struct is not None:
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
        batch_payload = [[{"role": "user", "content": msg}] for msg in truncated_messages]

        response_format = struct if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel) else None
        completion_kwargs = _litellm_completion_kwargs(model, temperature, response_format=response_format)

        total_tokens = 0
        parsed_responses: List[Optional[LLMLoadedResponse]] = []
        try:
            batch_raw = batch_completion(messages=batch_payload, **completion_kwargs)
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
                raw = completion(messages=[{"role": "user", "content": msg}], **completion_kwargs)
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
