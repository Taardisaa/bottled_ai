from pathlib import Path
from typing import Optional, Any, Union, Tuple, List
import os
import json
from litellm import completion, batch_completion
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


def _is_openrouter_model(model: str) -> bool:
    return model.startswith("openrouter/")


def _litellm_completion_kwargs(model: str, temperature: float) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if _is_openrouter_model(model):
        kwargs["api_base"] = config.openrouter_base_url
        kwargs["api_key"] = config.openrouter_key or config.openai_key
    return kwargs


# TODO: Current impl only supports OpenAI and Anthropic models. Add more providers as needed.
def _ensure_api_key_for_model(model: str) -> bool:
    """Ensure the appropriate API key is set for the given model.

    Returns:
        True if the API key is available, False otherwise.
    """
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

    if not config_key:
        logger.error(f"API key for model '{model}' is not set. Check your .env file.")
        return False

    if not os.environ.get(env_var):
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
    return MODEL_TOKEN_LIMITS.get(model, 8192)


def count_tokens(text: str, model) -> int:
    """Count the number of tokens in a text string.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The model to use for tokenization. Defaults to "gpt-4".
        
    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Try to get the encoding for the specific model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
            logger.warning(f"Model {model} not found in tiktoken, using cl100k_base encoding")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Failed to count tokens: {e}. Estimating based on character count.")
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
    
    # Binary search to find the right truncation point
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(message)
    truncated_tokens = tokens[:max_tokens]
    truncated_message = encoding.decode(truncated_tokens)
    
    logger.info(f"Message truncated from {current_tokens} to {len(truncated_tokens)} tokens")
    return truncated_message, max_tokens - len(truncated_tokens)


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


def ask_llm_once(message: str,
                 model: str = "gpt-5-mini",
                 struct: Optional[LLMAcceptStructParam] = None,
                 temperature: float = 0.0,
                 cache_namespace: Optional[str] = None,
                 enable_cache: bool = True) -> Tuple[Optional[LLMLoadedResponse], int]:
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

    Returns:
        A tuple of (response from the LLM, total tokens used).
            Returns (None, 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return None, 0

        if model in ['gpt-5', 'gpt-5-mini'] and temperature != 1.0:
            logger.warning(f"Model {model} does not support temperature other than 1.0. Setting temperature to 1.0.")
            temperature = 1.0

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

        completion_kwargs = _litellm_completion_kwargs(model, temperature)
        if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel):
            completion_kwargs["response_format"] = struct

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
                  enable_cache: bool = True) -> Tuple[List[Optional[LLMLoadedResponse]], int]:
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

    Returns:
        A tuple of (list of responses from the LLM in the same order as input messages, total tokens used).
            Returns ([None] * len(messages), 0) on error.
    """
    try:
        from rs.cache import CacheRegistry, ensure_cache_registry

        if not _ensure_api_key_for_model(model):
            return [None] * len(messages), 0

        if model in ['gpt-5', 'gpt-5-mini'] and temperature != 1.0:
            logger.warning(f"Model {model} does not support temperature other than 1.0. Setting temperature to 1.0.")
            temperature = 1.0

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

        completion_kwargs = _litellm_completion_kwargs(model, temperature)
        if struct is not None and isinstance(struct, type) and issubclass(struct, BaseModel):
            completion_kwargs["response_format"] = struct

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
