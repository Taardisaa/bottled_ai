---
name: Constrain LLM Output Prompts
overview: Tighten prompt wording only (no architecture or parser changes) to reduce malformed command JSON and avoid expensive second-pass format correction calls.
todos:
  - id: audit-prompts
    content: Identify all LLM decision prompts still using ambiguous plain-text field wording and list exact contract changes.
    status: pending
  - id: tighten-battle-prompt
    content: Update battle prompt text to enforce command-batch shape and provide explicit valid/invalid examples.
    status: pending
  - id: tighten-convert-prompt
    content: Refine second-pass conversion prompt instructions in llm_utils to preserve command-line integrity.
    status: pending
  - id: align-provider-prompts
    content: Apply the same strict JSON-output contract to event/map/shop/card-reward prompts.
    status: pending
  - id: verify-with-tests-and-telemetry
    content: Run focused tests and compare retry/token metrics on a short scenario replay.
    status: pending
isProject: false
---

# Constrain LLM Output With Prompt Tightening

## Goal

Reduce wasted two-layer format-correction calls by making first-pass outputs reliably schema-valid and command-valid through stricter prompt instructions.

## Scope (Prompt-Only)

- Keep `llm_two_layer_struct_convert: true` and current repair behavior unchanged.
- Do not change validators, parsing logic, retry constants, or execution flow.
- Only update prompt text in provider templates and shared conversion prompt wording.

## Target Files

- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/battle_llm_provider.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/battle_llm_provider.py)
- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/utils/llm_utils.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/utils/llm_utils.py)
- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/event_decision_prompt.txt](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/event_decision_prompt.txt)
- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/map_decision_prompt.txt](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/map_decision_prompt.txt)
- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/shop_purchase_decision_prompt.txt](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/shop_purchase_decision_prompt.txt)
- [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/card_reward_decision_prompt.txt](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/prompts/card_reward_decision_prompt.txt)

## Planned Changes

1. Add a uniform strict output contract to all plain-text prompt templates:
  - “Return exactly one JSON object with only schema keys.”
  - “No markdown/code fences, no preface, no trailing text.”
  - “`proposed_command` must be one complete protocol command string (or null), never tokenized fragments.”
2. Tighten battle prompt command-batch instructions in `battle_llm_provider.py`:
  - Require `commands` to be an array of complete command lines (e.g. `"play 2 0"` as one element).
  - Add explicit invalid examples (e.g. `["play", "2", "0"]`, raw tool prose).
  - Require tool requests to populate `tool_name` only when mode is `tool`; otherwise keep `tool_name=null`.
3. Strengthen `_build_struct_convert_prompt` wording in `llm_utils.py` so the second pass preserves command strings rather than reinterpretation:
  - “Do not paraphrase or invent command tokens.”
  - “If source contains one command line, keep it as one string element.”
  - “Map unknown/invalid intent to schema-safe null/empty fields, not free-form text.”
4. Improve validation-feedback handling language in prompts:
  - Instruct model to minimally edit prior output to satisfy the reported validation error.
  - Prioritize legality against `available_commands` and index/token constraints before confidence/explanation.

## Verification Plan

- Run focused tests that assert prompt expectations and provider behavior:
  - [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_event_llm_provider.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_event_llm_provider.py)
  - [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_generic_llm_provider.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_generic_llm_provider.py)
  - [d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_litellm_utils.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/tests/llm/test_litellm_utils.py)
- Replay a short battle scenario and compare telemetry/log counters before vs after:
  - second-stage parse retries,
  - first-pass restart count,
  - mean tokens per `ask_llm_once` call,
  - rate of guardrail fallback from malformed command batches.

## Success Criteria

- Noticeable drop in second-stage conversion retries and token spend per decision.
- Fewer malformed `commands` arrays (especially split-token command fragments).
- No regression in existing validator/provider test coverage.

