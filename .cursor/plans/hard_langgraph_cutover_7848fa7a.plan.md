---
name: Hard LangGraph Cutover
overview: Replace custom directive-based tool orchestration with LangGraph native tool-calling only across battle, reward, and campfire subagents, with no custom fallback path.
todos:
  - id: battle-native-cutover
    content: Refactor battle subagent to native LangGraph tool-calling and remove custom directive parsing.
    status: in_progress
  - id: reward-campfire-native-cutover
    content: Refactor reward and campfire subagents to native tool-calling only; remove custom proposal loops.
    status: pending
  - id: remove-custom-provider-contracts
    content: Delete or simplify provider directive schemas and wiring no longer used by hard cutover.
    status: pending
  - id: tool-schema-and-guardrails
    content: Implement strict tool schemas and migrate guardrails/validation into native tool-call execution path.
    status: pending
  - id: tests-and-gates
    content: Update tests and validator script, then run real and full suites as cutover gate.
    status: pending
isProject: false
---

# Hard Cutover to LangGraph Tool Calling

## Scope

- Migrate all runtime-aware subagents to native LangGraph/OpenAI tool-calling only:
  - `[rs/llm/battle_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/battle_subagent.py)`
  - `[rs/llm/reward_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/reward_subagent.py)`
  - `[rs/llm/campfire_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/campfire_subagent.py)`
- Remove custom `mode/tool_name/tool_payload` directive parsing path.

## Design Changes

- Replace provider-driven directive loop with LangGraph-native agent loop that reads `tool_calls` from model responses.
- Keep existing tool implementations and interfaces where possible:
  - battle tools in `[rs/llm/battle_tools.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/battle_tools.py)`
- Build tool schemas from current tool descriptions and payload contracts.
- Preserve existing validation/guardrail semantics by moving checks into tool execution wrappers and middleware:
  - choose-command validation
  - max loops / max tool calls
  - action commit / exit checks

## File-Level Plan

- Update `[rs/llm/battle_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/battle_subagent.py)`
  - remove `BattleDirective` decision parsing path
  - wire native tool-calling model node
  - keep runtime state transitions and session summary recording
- Update `[rs/llm/reward_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/reward_subagent.py)`
  - remove provider proposal action-only loop
  - adopt tool-call driven actions for reward flows
- Update `[rs/llm/campfire_subagent.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/campfire_subagent.py)`
  - replace command proposal parsing with native tool-calling actions
- Update/remove provider contracts that only exist for custom directives:
  - `[rs/llm/providers/battle_llm_provider.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/providers/battle_llm_provider.py)`
  - related reward/campfire provider proposal schemas if no longer needed
- Ensure graph bootstrap remains consistent in `[rs/llm/ai_player_graph.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/rs/llm/ai_player_graph.py)`
  - registration of tools stays explicit
  - execution path expects native tool calls only

## Validation and Exit Criteria

- Extend and run `[scripts/validate_langmem_stack.py](d:/Software/Steam/steamapps/common/SlayTheSpire/bottled_ai/scripts/validate_langmem_stack.py)` as hard gate for native tool-calling readiness.
- Add/adjust tests to assert no custom directive fallback exists and native tool-calling path is used.
- Full regression run:
  - `python -m unittest tests.llm.test_langmem_service_real -v`
  - `python -m unittest discover -s ./tests/`
- Hard-cutover acceptance:
  - all three subagents execute through native tool-calling path
  - no custom directive parser path remains
  - memory retrieval/writes continue to pass existing and real tests

