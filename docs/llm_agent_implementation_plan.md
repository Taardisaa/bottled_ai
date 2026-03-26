# LLM Multi-Agent Handler Plan

## Goal

Introduce an LLM-assisted decision system that integrates with the existing handler architecture, while preserving game stability and command correctness.

## Current Implementation Status (as of 2026-03-13)

- Phase 0: Implemented and stable (`rs/llm/config.py`, `rs/llm/orchestrator.py`, `rs/llm/validator.py`, `rs/llm/telemetry.py`, `tests/llm/test_config_validator_telemetry.py`, `tests/llm/test_base_agent.py`).
- Phase 1 (Event): Implemented and wired into shared handlers (`rs/llm/agents/event_advisor_agent.py`, `rs/llm/providers/event_llm_provider.py`, `rs/llm/integration/event_context.py`, `rs/common/handlers/common_event_handler.py`, `tests/ai/common/handlers/test_event_handler_llm_advisor.py`).
  Refinement still needed: broader seed coverage and better prompt/memory continuity across long runs.
- Phase 2 (Shop + Card Reward): Implemented with richer context and fixed-seed coverage (`rs/llm/agents/shop_purchase_advisor_agent.py`, `rs/llm/agents/card_reward_advisor_agent.py`, `rs/llm/providers/shop_purchase_llm_provider.py`, `rs/llm/providers/card_reward_llm_provider.py`, `rs/llm/integration/shop_purchase_context.py`, `rs/llm/integration/card_reward_context.py`, `rs/llm/benchmark_suite.py`, `tests/llm/test_benchmark_suite.py`).
  Refinement still needed: more quality tuning and broader replay/benchmark coverage.
- Phase 3 (Map): Implemented with deterministic-score-aware routing plus conservative overrides (`rs/llm/agents/map_advisor_agent.py`, `rs/llm/providers/map_llm_provider.py`, `rs/llm/integration/map_context.py`, `rs/common/handlers/common_map_handler.py`, `tests/ai/common/handlers/test_map_handler_llm_advisor.py`).
  Refinement still needed: more benchmark cases around elite/shop timing and Act 2/3 survivability tradeoffs.
- Phase 4 (Battle meta-advisor): Implemented as comparator-profile selection, not direct combat execution (`rs/llm/agents/battle_meta_advisor_agent.py`, `rs/llm/providers/battle_meta_llm_provider.py`, `rs/llm/integration/battle_context.py`, `rs/common/handlers/common_battle_handler.py`, `tests/ai/common/handlers/test_battle_handler_meta_advisor.py`).
  Refinement still needed: broader fight coverage and eventual follow-on path toward full battle-action advice.
- Phase 5 (Expansion / memory / caching): Partially implemented.
  Done: shared state-summary caching and compact run summaries (`rs/llm/state_summary_cache.py`), run-local decision memory (`rs/llm/decision_memory.py`), a reusable memory-backed LangGraph advisor base (`rs/llm/agents/memory_langgraph_agent.py`), concrete LangGraph-backed event/card/shop/map advisors (`rs/llm/agents/event_advisor_agent.py`, `rs/llm/agents/card_reward_advisor_agent.py`, `rs/llm/agents/shop_purchase_advisor_agent.py`, `rs/llm/agents/map_advisor_agent.py`), and a LangGraph-backed battle meta-advisor with run-local profile memory (`rs/llm/agents/battle_meta_advisor_agent.py`).
  In progress: refining memory summaries for longer runs and deciding how far to push LangGraph toward direct battle-action proposals later.
  Not implemented yet: persistent decision memory beyond process lifetime and any direct full-battle LangGraph executor.

---

## 1) Target Architecture

### 1.1 Core roles

- **AIPlayer (orchestrator)**
  Owns the main loop, selects which handler-agent to invoke, validates command legality, applies fallbacks, and sends final commands.

- **Handler subagents (advisors)**
  One subagent per decision domain (event, shop, card reward, map, etc.).
  Subagents propose structured decisions; they do not directly execute commands.

- **Tool layer**
  Shared deterministic tools:
  - battle calculator access
  - card/relic/potion/event database lookup
  - run/deck context helpers
  - command legality checker (`available_commands`, screen type constraints)

### 1.2 Integration principle

- Keep existing handlers as deterministic fallback path.
- Add LLM routing as opt-in, per-handler.
- First valid responder wins:
  1. LLM advisor output (if valid + in budget)
  2. Existing strategy/common handler
  3. Default machine handler

---

## 2) Phase Plan

## Phase 0 - Foundations (No behavior change)
- Add config flags:
  - `LLM_ENABLED`
  - `LLM_ENABLED_HANDLERS` (list)
  - `LLM_TIMEOUT_MS`
- Add telemetry schema for each decision:
  - state snapshot hash
  - handler name
  - tool calls used
  - proposed command
  - validation result
  - fallback used
  - latency/cost/confidence
- Add strict command validator:
  - command exists in `available_commands`
  - command syntax valid for screen/action
  - index/name references are in current choice list

Deliverable: infrastructure only, no strategy changes.

## Phase 1 - Single low-risk handler pilot (Event)
- Implement `EventAdvisorAgent`.
- Inputs:
  - event id/name
  - hp %, floor, relics, deck summary, choices
- Tools:
  - event DB/RAG
  - quick policy rules
- Output schema:
  - `proposed_command: str`
  - `reasoning: short`
  - `confidence: 0..1`
- Add timeout + retry (single retry max).
- Keep `CommonEventHandler` fallback active.

Success criteria:
- No protocol regressions.
- >= current baseline winrate on controlled seed set.
- Decision latency within budget.

## Phase 2 - Add Shop + Card Reward agents
- Implement:
  - `ShopPurchaseAdvisorAgent`
  - `CardRewardAdvisorAgent`
- Required tools:
  - card/relic valuation API
  - deck composition metrics
  - potion slot optimizer
- Use same strict output schema + validation.
- Keep existing handlers as fallback.

Success criteria:
- Stable run completion rate.
- Improved deck quality proxy metrics (dead draws, curve, synergy score).

## Phase 3 - Map routing advisor
- Implement `MapAdvisorAgent` that proposes path choice.
- Must consume deterministic path scores from existing map/path module first.
- LLM only adjusts priorities under uncertainty (hp risk, upcoming elites/shop timing).

Success criteria:
- No invalid map commands.
- Better survivability/reward tradeoff vs baseline.

## Phase 4 - Battle meta-advisor (not card-by-card executor)
- Do **not** replace calculator search initially.
- Implement advisor that selects/tunes comparator profile or calculator config.
- Keep action generation deterministic via current calculator.

Success criteria:
- No combat command regressions.
- Measure incremental EV gain in benchmark fights.

## Phase 5 - Memory and LangGraph expansion
- Keep the current handler/orchestrator contract, but allow concrete advisors to be implemented as LangGraph workflows instead of bespoke agent classes.
- Use `LangGraphBaseAgent` as the preferred extension point for future complex advisors.
- Add run-scoped memory infrastructure:
  - cached run summary
  - recent accepted LLM decisions
  - compact memory summaries injected into later prompts
- Keep memory storage separate from LangGraph itself:
  - LangGraph orchestrates flow
  - a plain memory store provides read/write state for graph nodes
- Start with lightweight graph shapes:
  1. load run summary + recent decision memory
  2. optional deterministic/tool enrichment
  3. LLM decision node
  4. parse/normalize back into the existing advisor output contract
- Defer long-horizon autonomous battle execution until the memory-backed graph pattern proves stable.

---

## 3) Data Contracts

## 3.1 Orchestrator -> subagent input
- `handler_type`
- `screen_type`
- `available_commands`
- `choice_list`
- `room_type`, `floor`, `act`
- player/deck/relic/potion summary
- handler-specific context blob

## 3.2 Subagent -> orchestrator output (strict JSON)
- `proposed_command: string`
- `confidence: number`
- `required_tools_used: string[]`
- `explanation: string` (1-2 lines)
- `fallback_recommended: boolean`

## 3.3 Validation gates (hard fail -> fallback)
- command in `available_commands`
- referenced index/card exists now
- expected follow-up sequence valid (`choose/confirm/wait` rules)
- within timeout and token budget

---

## 4) Tooling Plan

- **Battle tool**: wrap existing calculator entrypoint for read-only "best action context".
- **RAG/DB tools**:
  - card metadata + tags + upgrade deltas
  - relic metadata + synergies/anti-synergies
  - potion metadata + tactical use windows
  - event outcomes and risk notes
- **State summary tool**:
  - normalized summary of deck/relics/powers.

---

## 5) Reliability Controls

- Global timeout circuit breaker per decision.
- Per-handler disable switch when failure rate spikes.
- Confidence threshold: low confidence auto-fallback.
- Deterministic seed replay harness for regression checks.
- Red-team tests for invalid command proposals.

---

## 6) Evaluation & Benchmarking

- Build fixed seed suite (acts 1-3 mix, all classes, edge events).
- Track:
  - run completion rate
  - winrate
  - average floor reached
  - invalid command rate
  - fallback rate
  - decision latency p50/p95
  - token cost per run
- Compare against current deterministic baseline each phase.

---

## 7) Repo Implementation Tasks (Suggested)

1. Add `rs/llm/` package:
   - `orchestrator.py`
   - `agents/`
   - `schemas.py`
   - `tooling/`
2. Add config surface (env or config file) for rollout flags.
3. Add logging hooks in game loop around handler execution.
4. Add tests:
   - schema validation tests
   - command legality tests
   - fallback behavior tests
   - deterministic replay smoke tests
5. Add docs:
   - architecture
   - operational runbook
   - prompt + tool contracts

---

## 8) Risks and Mitigations

- **Latency spikes** -> strict timeout + fallback.
- **Hallucinated commands** -> hard validator.
- **Cost growth** -> selective handler activation + cached summaries.
- **Behavior drift** -> replay tests + phase gates.
- **Prompt/tool coupling fragility** -> typed schemas and compatibility tests.

---

## 9) Definition of Done (per phase)

- All new tests pass.
- No increase in invalid command rate.
- Fallback path verified.
- Metrics dashboard updated with phase comparison.
- Rollout flag default remains safe/off until acceptance.

---

## 10) Immediate Next Step

Stabilize the LangGraph-backed Phase 5 rollout:
- keep using the plain decision memory store as the graph-readable memory source
- tune longer-run memory summaries so prompts stay compact and useful
- decide whether the next experimental step is persistent run memory or a direct battle-action pilot behind a hard safety flag
