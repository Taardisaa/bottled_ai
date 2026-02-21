# LLM Multi-Agent Handler Plan

## Goal

Introduce an LLM-assisted decision system that integrates with the existing handler architecture, while preserving game stability and command correctness.

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
  - `LLM_MAX_TOKENS_PER_DECISION`
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

## Phase 5 - Optional full multi-agent expansion
- Enable additional specialized agents only if earlier phases show value/cost efficiency.
- Add caching + memory summaries to control token use.

---

## 3) Data Contracts

## 3.1 Orchestrator -> subagent input
- `handler_type`
- `screen_type`
- `available_commands`
- `choice_list`
- `room_type`, `floor`, `act`
- compact player/deck/relic/potion summary
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
  - normalized compact summary of deck/relics/powers to reduce prompt size.

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

Implement Phase 0 + Phase 1 (EventAdvisor) only, behind feature flags, and run benchmark seeds before enabling wider handler coverage.
