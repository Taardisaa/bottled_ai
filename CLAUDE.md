# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bottled AI is a Python bot for Slay the Spire that communicates with the game via the Communication Mod's text I/O protocol. It has two decision layers: a rule-based handler pipeline and an LLM-backed advisor system (LangGraph/LangChain).

## Commands

### Run tests (CI parity)
```bash
python -m unittest discover -s ./tests/
```

### Run with extensive tests (as CI does)
```bash
EXTENSIVE_TESTS=true python -m unittest discover -s ./tests/
```

### Run a single test file / class / method
```bash
python -m unittest tests.helpers.test_seed
python -m unittest tests.helpers.test_seed.TestSeed
python -m unittest tests.helpers.test_seed.TestSeed.test_make_seed_string_number
```

### Pytest-style tests (only where explicitly used)
```bash
python -m pytest tests/handlers/test_grid_select_handler.py
```

### Run the bot (requires Communication Mod pipe; blocks on stdin without it)
```bash
python main.py --strategy peaceful_pummeling
```

### No build or lint step is configured.

## Architecture

### Game loop (`rs/machine/`)
`Game` receives JSON state from Communication Mod via `Client`, wraps it in `GameState`, then dispatches to handlers. Two dispatch paths exist in priority order:

1. **AIPlayerGraph** (LangGraph): if enabled in `configs/llm_config.yaml`, a LangGraph state machine routes to LLM advisor agents for events, shop, card rewards, map, and battle meta decisions.
2. **Handler pipeline**: strategy-specific handlers (`strategy.handlers`) are tried first, then `DEFAULT_GAME_HANDLERS`. First handler where `can_handle(state)` returns `True` wins.

### Handler contract (`rs/machine/handlers/handler.py`)
Every handler implements:
- `can_handle(state: GameState) -> bool`
- `handle(state: GameState) -> HandlerAction`

`HandlerAction` wraps a list of protocol command strings and an optional `TheBotsMemoryBook` update.

### Strategies (`rs/ai/`)
Each strategy (e.g., `PEACEFUL_PUMMELING`, `CLAW_IS_LAW`) is an `AiStrategy` with a character, a name, and an ordered list of handlers. Strategies are selected in `main.py`.

### Battle calculator (`rs/calculator/`)
Simulates combat outcomes via graph traversal to find optimal card play sequences. Weighs ~40 evaluation factors.

**Critical rule for `rs/calculator/interfaces/`**: this package must only import from `interfaces/` or `enums/` — never from concrete calculator classes. Outside of creation points (converter/tests), prefer interface types over concrete classes. See `_A_NOTE_ABOUT_INTERFACES.md`.

### LLM advisor system (`rs/llm/`)
- `config.py` / `configs/llm_config.yaml`: controls which handlers get LLM advice, timeouts, confidence thresholds. Env vars override YAML values.
- `ai_player_graph.py`: LangGraph state machine that routes decisions to handler-specific nodes.
- `agents/`: per-handler advisor agents (`EventAdvisorAgent`, `MapAdvisorAgent`, etc.) subclassing `BaseAgent`.
- `providers/`: build LLM prompts and parse responses for each handler type.
- `integration/`: build `AgentContext` from game state for each handler.
- `langmem_service.py`: episodic/semantic memory via LangMem, stores in SQLite.
- `runtime.py`: singleton factories for orchestrator and graph instances.

### Other key packages
- `rs/game/`: domain models — cards, deck, events, map/path, rooms, screen types.
- `rs/common/`: shared handlers and comparators reused across strategies.
- `rs/api/`: Communication Mod client and transport layer.
- `rs/helper/`: logging, run controller, seed utilities.
- `rs/cache/`: caching utilities.
- `rs/utils/`: LLM utilities, YAML loading, path helpers.

### Config files (`configs/`)
- `llm_config.yaml`: LLM subsystem toggle, handler allowlist, timeouts, LangMem settings.
- `llm_utils_config.yaml`: LLM provider/model configuration for litellm.
- `presentation_config.yaml`: delays and presentation mode for video recording.

### Test layout (`tests/`)
Tests mirror `rs/` structure. JSON fixtures live in `tests/res/`. Use `base_test_handler_fixture.py` for handler test scaffolding.

## Commit Message Format

```
A brief description of this commit.

- file1: what's changed
- file2: ...
```

## Key Conventions
- Absolute imports from repo root (`rs...`, `definitions...`).
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module constants and strategy objects.
- Protocol command sequence ordering (`choose`/`confirm`/`wait`) matters — don't reorder without verifying behavior.
- Copy mutable inputs when storing configs (`.copy()` pattern).
- Preserve `TheBotsMemoryBook` semantics across run/battle/turn transitions.
- When pushing to remote: if this is a forked repo, push to the fork, not the original as a PR (unless explicitly asked).
