# AGENTS.md

Guidance for autonomous coding agents working in this repository.

## Project Snapshot
- Language: Python.
- Runtime target: 3.11.8 (README and CI matrix).
- Domain: Slay the Spire bot over Communication Mod text I/O.
- Entry point: `main.py`.
- Core package: `rs/`.
- Tests: `tests/` (mostly `unittest`, some `pytest` usage exists).
- CI: `.github/workflows/ci-tests.yml` runs unittest discovery.

## Setup and Runtime Commands

### Environment setup
- Check Python version: `python --version`
- Install dependencies if present:
  - `python -m pip install --upgrade pip`
  - `if [ -f requirements.txt ]; then pip install -r requirements.txt; fi`

Notes:
- `requirements.txt` is currently empty.
- Real gameplay requires Slay the Spire and required mods listed in `README.md`.

### Run bot (local process)
- Standard run: `python main.py`

Important:
- The bot expects Communication Mod protocol input/output. Running directly without the game/mod pipe will block on input.

### Build and lint
- No formal build/package step is configured.
- CI "build" job runs tests only.
- No dedicated lint tool is configured in repo or CI.

## Test Commands

### Full suite (CI parity)
- `python -m unittest discover -s ./tests/`

### Extensive tests enabled (matches CI behavior)
- PowerShell:
  - `$env:EXTENSIVE_TESTS='true'; python -m unittest discover -s .\tests\`
- CMD:
  - `set EXTENSIVE_TESTS=true && python -m unittest discover -s .\tests\`
- Bash:
  - `EXTENSIVE_TESTS=true python -m unittest discover -s ./tests/`

### Single test file
- Preferred module form:
  - `python -m unittest tests.helpers.test_seed`
- Alternative pattern form:
  - `python -m unittest discover -s tests -p "test_seed.py"`

### Single test class
- `python -m unittest tests.helpers.test_seed.TestSeed`

### Single test method
- `python -m unittest tests.helpers.test_seed.TestSeed.test_make_seed_string_number`

### Pytest-style file (only when needed)
- `python -m pytest tests/handlers/test_grid_select_handler.py`

Notes:
- Default to `unittest` unless a file explicitly uses pytest style.
- Keep iteration runs targeted; run broader suite before handoff.

## Repository Structure
- `main.py`: process bootstrap, strategy selection, seed/run loop.
- `rs/machine/`: core game loop, state wrapper, default handlers.
- `rs/ai/`: per-strategy handler pipelines and configs.
- `rs/common/`: shared handlers and comparators.
- `rs/calculator/`: battle simulation, path search, battle evaluation.
- `rs/game/`: game-domain models (cards, events, map/path abstractions).
- `rs/helper/`: logging, run controller, seed helpers.
- `tests/res/`: JSON fixtures for behavior regression.

## Code Style and Conventions

### Imports
- Use absolute imports from repo root (`rs...`, `definitions...`).
- Group imports in this order:
  1) Python stdlib
  2) first-party repo imports
- Avoid wildcard imports.

### Formatting
- Follow existing style in each file.
- Use 4-space indentation.
- Keep lines readable; avoid compressing complex logic into one-liners.
- Add comments only when intent is not obvious.

### Types
- Add type hints for function signatures and important attributes.
- Use concrete types where practical (`List[str]`, `dict[str, int]`, enums).
- Match existing typing style in the file (some modules use `typing.List`).
- Prefer enums/constants over raw magic strings when enums exist.

### Naming
- `snake_case`: variables, functions, methods.
- `PascalCase`: classes.
- `UPPER_CASE`: module constants.
- Strategy constants are uppercase objects (example: `PEACEFUL_PUMMELING`).
- Test methods should use `test_*` naming.

### Handler architecture rules
- Every handler should implement:
  - `can_handle(state) -> bool`
  - `handle(state) -> HandlerAction`
- Keep handlers deterministic and side-effect light.
- Return protocol commands via `HandlerAction(commands=[...])`.
- Respect pipeline ordering: first matching handler wins.
- If behavior changes, verify handler order still makes sense.

### Calculator and interface rules
- Respect boundary notes in `rs/calculator/interfaces/_A_NOTE_ABOUT_INTERFACES.md`.
- In `interfaces/`, import only interfaces/enums to avoid circular dependencies.
- Outside creation points (converter/tests), prefer interface types over concrete classes.

### Error handling and logging
- Fail fast on impossible states with explicit exceptions.
- Log runtime state using helper logging utilities.
- Top-level runtime exception handling belongs in `main.py`; avoid broad silent catches lower down.
- Do not swallow exceptions without strong justification.

### State and mutability patterns
- Copy mutable inputs when storing configs (`.copy()` pattern is common).
- Preserve `TheBotsMemoryBook` semantics across run/battle/turn transitions.
- Be careful changing `choose`/`confirm`/`wait` command ordering; protocol sequence matters.

## Testing Expectations for Code Changes
- Add or update tests for behavior changes.
- Prefer fixture-driven tests using `tests/res` states.
- Keep tests near changed subsystem:
  - handlers -> `tests/ai/.../handlers/`
  - calculator -> `tests/calculator/`
  - converter/state -> `tests/game_state_converter/`
- For bug fixes, add a regression test that would fail pre-fix.

## Git and Change Hygiene
- Keep changes focused and minimal.
- Avoid unrelated refactors in the same patch.
- Match repository commit style (imperative, concise subjects).
- Avoid broad architecture rewrites unless explicitly requested.

## Agent Guardrails
- Before editing, inspect neighboring files to match patterns.
- Prefer extending common handlers instead of duplicating strategy-specific logic.
- Preserve public behavior unless the task explicitly requests behavior changes.
- If adding new command sequences, validate with representative fixtures/tests.

## Cursor/Copilot Rules Check
- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` file found.
- If these are added later, treat them as higher-priority instructions.
