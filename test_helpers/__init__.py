from pathlib import Path

_tests_helpers_path = Path(__file__).resolve().parents[1] / "tests" / "test_helpers"
if _tests_helpers_path.exists():
    __path__.append(str(_tests_helpers_path))
