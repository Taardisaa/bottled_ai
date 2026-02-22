from pathlib import Path

_tests_ai_path = Path(__file__).resolve().parents[1] / "tests" / "ai"
if _tests_ai_path.exists():
    __path__.append(str(_tests_ai_path))
