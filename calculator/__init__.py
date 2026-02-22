from pathlib import Path

_tests_calculator_path = Path(__file__).resolve().parents[1] / "tests" / "calculator"
if _tests_calculator_path.exists():
    __path__.append(str(_tests_calculator_path))
