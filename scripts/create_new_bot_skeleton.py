import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new bot skeleton from rs/ai/_example.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="New bot name (used for folder, module, and import replacements).",
    )
    return parser.parse_args()


def _replace_in_file(file_path: Path, replacement_name: str, strategy_constant: str) -> None:
    content = file_path.read_text(encoding="utf-8")
    updated = content.replace("EXAMPLE_STRATEGY", strategy_constant)
    updated = updated.replace("rs.ai._example", f"rs.ai.{replacement_name}")

    if updated != content:
        file_path.write_text(updated, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    name = args.name

    root_dir = Path(__file__).resolve().parents[1]
    source_dir = root_dir / "rs" / "ai" / "_example"
    target_dir = root_dir / "rs" / "ai" / name

    if not source_dir.exists():
        raise FileNotFoundError(f"Source strategy template not found: {source_dir}")

    if target_dir.exists():
        raise FileExistsError(f"Target strategy directory already exists: {target_dir}")

    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    source_strategy_file = target_dir / "example.py"
    target_strategy_file = target_dir / f"{name}.py"

    if not source_strategy_file.exists():
        raise FileNotFoundError(f"Template file missing after copy: {source_strategy_file}")

    source_strategy_file.rename(target_strategy_file)

    strategy_constant = name.replace("-", "_").upper()
    for python_file in target_dir.rglob("*.py"):
        _replace_in_file(
            file_path=python_file,
            replacement_name=name,
            strategy_constant=strategy_constant,
        )

    print(f"Created bot skeleton at: {target_dir}")
    print(f"Renamed strategy module to: {target_strategy_file.name}")
    print(f"Replaced EXAMPLE_STRATEGY with: {strategy_constant}")
    print(f"Replaced imports from rs.ai._example to rs.ai.{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
