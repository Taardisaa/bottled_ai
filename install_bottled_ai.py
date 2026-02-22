import argparse
import os
from pathlib import Path
import platform


def _validate_debugpy_listen(value: str) -> str:
    if ":" in value:
        host, port_text = value.rsplit(":", 1)
        if not host:
            raise argparse.ArgumentTypeError(
                "--debugpy-listen host cannot be empty (use host:port)."
            )
    else:
        port_text = value

    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--debugpy-listen must be a port or host:port with a numeric port."
        ) from exc

    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError(
            "--debugpy-listen port must be in range 1..65535."
        )

    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure CommunicationMod to launch Bottled AI.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python install_bottled_ai.py --mode python\n"
            "  python install_bottled_ai.py --mode debugpy --debugpy-listen 5678 --wait-for-client"
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to CommunicationMod config.properties. Auto-discovered when omitted.",
    )
    parser.add_argument(
        "--python-path",
        type=Path,
        help="Python executable to run Bottled AI. Defaults to repo venv python.",
    )
    parser.add_argument(
        "--mode",
        choices=["python", "debugpy"],
        default="python",
        help="Launch mode for CommunicationMod command.",
    )
    parser.add_argument(
        "--debugpy-listen",
        type=_validate_debugpy_listen,
        default="5678",
        help="debugpy listen endpoint (port or host:port). Used in debugpy mode.",
    )
    parser.add_argument(
        "--wait-for-client",
        action="store_true",
        help="Add --wait-for-client in debugpy mode.",
    )
    return parser.parse_args()


def _default_modthespire_roots() -> list[Path]:
    roots: list[Path] = []

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(Path(local_app_data) / "ModTheSpire")

    app_data = os.environ.get("APPDATA")
    if app_data:
        roots.append(Path(app_data) / "ModTheSpire")

    roots.append(Path.home() / "Library" / "Preferences" / "ModTheSpire")
    roots.append(Path.home() / ".config" / "ModTheSpire")

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _discover_config_path() -> Path:
    matches: list[Path] = []
    for root in _default_modthespire_roots():
        direct = root / "CommunicationMod" / "config.properties"
        if direct.exists():
            matches.append(direct)
            continue

        if root.exists():
            matches.extend(root.glob("**/CommunicationMod/config.properties"))

    if not matches:
        raise FileNotFoundError(
            "Could not find CommunicationMod config.properties. "
            "Pass --config-path explicitly."
        )

    unique_matches = sorted({path.resolve() for path in matches})
    if len(unique_matches) > 1:
        options = "\n".join(f"- {path}" for path in unique_matches)
        raise RuntimeError(
            "Found multiple CommunicationMod config files. "
            "Pass --config-path to choose one:\n"
            f"{options}"
        )

    return unique_matches[0]


def _default_python_path(repo_root: Path) -> Path:
    if platform.system().lower().startswith("win"):
        candidate = repo_root / "venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / "venv" / "bin" / "python"

    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find default venv python at {candidate}. "
        "Pass --python-path explicitly."
    )


def _build_command(main_py: Path, python_path: Path, mode: str, debugpy_listen: str, wait_for_client: bool) -> str:
    plain_python = python_path.as_posix()
    plain_main = main_py.as_posix()

    if mode == "debugpy":
        wait_flag = " --wait-for-client" if wait_for_client else ""
        return f"{plain_python} -m debugpy --listen {debugpy_listen}{wait_flag} {plain_main}"

    return f"{plain_python} {plain_main}"


def _write_command_to_properties(config_path: Path, command: str) -> None:
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    lines = existing.splitlines()

    replaced = False
    updated_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("command="):
            updated_lines.append(f"command={command}")
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        if updated_lines and updated_lines[-1] != "":
            updated_lines.append("")
        updated_lines.append(f"command={command}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    main_py = (repo_root / "main.py").resolve()

    config_path = args.config_path.resolve() if args.config_path else _discover_config_path()
    python_path = args.python_path.resolve() if args.python_path else _default_python_path(repo_root)

    command = _build_command(
        main_py=main_py,
        python_path=python_path,
        mode=args.mode,
        debugpy_listen=args.debugpy_listen,
        wait_for_client=args.wait_for_client,
    )
    _write_command_to_properties(config_path, command)

    print(f"Updated CommunicationMod config: {config_path}")
    print(f"command={command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
