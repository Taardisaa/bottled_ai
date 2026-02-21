from pathlib import Path


def get_repo_root() -> Path:
    """Return repository root path from utility module location.

    Args:
        None.

    Returns:
        Path: Absolute repository root path.
    """
    return Path(__file__).resolve().parents[2]


def resolve_from_repo_root(path: str) -> Path:
    """Resolve path relative to repository root when needed.

    Args:
        path: Relative or absolute filesystem path.

    Returns:
        Path: Absolute path for the input.
    """
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return get_repo_root() / output_path
