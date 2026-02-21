from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load YAML file ensuring the root node is a mapping.

    Args:
        path: YAML file path.

    Returns:
        Dict[str, Any]: Parsed root mapping or empty mapping when file is missing/empty.
    """
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return dict(loaded)
