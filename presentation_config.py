from pathlib import Path

from rs.utils.path_utils import get_repo_root
from rs.utils.yaml_utils import load_yaml_mapping


def _load_presentation_values(config_path: str | None = None) -> dict[str, object]:
    root = get_repo_root()
    path = Path(config_path) if config_path is not None else root / "configs" / "presentation_config.yaml"
    return load_yaml_mapping(path)


_values = _load_presentation_values()

presentation_mode = bool(_values.get("presentation_mode", False))
p_delay = str(_values.get("p_delay", "wait 60"))
p_delay_s = str(_values.get("p_delay_s", "wait 30"))
slow_events = bool(_values.get("slow_events", False))
slow_pathing = bool(_values.get("slow_pathing", False))
