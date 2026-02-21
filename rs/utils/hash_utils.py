import hashlib
import json
from typing import Any


def sha256_from_json_payload(payload: Any) -> str:
    """Compute SHA256 hash from normalized JSON payload.

    Args:
        payload: JSON-serializable payload.

    Returns:
        str: SHA256 hex digest.
    """
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
