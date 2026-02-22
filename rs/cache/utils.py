"""
Cache utility functions for hashing, TTL validation, and atomic file operations.

This module re-exports utilities from the main utils package for use by cache implementations,
and provides compressed JSON read/write helpers for the BaseCache compression feature.
"""

import fcntl
import io
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

from utils.common_utils import do_hash
from utils.file_lock_utils import (
    read_with_lock,
    write_with_lock,
    read_json_with_lock,
    write_json_with_lock,
)
from utils.file_utils import (
    is_file_valid,
    format_id as format_cache_id,
    safe_remove,
)

logger = logging.getLogger("cache.utils")

__all__ = [
    'compute_hash',
    'format_cache_id',
    'is_file_valid',
    'read_with_lock',
    'write_with_lock',
    'read_json_with_lock',
    'write_json_with_lock',
    'read_json_from_zip_with_lock',
    'write_json_as_zip_with_lock',
    'safe_remove',
]


def compute_hash(input_str: str, algorithm: str = "sha256") -> str:
    """
    Compute a hash of the input string.

    Args:
        input_str: String to hash
        algorithm: Hash algorithm ("sha256" or "md5")

    Returns:
        Hexadecimal hash string
    """
    return do_hash(input_str, algorithm)


def read_json_from_zip_with_lock(zip_path: Path) -> Optional[Any]:
    """
    Read JSON data from a .json.zip file with shared file lock.

    The zip is expected to contain exactly one JSON member.
    Reads the entire zip into memory while holding the lock, then
    releases the lock before decompressing/parsing.

    Args:
        zip_path: Path to the .json.zip file

    Returns:
        Parsed JSON data, or None if the file doesn't exist or on error
    """
    if not zip_path.is_file():
        return None

    try:
        with open(zip_path, 'rb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                zip_bytes = f.read()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            members = zf.namelist()
            if not members:
                logger.error(f"Zip file is empty: {zip_path}")
                return None
            json_content = zf.read(members[0]).decode('utf-8')
            return json.loads(json_content)
    except Exception as e:
        logger.error(f"Failed to read JSON from zip {zip_path}: {e}")
        return None


def write_json_as_zip_with_lock(
    zip_path: Path,
    data: Any,
    json_filename: Optional[str] = None,
    indent: Optional[int] = 4,
) -> bool:
    """
    Write JSON data atomically as a .json.zip file with exclusive file lock.

    Uses the same atomic temp-file + os.replace() pattern as write_with_lock.

    Args:
        zip_path: Path for the output .json.zip file
        data: Data to serialize as JSON
        json_filename: Name of the JSON file inside the zip.
                       Defaults to the zip stem (e.g. "abc.json" from "abc.json.zip").
        indent: JSON indentation (default 4)

    Returns:
        True if write succeeded, False otherwise
    """
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        if json_filename is None:
            # "abc.json.zip" -> stem is "abc.json"
            json_filename = zip_path.stem
            if not json_filename.endswith('.json'):
                json_filename += '.json'

        json_bytes = json.dumps(data, indent=indent).encode('utf-8')

        temp_fd, temp_path = tempfile.mkstemp(
            dir=zip_path.parent, suffix='.tmp.zip'
        )
        try:
            with os.fdopen(temp_fd, 'wb') as temp_f:
                fcntl.flock(temp_f.fileno(), fcntl.LOCK_EX)
                try:
                    with zipfile.ZipFile(temp_f, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(json_filename, json_bytes)
                    temp_f.flush()
                    os.fsync(temp_f.fileno())
                finally:
                    fcntl.flock(temp_f.fileno(), fcntl.LOCK_UN)

            os.replace(temp_path, zip_path)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception as e:
        logger.error(f"Failed to write JSON as zip to {zip_path}: {e}")
        return False
