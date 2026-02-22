"""File locking and atomic write utilities for safe concurrent file operations."""

import json
import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, List, Dict

from loguru import logger

if os.name == "nt":
    import msvcrt
else:
    import fcntl

T = TypeVar('T')


def _lock_file(file_obj: Any, shared: bool) -> None:
    if os.name == "nt":
        file_obj.flush()
        file_obj.seek(0)
        mode = msvcrt.LK_RLCK if shared else msvcrt.LK_LOCK
        msvcrt.locking(file_obj.fileno(), mode, 1)
        return

    lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fcntl.flock(file_obj.fileno(), lock_type)


def _unlock_file(file_obj: Any) -> None:
    if os.name == "nt":
        file_obj.flush()
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def read_with_lock(file_path: Path, deserializer: Optional[Callable[[str], T]] = None) -> Optional[T]:
    """
    Read file content with shared lock (allows concurrent reads).
    
    Args:
        file_path: Path to the file to read
        deserializer: Optional function to deserialize the content (e.g., json.loads)
    
    Returns:
        Deserialized content if deserializer provided, else raw string content.
        Returns None if file doesn't exist or on error.
    """
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            _lock_file(f, shared=True)
            try:
                content = f.read()
                if deserializer:
                    return deserializer(content)
                return content  # type: ignore
            finally:
                _unlock_file(f)
    except Exception as e:
        logger.error(f"Failed to read file {file_path} with lock: {e}")
        return None


def write_with_lock(
    file_path: Path,
    data: Any,
    serializer: Optional[Callable[[Any], str]] = None,
    merge_fn: Optional[Callable[[Any, Any], Any]] = None
) -> bool:
    """
    Write data to file atomically with exclusive lock.

    Args:
        file_path: Path to the file to write
        data: Data to write
        serializer: Optional function to serialize data to string.
                   If None, auto-detects: dict/list→json.dumps, str→direct, else→str()
        merge_fn: Optional function to merge data with existing content.
                  Signature: merge_fn(new_data, existing_data) -> merged_data

    Returns:
        True if write succeeded, False otherwise
    """
    # Auto-detect serializer if not provided
    if serializer is None:
        if isinstance(data, (dict, list)):
            serializer = json.dumps
        elif isinstance(data, str):
            serializer = lambda x: x
        else:
            serializer = str

    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create temp file in same directory to ensure same filesystem
        temp_fd, temp_path = tempfile.mkstemp(dir=file_path.parent, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w') as temp_f:
                _lock_file(temp_f, shared=False)
                try:
                    # Merge with existing data if merge function provided
                    write_data = data
                    if merge_fn and file_path.exists():
                        try:
                            with open(file_path, 'r') as existing_f:
                                existing_content = existing_f.read()
                                # Parse existing content using same serializer pattern
                                # Assume inverse operation (e.g., json.loads for json.dumps)
                                if hasattr(serializer, '__self__') and hasattr(serializer.__self__, 'loads'):   # type: ignore
                                    existing_data = serializer.__self__.loads(existing_content)  # type: ignore
                                elif serializer == json.dumps:
                                    existing_data = json.loads(existing_content)
                                else:
                                    # For custom serializers, pass raw content
                                    existing_data = existing_content
                                
                                write_data = merge_fn(data, existing_data)
                        except Exception as e:
                            logger.warning(f"Failed to merge with existing file {file_path}: {e}")
                    
                    # Write merged data
                    temp_f.write(serializer(write_data))
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Ensure data written to disk
                finally:
                    _unlock_file(temp_f)
            
            # Atomic rename (overwrites existing file)
            os.replace(temp_path, file_path)
            return True
        except:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception as e:
        logger.error(f"Failed to write file {file_path} with lock: {e}")
        return False


def read_json_with_lock(file_path: Path) -> Optional[Any]:
    """
    Convenience function to read JSON file with lock.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Parsed JSON data or None on error
    """
    return read_with_lock(file_path, deserializer=json.loads)


def write_json_with_lock(
    file_path: Path,
    data: Any,
    merge_fn: Optional[Callable[[Any, Any], Any]] = None,
    indent: Optional[int] = 4
) -> bool:
    """
    Convenience function to write JSON file with lock.

    Args:
        file_path: Path to JSON file
        data: Data to serialize as JSON
        merge_fn: Optional merge function (new_data, existing_data) -> merged_data
        indent: JSON indentation (default 4)

    Returns:
        True if write succeeded, False otherwise
    """
    serializer = lambda d: json.dumps(d, indent=indent)
    return write_with_lock(file_path, data, serializer, merge_fn)


def _deserialize_jsonl(content: str) -> List[Dict[str, Any]]:
    """
    Deserialize JSONL content into a list of dicts.

    Args:
        content: JSONL formatted string

    Returns:
        List of parsed JSON objects
    """
    items: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _serialize_jsonl(data: List[Dict[str, Any]]) -> str:
    """
    Serialize a list of dicts into JSONL format.

    Args:
        data: List of dicts to serialize

    Returns:
        JSONL formatted string (one JSON object per line)
    """
    return '\n'.join(json.dumps(item) for item in data)


def read_jsonl_with_lock(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Read JSONL file with shared lock (allows concurrent reads).

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects or None on error/if file doesn't exist
    """
    return read_with_lock(file_path, deserializer=_deserialize_jsonl)


def write_jsonl_with_lock(
    file_path: Path,
    data: List[Dict[str, Any]],
    merge_fn: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
) -> bool:
    """
    Write JSONL file atomically with exclusive lock.

    Args:
        file_path: Path to JSONL file
        data: List of dicts to serialize as JSONL
        merge_fn: Optional merge function (new_data, existing_data) -> merged_data
                  Example: lambda new, old: old + new  (append mode)

    Returns:
        True if write succeeded, False otherwise
    """
    return write_with_lock(file_path, data, _serialize_jsonl, merge_fn)


def read_json_from_zip_with_lock(zip_path: Path) -> Optional[Any]:
    if not zip_path.is_file():
        return None

    try:
        with open(zip_path, "rb") as f:
            _lock_file(f, shared=True)
            try:
                zip_bytes = f.read()
            finally:
                _unlock_file(f)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            members = zf.namelist()
            if not members:
                logger.error(f"Zip file is empty: {zip_path}")
                return None
            json_content = zf.read(members[0]).decode("utf-8")
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
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        if json_filename is None:
            json_filename = zip_path.stem
            if not json_filename.endswith(".json"):
                json_filename += ".json"

        json_bytes = json.dumps(data, indent=indent).encode("utf-8")

        temp_fd, temp_path = tempfile.mkstemp(dir=zip_path.parent, suffix=".tmp.zip")
        try:
            with os.fdopen(temp_fd, "wb") as temp_f:
                _lock_file(temp_f, shared=False)
                try:
                    with zipfile.ZipFile(temp_f, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(json_filename, json_bytes)
                    temp_f.flush()
                    os.fsync(temp_f.fileno())
                finally:
                    _unlock_file(temp_f)

            os.replace(temp_path, zip_path)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception as e:
        logger.error(f"Failed to write JSON as zip to {zip_path}: {e}")
        return False
