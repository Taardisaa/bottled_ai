"""
A module providing basic utility functions for file operations.

Functions:
 - is_path_allowed: Checks if a target path is allowed based on a list of allowed paths.
 - safe_remove: Safely removes a directory or file within the allowed dataset directory.
 - safe_clear_dir: Safely cleans a directory by removing all its contents within the allowed dataset directory.
 - safe_rename_dir: Safely renames (moves) a directory within the allowed dataset directory.
 - safe_copy: Safely copies a directory or file within the allowed dataset directory.
 - is_file_valid: Checks if a file is valid based on its age.
 - format_id: Formats an identifier to be filesystem-friendly.
 - invalidate_cache: Invalidates a cache file by deleting it.
 - remove_symlinks: Removes symbolic links in a directory tree.
"""

import shutil
from pathlib import Path
from typing import Union, List
import sys
import os
from tqdm import tqdm
import time
from typing import Optional
from loguru import logger

from rs.utils.config import config

def is_path_allowed(target_path: Path, allowed_paths: List[Path]) -> bool:
    """
    Check if a target path is allowed based on a list of allowed paths.

    Allowed paths can be either directories or specific files:
    - If an allowed path is a directory, the target must be within that directory
    - If an allowed path is a file, the target must match that file exactly

    Args:
        target_path: The path to check (should be resolved)
        allowed_paths: List of allowed paths (should be resolved)

    Returns:
        bool: True if the target path is allowed, False otherwise
    """
    for allowed_path in allowed_paths:
        # If allowed_path is a directory, check containment
        if allowed_path.is_dir():
            if target_path.is_relative_to(allowed_path):
                return True
        # If allowed_path is a file, check exact match
        else:
            if target_path == allowed_path:
                return True
    return False


def safe_remove(src_path: Union[str, Path], 
                    allowed_dir_path: Union[str, Path]=config.dataset_dir_path) -> bool:
    """
    Safely remove a directory or file if it is within the allowed dataset directory path.

    Args:
        src_path (Union[str, Path]): The directory or file path to remove.
        allowed_dir_path (Union[str, Path]): The base dataset directory path within which removal is allowed. 
            Defaults to config.dataset_dir_path.

    Return:
        bool: True if the directory/file was successfully removed or did not exist, 
            False if failed or error occurred.
    
    Security:
        Uses .resolve() to prevent directory traversal attacks (e.g., ../../etc/passwd).
        Resolves symlinks to ensure the actual target path is within allowed boundaries.
    """
    try:
        # SECURITY: Use resolve() to canonicalize paths and follow symlinks
        # This prevents directory traversal attacks using .. or symlinks
        src_path = Path(src_path).resolve()
        dataset_path = Path(allowed_dir_path).resolve()
        
        if not src_path.is_relative_to(dataset_path):
            logger.error(f"Refusing to remove path outside of dataset path: {src_path}")
            return False
        
        if src_path.exists():
            if src_path.is_dir():
                shutil.rmtree(src_path)
            elif src_path.is_file():
                src_path.unlink()
            else:
                logger.warning(f"Path {src_path} is neither a file nor directory. Skipping removal.")
                return False
        
        if not src_path.exists():
            logger.info(f"Path {src_path} has been safely removed.")
            return True
        else:
            logger.error(f"Failed to remove path {src_path}.")
            return False
    except Exception as e:
        logger.error(f"Unexpected error while removing path {src_path}: {e}")
        return False


def safe_clear_dir(dir_path: Union[str, Path],
                    allowed_dir_path: Union[str, Path]=config.dataset_dir_path) -> bool:
    """
    Safely clean a directory by removing all its contents if it is within the allowed dataset directory path.

    Args:
        dir_path (Union[str, Path]): The directory path to clean.
        allowed_dir_path (Union[str, Path]): The base dataset directory path within which cleaning is allowed. Defaults to config.dataset_dir_path.

    Return:
        bool: True if the directory was successfully cleaned, False otherwise.

    NOTE: `allowed_dir_path` and path resolution will be handled by `safe_remove` internally.
    """
    try:
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            logger.error(f"Path {dir_path} is not a directory. Cannot clean.")
            return False

        if safe_remove(dir_path, allowed_dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory {dir_path} has been safely cleaned.")
            return True
        else:
            logger.error(f"Failed to clean directory {dir_path}.")
            return False
    except Exception as e:
        logger.error(f"Unexpected error while cleaning directory {dir_path}: {e}")
        return False
    

def safe_rename_dir(src_dir: Union[str, Path], dest_dir: Union[str, Path],
                    override: bool = False, 
                    allowed_dir: Union[str, Path] = config.dataset_dir_path) -> bool:
    """
    Safely rename (move) a directory to a new location within the allowed dataset directory path.

    Args:
        src_dir (Path): The source directory path to rename.
        dest_dir (Path): The destination directory path.
        allowed_dir (Path): The base dataset directory path within which renaming is allowed. Defaults to config.dataset_dir_path.

    Returns:
        bool: True if the directory was successfully renamed, False otherwise.
        
    Security:
        Uses .resolve() to prevent directory traversal attacks.
    """
    try:
        src_dir = Path(src_dir).resolve()
        dest_dir = Path(dest_dir).resolve()
        allowed_dir = Path(allowed_dir).resolve()

        if not src_dir.is_dir() or not src_dir.exists():
            logger.error(f"Source directory {src_dir} does not exist or is not a directory. Rename aborted.")
            return False

        if not src_dir.is_relative_to(allowed_dir) or not dest_dir.is_relative_to(allowed_dir):
            logger.error(f"Refusing to rename directory outside of allowed path: {src_dir} to {dest_dir}")
            return False

        if dest_dir.exists():
            if override:
                logger.warning(f"Destination directory {dest_dir} already exists and will be overridden.")
                safe_clear_dir(dest_dir, allowed_dir)
            else:
                logger.error(f"Destination directory {dest_dir} already exists. Dir rename aborted.")
                return False

        shutil.move(str(src_dir), str(dest_dir))
        logger.info(f"Directory {src_dir} has been renamed to {dest_dir}.")
        return True
    except Exception as e:
        logger.error(f"Error renaming directory {src_dir} to {dest_dir}: {e}")
        return False
    

def safe_copy(src_path: Union[str, Path], dest_path: Union[str, Path],
                override: bool = False,
                allowed_dir: Union[str, Path] = config.dataset_dir_path) -> bool:
    """
    Safely copy a directory or file to a new location within the allowed dataset directory path.

    Args:
        src_path (Union[str, Path]): The source directory or file path to copy.
        dest_path (Union[str, Path]): The destination directory or file path.
        allowed_dir (Union[str, Path]): The base dataset directory path within which copying is allowed. Defaults to config.dataset_dir_path.

    Returns:
        bool: True if the copy was successful, False otherwise.
        
    Security:
        Uses .resolve() to prevent directory traversal attacks.
    """
    try:
        src_path = Path(src_path).resolve()
        dest_path = Path(dest_path).resolve()
        allowed_dir = Path(allowed_dir).resolve()

        if not src_path.exists():
            logger.error(f"Source path {src_path} does not exist. Copy aborted.")
            return False

        if not src_path.is_relative_to(allowed_dir) or not dest_path.is_relative_to(allowed_dir):
            logger.error(f"Refusing to copy path outside of allowed path: {src_path} to {dest_path}")
            return False

        if dest_path.exists():
            if override:
                logger.warning(f"Destination path {dest_path} already exists and will be overridden.")
                safe_remove(dest_path, allowed_dir)
            else:
                logger.error(f"Destination path {dest_path} already exists. Copy aborted.")
                return False

        if src_path.is_dir():
            shutil.copytree(src_path, dest_path)
        elif src_path.is_file():
            shutil.copy2(src_path, dest_path)
        else:
            logger.error(f"Source path {src_path} is neither a file nor directory. Copy aborted.")
            return False

        if dest_path.exists():
            logger.info(f"Path {src_path} has been copied to {dest_path}.")
            return True
        else:
            logger.error(f"Failed to copy path {src_path} to {dest_path}.")
            return False
    except Exception as e:
        logger.error(f"Error copying path {src_path} to {dest_path}: {e}")
        return False


def is_file_valid(file_path: Optional[Path], max_age_days: Optional[int] = 21) -> bool:
    """
    Check if a file is valid based on its age. Usually used to check if a cache file is still valid.

    Args:
        file_path (Path): The path to the file.
        max_age_days (Optional[int]): The maximum age of the file in days. Defaults to 21.

    Returns:
        bool: True if the file is valid (exists and is within the age limit), False otherwise.
    """
    try:
        if file_path is None:
            logger.info(f"File path is None.")
            return False
        
        if not file_path.is_file():
            logger.info(f"Cache file {file_path} does not exist.")
            return False
        
        if max_age_days is None:
            # No age limit specified, consider file always valid as long as it exists
            return True

        # Get the modification time of the cache file
        file_mod_time = file_path.stat().st_mtime
        file_age_days = (time.time() - file_mod_time) / (60 * 60 * 24)

        if file_age_days > max_age_days:
            logger.info(f"Cache file {file_path} is older than {max_age_days} days.")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking validity of cache file {file_path}: {e}")
        return False


def format_id(identifier: str) -> str:
    """
    Format an identifier to be filesystem-friendly by replacing or removing problematic characters.
    Uses whitelist approach: only allows A-Z, a-z, 0-9, and hyphens (-).
    All other characters are replaced with '#'.

    Args:
        identifier (str): The original identifier string (e.g., "CVE-2022-40304").
    Returns:
        str: The formatted identifier string.
    """
    formatted_id = ""
    for char in identifier:
        if char.isalnum() or char == '-':
            formatted_id += char
        else:
            formatted_id += '#'
    return formatted_id


def invalidate_cache(cache_path: Path) -> bool:
    """
    Invalidate a cache file by deleting it.

    NOTE: This function uses `safe_remove` to ensure safe deletion. Therefore, 
    the `cache_path` must be within the default allowed dataset directory path `config.dataset_dir_path`.

    Args:
        cache_path (Path): The path to the cache file.

    Returns:
        bool: True if the cache file was successfully deleted or did not exist, False otherwise.
    """
    return safe_remove(cache_path)


def remove_symlinks(dir_path: Union[str, Path], broken_only: bool = True) -> Optional[int]:
    """
    Remove symbolic links in a directory tree.
    
    Args:
        dir_path: Path to the directory to scan
        broken_only: If True, only remove broken symlinks. If False, remove all symlinks.

    Returns:
        Optional[int]: The number of symlinks removed, or None if the target path is not a directory.
    """
    try:
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            logger.error(f"Path {dir_path} is not a directory. Cannot remove symlinks.")
            return None
    except Exception as e:
        logger.error(f"Error resolving directory {dir_path}: {e}")
        return None
    
    try:
        # Scan and remove with progress bar
        removed_count = 0
        mode_str = "broken symlinks" if broken_only else "all symlinks"
        
        for item in tqdm(dir_path.rglob('*'), desc=f"Checking {mode_str}", unit="item"):
            # Check if it's a symlink
            if item.is_symlink():
                should_remove = False
                
                if broken_only:
                    # Check if symlink is broken
                    try:
                        item.resolve(strict=True)
                        should_remove = False
                    except (FileNotFoundError, RuntimeError):
                        # Broken symlink - target doesn't exist
                        should_remove = True
                else:
                    # Remove all symlinks regardless of validity
                    should_remove = True
                
                if should_remove:
                    symlink_type = "broken " if broken_only else ""
                    tqdm.write(f"Removing {symlink_type}symlink: {item}")
                    try:
                        item.unlink()
                        removed_count += 1
                    except Exception as e:
                        tqdm.write(f"  ✗ Failed to remove: {e}")
        
        print(f"\n✓ Done! Removed {removed_count} {mode_str}.")
        return removed_count
    except Exception as e:
        logger.error(f"Unexpected error while removing symlinks in {dir_path}: {e}")
        return None
    
    
if __name__ == "__main__":
    raise RuntimeError(f"Module {__file__} is not meant to be run directly.")

