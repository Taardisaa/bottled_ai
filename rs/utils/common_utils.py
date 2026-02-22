"""
This module provides common utility functions.
- run_cmd: Execute shell commands with logging and timing.
- convert_path_to_str: Convert Path objects in a list to strings.
- run_compile_cmd: Compile C files to object files using GCC.
- format_name: Format strings to be valid C identifiers.
- looks_like_int: Check if the given string can be safely converted to an integer.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
import os
from typing import Union, List, Optional, Dict, Any
import sys
import time
from contextlib import contextmanager
import signal
import json
import ctypes
import threading
import threading
import pexpect
import hashlib
import socket
from contextlib import closing

from loguru import logger


def run_cmd(cmd: List[str],
            cwd: Optional[Union[str, Path]]=None,
            timeout:Optional[int]=60) -> Optional[Dict[str, Union[str, int, float]]]:
    """Execute a shell command and return its success status and timing info.
    
    This function runs a shell command using subprocess.run and handles the output logging.
    It also measures the execution time of the command.
    
    Args:
        cmd: Command to execute as a list of strings
        cwd: Working directory to execute the command in
        timeout: Maximum time to wait for command completion in seconds
        
    Returns:
        dict: A dictionary containing:
            - stdout: Command's standard output
            - stderr: Command's standard error
            - returncode: Command's return code
            - time_elapsed: Time taken to execute the command in seconds
        None: If an exception occurs
        
    Example:
        >>> result = run_cmd(["ls", "-l"], logger)
        >>> print(f"Command took {result['time_elapsed']:.2f} seconds")
    """
    try:   
        start_time = time.perf_counter()
        if timeout is None:
            result = subprocess.run(cmd, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True,
                                    cwd=cwd)
        else:
            result = subprocess.run(cmd, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True,
                                timeout=timeout,
                                cwd=cwd)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time
        
        # if logger: 
        logger.info(f"Run Command Stderr: {result.stderr}")
        logger.info(f"Run Command Stdout: {result.stdout}")
        logger.info(f"Run Command took {time_elapsed:.2f} seconds")
            
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "time_elapsed": time_elapsed
        }
    except Exception as e:
        logger.error(f"Exception occurred while running command {' '.join(cmd)}: {e}")
        return None
    

def run_cmd_no_block(cmd: List[str], cwd: Optional[Union[str, Path]] = None) -> Optional[pexpect.spawn]:
    try:
        if len(cmd) == 0:
            logger.error("Empty command provided to run_cmd_no_block.")
            return None
        
        proc = pexpect.spawn(cmd[0], args=cmd[1:],
                             cwd=cwd, timeout=None)
        return proc
    except Exception as e:
        logger.error(f"Exception occurred while running command {' '.join(cmd)}: {e}")
        return None


def convert_path_to_str(ls: list) -> list:
    """Convert all Path objects in a list to str objects.
    Args:
        ls: List of objects, some may be Paths.
    Returns:
        list: List of objects, all as str objects.
    """
    new_ls = []
    for item in ls:
        if isinstance(item, Path):
            new_ls.append(str(item))
        else:
            new_ls.append(item)
    return new_ls


def run_compile_cmd(input_file: Union[str, Path], output_file: Union[str, Path],
                    no_compile: bool = False, timeout:int = 60,
                    keep_debug_info: bool=False) -> Union[dict, None]:
    """
    A utility function that compile a .c file into a .o file using GCC.
    """
    if isinstance(input_file, Path):
        input_file = str(input_file)
    if isinstance(output_file, Path):
        output_file = str(output_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if not input_file.endswith(".c"):
        raise ValueError("Input file should be a .c file, but got: ", input_file)

    if no_compile:
        cmd = [
            "gcc",
            "-fsyntax-only",
            "-O0",
            "-Wno-error",
            "-Wno-return-type",
            "-Wno-invalid-pch",
            "-Wno-unused-parameter",
            "-fno-common",
            "-fno-strict-aliasing",
            input_file
        ]
    else:
        if not keep_debug_info:
            cmd = [
                "gcc",
                "-c",                         # generate only .o file
                # "-g",                         # include debug info
                "-O0",                        # no optimization
                # "-fno-diagnostics-show-caret",# disable error location display
                "-fdiagnostics-format=json",  # output diagnostics in json format
                "-Wno-error",                 # prevent warnings becoming errors  
                "-Wno-invalid-pch",          # ignore precompiled header errors
                "-Wno-return-type",         # ignore missing return type warnings
                "-Wno-unused-parameter",    # ignore unused parameter warnings
                "-fno-common",               # allow duplicate definitions
                "-fno-strict-aliasing",      # ignore strict aliasing violations
                input_file,
                "-o",
                output_file
            ]
        else:
            cmd = [
                "gcc",
                "-c",                         # generate only .o file
                "-g",                         # include debug info
                "-O0",                        # no optimization
                # "-fno-diagnostics-show-caret",# disable error location display
                "-fdiagnostics-format=json",  # output diagnostics in json format
                "-Wno-error",                 # prevent warnings becoming errors
                "-Wno-invalid-pch",          # ignore precompiled header errors
                "-Wno-return-type",         # ignore missing return type warnings
                "-Wno-unused-parameter",    # ignore unused parameter warnings
                "-fno-common",               # allow duplicate definitions
                "-fno-strict-aliasing",      # ignore strict aliasing violations
                input_file,
                "-o",
                output_file
            ]

    res = run_cmd(cmd, timeout=timeout)
    return res


def format_name(name): # type: (str) -> str
    """Format a name to be a legal C identifier (only 0-9a-zA-Z_)."""
    # Replace any non-alphanumeric chars with underscore
    formatted = ''.join(c if c.isalnum() else '_' for c in name)
    # Ensure name doesn't start with a number
    if formatted and formatted[0].isdigit():
        formatted = '_' + formatted
    return formatted


def do_hash(input: str, hash_type: str="sha256") -> str:
    match hash_type:
        case "sha256":
            return hashlib.sha256(input.encode()).hexdigest()
        case "md5":
            return hashlib.md5(input.encode()).hexdigest()
        case _:
            logger.warning(f"Unsupported hash type: {hash_type}, defaulting to sha256.")
            return hashlib.sha256(input.encode()).hexdigest()

       
def do_hash_file(file_path: Union[str, Path], hash_type:str="sha256") -> str:
    """Compute the hash of a file's contents."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found.")
    
    hash_func = None
    match hash_type:
        case "sha256":
            hash_func = hashlib.sha256()
        case "md5":
            hash_func = hashlib.md5()
        case _:
            logger.warning(f"Unsupported hash type: {hash_type}, defaulting to sha256.")
            hash_func = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def set_env(var: str, value: str, override: bool = True) -> None:
    """
    Set an environment variable to a specified value.

    Args:
        var (str): The name of the environment variable to set.
        value (str): The value to assign to the environment variable.
        override (bool): Whether to override the variable if it already exists. Defaults to True.

    Note:
        This function will set the environment variable unless override is False
        and the variable already exists.
    """
    if override or var not in os.environ:
        os.environ[var] = value


def looks_like_int(val: str) -> bool:
    """
    Check if the given string can be safely converted to an integer.

    Args:
        val (str): The string to test.

    Returns:
        bool: True if the string represents an integer, False otherwise.
    """
    try:
        int(val)
        return True
    except Exception as e:
        logger.error(f"Value {val} cannot be converted to int: {e}")
        return False
      
      
def byte_lines_to_str(lines: List[bytes]) -> List[str]:
    return [line.decode('utf-8', errors='ignore') for line in lines]
            

def bytes_to_string(b: Union[bytes, str],
                    encoding: str = 'utf-8') -> str:
    if isinstance(b, bytes) or isinstance(b, bytearray):
        return b.decode(encoding)
    elif isinstance(b, memoryview):
        return b.tobytes().decode(encoding)
    else:
        # Return as-is if already a string
        return b


def find_free_port() -> int:
    """
    Find a free port on localhost.
    NOTE: This function does NOT ensure thread-safety, because the lock should be locked 
        on a transaction basis (i.e., find a free port and start a server on it immediately).
        So the caller should ensure thread-safety if needed.
    
    Returns:
        int: A free port number.
    """
    # with _port_allocation_lock:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    

def kill_spawn(proc: pexpect.spawn) -> bool:
    """Kill a pexpect.spawn process."""
    try:
        if proc.isalive():
            proc.sendintr()
            time.sleep(1)
            if not proc.isalive():
                return True
            
            proc.terminate(force=True)
            time.sleep(1)
            if not proc.isalive():
                return True
    except Exception as e:
        logger.error(f"Exception occurred while killing process with PID {proc.pid}: {e}")
    return False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string
    into another.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The Levenshtein distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


