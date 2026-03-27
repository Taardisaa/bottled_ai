#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3.10"
export PYTHONHOME="$SCRIPT_DIR/python310"
echo "Python: $PYTHON" >&2
echo "Version: $($PYTHON --version 2>&1)" >&2
echo "yaml check: $($PYTHON -c 'import yaml; print("OK")' 2>&1)" >&2
exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
