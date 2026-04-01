#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"
export PYTHONHOME="$SCRIPT_DIR/python311"
export LD_LIBRARY_PATH="$SCRIPT_DIR/python311/lib:${LD_LIBRARY_PATH:-}"
DEBUG_PORT=11451

if [ ! -f "$PYTHON" ]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

echo "Python: $PYTHON" >&2
echo "Version: $($PYTHON --version 2>&1)" >&2
echo "Debugpy: listen $DEBUG_PORT, wait-for-client" >&2

exec "$PYTHON" -m debugpy --listen "$DEBUG_PORT" --wait-for-client "$SCRIPT_DIR/main.py" "$@"
