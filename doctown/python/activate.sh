#!/bin/bash
# Quick activation script for the Python virtual environment
# Usage: source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please run ./setup_venv.sh first to create it."
    return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo "Python: $(which python)"
echo "To deactivate, run: deactivate"
