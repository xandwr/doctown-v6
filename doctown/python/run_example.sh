#!/bin/bash
# Helper script to run example scripts with proper environment setup

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if an example file was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_example.sh <example_file>"
    echo ""
    echo "Available examples:"
    for file in "$SCRIPT_DIR/examples"/*.py; do
        echo "  - $(basename "$file")"
    done
    exit 1
fi

EXAMPLE_FILE="$1"

# If the example file doesn't include the path, assume it's in examples/
if [[ ! "$EXAMPLE_FILE" =~ "/" ]]; then
    EXAMPLE_FILE="examples/$EXAMPLE_FILE"
fi

# Check if the file exists
if [ ! -f "$SCRIPT_DIR/$EXAMPLE_FILE" ]; then
    echo "Error: Example file not found: $EXAMPLE_FILE"
    exit 1
fi

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please run: ./setup_venv.sh"
    exit 1
fi

# Activate virtual environment and run the example
echo "Running: $EXAMPLE_FILE"
echo "========================================"
cd "$SCRIPT_DIR"
source venv/bin/activate
PYTHONPATH="$SCRIPT_DIR" python3 "$EXAMPLE_FILE"
