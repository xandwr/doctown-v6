#!/bin/bash
# Setup script for Python virtual environment in doctown/python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up Python virtual environment for Doctown..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "Using $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "Dependencies installed successfully!"
else
    echo "Warning: requirements.txt not found at $SCRIPT_DIR/requirements.txt"
fi

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
