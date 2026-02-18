#!/bin/bash
set -e

echo "Setting up ant-coding project..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install project with dev dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"

echo "Setup complete. You can now run tests with: PYTHONPATH=src pytest"
