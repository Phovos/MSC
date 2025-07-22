#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Running post-create setup as user: $(whoami) ---"
# --- Python Environment Setup ---
# Create a virtual environment using uv and install packages from requirements.txt
# This makes your Python setup reproducible and ready-to-go.
echo "Setting up Python virtual environment with uv..."
uv venv .venv --python 3.13
# Use 'sync' to ensure the venv matches requirements.txt exactly.
# Create a requirements.txt file if you don't have one!
if [ -f "requirements.txt" ]; then
    uv pip sync requirements.txt -p .venv/bin/python
else
    echo "No requirements.txt found. Skipping Python package installation."
    echo "You can create one and run 'uv pip sync requirements.txt' later."
fi

echo "--- Setup complete! ---"