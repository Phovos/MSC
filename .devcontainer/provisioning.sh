#!/bin/bash
FILE=$1

echo "Initializing Python bridge..."

# Extract version using a Python script
version=$(python3 appBridge.py)

if [ -z "$version" ]; then
  echo "Error: Version not found in pyproject.toml."
  exit 1
fi

echo "[PYTHON] Sending version $version to Python"

# Assuming the version is sent as a JSON object
echo "{\"version\": \"$version\"}" | python3 -u main.py