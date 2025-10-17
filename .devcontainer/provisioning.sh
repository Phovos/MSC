#!/bin/bash
FILE=$1

echo "Initializing Python bridge..."

if [ -f "./scripts/appBridge.py" ] && [ -f "pyproject.toml" ]; then
  VERSION=$(python3 ./scripts/appBridge.py)
  echo "Detected version: $VERSION"
else
  echo "[WARN] pyproject.toml or appBridge.py not found. Skipping version extraction."
fi

echo "[PYTHON] Sending version $version to Python"

# Assuming the version is sent as a JSON object
echo "{\"version\": \"$version\"}" | python3 -u main.py

# "portsAttributes": {
#   "8000": { "label": "App Server", "onAutoForward": "openBrowser" },
#   "8888": { "label": "Jupyter", "onAutoForward": "notify" }
# }

sleep 2
# LSP server check
PORT=2087
if nc -z localhost $PORT; then
  echo "pylsp server already listening on $PORT"
else
  echo "pylsp not running yet, will try to start"
fi
sleep 2
curl -v http://localhost:2087 || echo "pylsp not responding"
