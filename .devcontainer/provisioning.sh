#!/bin/bash
FILE=$1

echo "Initializing Python bridge..."

jq -c '.repository[]' "$FILE" | while read -r instance; do
  id=$(echo "$instance" | jq -r '.id')
  echo "[PYTHON] Sending instance $id to Python"

  echo "$instance" | python3 -u main.py
done
