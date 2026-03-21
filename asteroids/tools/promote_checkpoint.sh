#!/usr/bin/env bash
# Copy the most recently modified .pth file from nn_checkpoints/ to nn_weights/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$PROJECT_DIR/nn_checkpoints"
DST="$PROJECT_DIR/nn_weights"

GLOB="*.pth"
if [ "${1:-}" = "--best" ]; then
    GLOB="*best*.pth"
fi

latest=$(ls -t "$SRC"/$GLOB 2>/dev/null | head -1)

if [ -z "$latest" ]; then
    echo "No matching .pth files found in $SRC"
    exit 1
fi

filename=$(basename "$latest")
cp "$latest" "$DST/$filename"
echo "Copied $filename -> nn_weights/" >&2
echo "$filename"
