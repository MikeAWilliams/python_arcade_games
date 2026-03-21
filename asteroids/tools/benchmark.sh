#!/usr/bin/env bash
# Benchmark a polar2 model via headless runs.
# Usage:
#   tools/benchmark.sh nn_weights/model.pth          # 1000 games
#   tools/benchmark.sh nn_weights/model.pth 500      # 500 games
#   tools/promote_checkpoint.sh | tools/benchmark.sh  # pipe from promote
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${1:-}"
N="${2:-1000}"

# Read model path from stdin if not provided as argument
if [ -z "$MODEL_PATH" ] && [ ! -t 0 ]; then
    MODEL_PATH=$(cat)
fi

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: benchmark.sh <model-path> [n-games]" >&2
    echo "       promote_checkpoint.sh | benchmark.sh" >&2
    exit 1
fi

cd "$PROJECT_DIR"
python main_headless.py -n "$N" --ai-type polar2 --model-path "$MODEL_PATH"
