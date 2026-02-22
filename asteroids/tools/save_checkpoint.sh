#!/bin/bash
# Copy the latest bearing_weighted checkpoint to nn_weights with a percent-complete name.
# Usage: ./tools/save_checkpoint.sh

TOTAL_ITERATIONS=56250000
CHECKPOINT_DIR="nn_checkpoints"
OUTPUT_DIR="nn_weights"
RUN_NAME="bearing_weighted"      # matches TRAINING_RUN_NAME in cross_entropy.py
OUTPUT_NAME="bearing_weighted"   # base name for the output file in nn_weights

latest=$(ls -t "$CHECKPOINT_DIR"/${RUN_NAME}_checkpoint_*.pth 2>/dev/null | head -1)

if [ -z "$latest" ]; then
    echo "No ${RUN_NAME} checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

filename=$(basename "$latest")
iter=$(echo "$filename" | sed "s/${RUN_NAME}_checkpoint_\([0-9]*\)\.pth/\1/")
pct=$(( (iter * 100 + TOTAL_ITERATIONS / 2) / TOTAL_ITERATIONS ))

output="$OUTPUT_DIR/${OUTPUT_NAME}_$(printf '%03d' $pct).pth"

echo "Checkpoint : $latest"
echo "Iteration  : $iter / $TOTAL_ITERATIONS"
echo "Completion : $pct%"
echo "Output     : $output"

cp "$latest" "$output"
echo "Done."
