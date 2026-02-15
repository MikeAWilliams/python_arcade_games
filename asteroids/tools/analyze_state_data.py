"""
Analyze the range of state data columns in training data files.

Samples rows evenly from the first data file and computes per-column
statistics (min, max, mean, std). Column names are derived from the
state encoding in asteroids/ai/neural.py:compute_state().

Usage:
    python tools/analyze_state_data.py
    python tools/analyze_state_data.py --base-name training_data20k_combinded
    python tools/analyze_state_data.py --sample-size 100000
"""

import argparse
import glob
from pathlib import Path

import numpy as np


def build_column_names():
    """Build human-readable names for each of the 141 state columns."""
    cols = []
    cols.append("player_x (norm)")
    cols.append("player_y (norm)")
    cols.append("player_vx (norm)")
    cols.append("player_vy (norm)")
    cols.append("player_angle (norm)")
    cols.append("shoot_cooldown (norm)")

    for i in range(27):
        cols.append(f"ast_{i:02d}_x (norm)")
        cols.append(f"ast_{i:02d}_y (norm)")
        cols.append(f"ast_{i:02d}_vx (norm)")
        cols.append(f"ast_{i:02d}_vy (norm)")
        cols.append(f"ast_{i:02d}_active")

    return cols


def analyze_file(filepath, sample_size=500000):
    """Load a data file and compute per-column statistics on a sample."""
    raw = np.load(filepath)
    states = raw["states"]
    n = len(states)

    step = max(1, n // sample_size)
    sample = states[::step]

    return {
        "total_rows": n,
        "sampled_rows": len(sample),
        "step": step,
        "min": np.min(sample, axis=0),
        "max": np.max(sample, axis=0),
        "mean": np.mean(sample, axis=0),
        "std": np.std(sample, axis=0),
    }


def print_results(stats, cols):
    """Print a formatted table of per-column statistics."""
    print(
        f"Sampled {stats['sampled_rows']:,} rows from {stats['total_rows']:,} "
        f"total (every {stats['step']}th row)"
    )
    print()
    print(f"{'Col':>3}  {'Name':<26}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Std':>10}")
    print("-" * 80)

    for i, name in enumerate(cols):
        print(
            f"{i:3d}  {name:<26}  {stats['min'][i]:10.4f}  {stats['max'][i]:10.4f}  "
            f"{stats['mean'][i]:10.4f}  {stats['std'][i]:10.4f}"
        )

        # Print separator between player and asteroid sections
        if i == 5:
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze state data column ranges from training data"
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="training_data20k_combinded",
        help="Base name for data files (default: training_data20k_combinded)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500000,
        help="Target number of rows to sample (default: 500000)",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="Which data file to analyze (default: 0, the first file)",
    )
    args = parser.parse_args()

    pattern = f"data/{args.base_name}_*.npz"
    files = sorted(
        glob.glob(pattern), key=lambda x: int(Path(x).stem.split("_")[-1])
    )

    if not files:
        print(f"No data files found matching: {pattern}")
        return

    if args.file_index >= len(files):
        print(f"File index {args.file_index} out of range (found {len(files)} files)")
        return

    filepath = files[args.file_index]
    print(f"Analyzing: {filepath}")
    print()

    cols = build_column_names()
    stats = analyze_file(filepath, args.sample_size)
    print_results(stats, cols)


if __name__ == "__main__":
    main()
