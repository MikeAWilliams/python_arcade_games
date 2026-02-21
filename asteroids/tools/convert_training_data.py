"""
Convert training data from old angle format to new bearing format.

Old format: column 4 is normalized angle (angle / 2*pi), 141 total columns.
New format: column 4 is cos(angle), column 5 is sin(angle), 142 total columns.

Usage:
    python tools/convert_training_data.py --input-base test_data --output-base test_data_v2
    python tools/convert_training_data.py --input-base training_data20k_combinded --output-base training_data20k_v2
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def convert_file(input_path: str, output_path: str) -> bool:
    """
    Convert a single NPZ file from old angle format to new bearing format.

    Args:
        input_path: Path to input NPZ file
        output_path: Path to output NPZ file

    Returns:
        True if successful, False if an error occurred
    """
    try:
        raw = np.load(input_path)
        states = raw["states"]

        if states.shape[1] != 141:
            print(
                f"  WARNING: {input_path} has {states.shape[1]} columns (expected 141), skipping"
            )
            return False

        # Column 4 is normalized angle: angle / (2*pi)
        angle_norm = states[:, 4]
        angle_rad = angle_norm * 2 * np.pi
        cos_col = np.cos(angle_rad).reshape(-1, 1)
        sin_col = np.sin(angle_rad).reshape(-1, 1)

        # Build new states: cols 0-3, cos, sin, cols 5-140
        new_states = np.concatenate(
            [states[:, :4], cos_col, sin_col, states[:, 5:]], axis=1
        )

        np.savez_compressed(
            output_path,
            states=new_states,
            actions=raw["actions"],
            game_ids=raw["game_ids"],
            tick_nums=raw["tick_nums"],
        )
        return True

    except Exception as e:
        print(f"  ERROR: Failed to convert {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert training data from angle format to bearing format"
    )
    parser.add_argument(
        "--input-base",
        type=str,
        required=True,
        help="Base name for input data files (data/<input_base>_*.npz)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        required=True,
        help="Base name for output data files (data/<output_base>_*.npz)",
    )
    args = parser.parse_args()

    pattern = f"data/{args.input_base}_*.npz"
    files = sorted(glob.glob(pattern), key=lambda x: int(Path(x).stem.split("_")[-1]))

    if not files:
        print(f"No data files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} files matching '{args.input_base}'")

    success_count = 0
    fail_count = 0

    for file in files:
        suffix = Path(file).stem.split("_")[-1]
        output_path = f"data/{args.output_base}_{suffix}.npz"
        print(f"Converting {file} -> {output_path}")

        if convert_file(file, output_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDone: {success_count} converted, {fail_count} failed")


if __name__ == "__main__":
    main()
