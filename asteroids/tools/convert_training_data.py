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
import logging
import os
import sys
from pathlib import Path

import numpy as np


def setup_logging(output_base: str):
    """
    Set up dual logging (console + file).

    Log file saved to data directory as <output_base>_convert.log.
    """
    os.makedirs("data", exist_ok=True)

    log_file = f"data/{output_base}_convert.log"

    logger = logging.getLogger("convert_training_data")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Simple format
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def convert_file(input_path: str, output_path: str, logger) -> bool:
    """
    Convert a single NPZ file from old angle format to new bearing format.

    Args:
        input_path: Path to input NPZ file
        output_path: Path to output NPZ file
        logger: Logger instance

    Returns:
        True if successful, False if an error occurred
    """
    try:
        raw = np.load(input_path)
        states = raw["states"]

        if states.shape[1] != 141:
            logger.warning(
                f"  {input_path} has {states.shape[1]} columns (expected 141), skipping"
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
        logger.error(f"  Failed to convert {input_path}: {e}")
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

    logger = setup_logging(args.output_base)

    pattern = f"data/{args.input_base}_*.npz"
    files = sorted(
        glob.glob(pattern), key=lambda x: int(Path(x).stem.split("_")[-1])
    )

    if not files:
        logger.error(f"No data files found matching: {pattern}")
        sys.exit(1)

    logger.info(f"Found {len(files)} files matching '{args.input_base}'")

    success_count = 0
    fail_count = 0

    for file in files:
        suffix = Path(file).stem.split("_")[-1]
        output_path = f"data/{args.output_base}_{suffix}.npz"
        logger.info(f"Converting {file} -> {output_path}")

        if convert_file(file, output_path, logger):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\nDone: {success_count} converted, {fail_count} failed")


if __name__ == "__main__":
    main()
