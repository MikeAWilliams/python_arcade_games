"""
Helper script to load and work with individual game recording files.

This script demonstrates how to:
1. Load individual game recordings
2. Combine multiple games
3. Filter and analyze the data
"""

import glob
import os

import numpy as np


def load_single_game(filename):
    """
    Load a single game recording file.

    Args:
        filename: Path to NPZ file

    Returns:
        Dictionary with arrays: game_ids, tick_nums, states, actions
    """
    data = np.load(filename)
    return {
        "game_ids": data["game_ids"],
        "tick_nums": data["tick_nums"],
        "states": data["states"],
        "actions": data["actions"],
    }


def save_combined(base_name, out_name, data_dir="./data", target_size=1000000):
    """
    Combine games into chunked NPZ files of approximately target_size recordings each.

    Args:
        base_name: Base name used during recording
        out_name: Base name of output files (creates out_name_0.npz, out_name_1.npz, etc.)
        data_dir: Directory containing the NPZ files
        target_size: Target number of recordings per output file
    """
    pattern = os.path.join(data_dir, f"{base_name}_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    print(f"Found {len(files)} game files to process")
    print(f"Target size per chunk: {target_size:,} recordings")
    print()

    chunk_index = 0
    current_game_ids = []
    current_tick_nums = []
    current_states = []
    current_actions = []
    current_size = 0
    total_recordings = 0
    chunk_start_game = 0

    for i, filename in enumerate(files):
        # Load game data
        data = np.load(filename)
        game_size = len(data["game_ids"])

        # Add to current chunk
        current_game_ids.append(data["game_ids"])
        current_tick_nums.append(data["tick_nums"])
        current_states.append(data["states"])
        current_actions.append(data["actions"])
        current_size += game_size

        # Check if we've reached target size
        if current_size >= target_size:
            # Save current chunk
            output_file = os.path.join(data_dir, f"{out_name}_{chunk_index}.npz")

            combined = {
                "game_ids": np.concatenate(current_game_ids),
                "tick_nums": np.concatenate(current_tick_nums),
                "states": np.concatenate(current_states),
                "actions": np.concatenate(current_actions),
            }

            np.savez_compressed(
                output_file,
                game_ids=combined["game_ids"],
                tick_nums=combined["tick_nums"],
                states=combined["states"],
                actions=combined["actions"],
            )

            num_games_in_chunk = len(current_game_ids)
            print(
                f"Chunk {chunk_index}: Saved {current_size:,} recordings to {os.path.basename(output_file)}"
            )
            print(
                f"  Games {chunk_start_game} to {chunk_start_game + num_games_in_chunk - 1} ({num_games_in_chunk} games)"
            )

            total_recordings += current_size
            chunk_index += 1
            chunk_start_game = i + 1

            # Reset for next chunk
            current_game_ids = []
            current_tick_nums = []
            current_states = []
            current_actions = []
            current_size = 0

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} games...")

    # Save remaining data if any
    if current_size > 0:
        output_file = os.path.join(data_dir, f"{out_name}_{chunk_index}.npz")

        combined = {
            "game_ids": np.concatenate(current_game_ids),
            "tick_nums": np.concatenate(current_tick_nums),
            "states": np.concatenate(current_states),
            "actions": np.concatenate(current_actions),
        }

        np.savez_compressed(
            output_file,
            game_ids=combined["game_ids"],
            tick_nums=combined["tick_nums"],
            states=combined["states"],
            actions=combined["actions"],
        )

        num_games_in_chunk = len(current_game_ids)
        print(
            f"Chunk {chunk_index}: Saved {current_size:,} recordings to {os.path.basename(output_file)}"
        )
        print(
            f"  Games {chunk_start_game} to {chunk_start_game + num_games_in_chunk - 1} ({num_games_in_chunk} games)"
        )

        total_recordings += current_size

    print()
    print(
        f"Done! Created {chunk_index + 1} chunk(s) with {total_recordings:,} total recordings"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and combine game recording files"
    )
    parser.add_argument(
        "--base-name", help="Base name of recordings (e.g., 'test_run')"
    )
    parser.add_argument(
        "--out-name", help="Base name of recordings (e.g., 'test_run_combined')"
    )
    parser.add_argument(
        "--target-size",
        default=10000000,
        type=int,
        help="Target size of combined file",
    )
    parser.add_argument(
        "--data-dir", default="./data", help="Directory containing NPZ files"
    )

    args = parser.parse_args()

    save_combined(args.base_name, args.out_name, args.data_dir, args.target_size)
