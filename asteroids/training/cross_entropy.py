"""
Cross-entropy supervised learning for Asteroids AI.

User implements: Training loop, loss function, model architecture
Agent provides: Argument parsing, logging, file I/O scaffolding
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent to path for asteroids imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_logging(base_name):
    """
    Set up dual logging (console + file).

    Args:
        base_name: Base name for log file

    Returns:
        logger instance, timestamp string
    """
    # Create nn_checkpoints directory
    os.makedirs("nn_checkpoints", exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("nn_checkpoints", f"{base_name}_{timestamp}.log")

    # Configure logger (same pattern as policy_gradient.py)
    logger = logging.getLogger("cross_entropy")
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

    return logger, timestamp


def load_training_data(base_name, logger):
    """
    Load training data from data/<base_name>_*.npz files.

    Args:
        base_name: Base name pattern for data files
        logger: Logger instance

    Returns:
        List of loaded numpy archives (or processed data)

    Raises:
        FileNotFoundError: If no matching files found
    """
    # Find all matching files
    pattern = f"data/{base_name}_*.npz"
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No data files found matching pattern: {pattern}\n"
            f"Please ensure data files exist in data/ directory."
        )

    logger.info(f"Found {len(files)} data files matching '{base_name}'")

    # Load files
    loaded_data = []
    for filepath in files:
        try:
            data = np.load(filepath)
            loaded_data.append(data)
            logger.info(f"  Loaded: {filepath}")
        except Exception as e:
            logger.error(f"  Failed to load {filepath}: {e}")
            raise

    logger.info(f"Successfully loaded {len(loaded_data)} files")

    # TODO: User implements data processing/concatenation
    return loaded_data


def train_model(base_name, batch_size, learning_rate, epochs, device):
    """
    Train model using cross-entropy loss on supervised data.

    Args:
        base_name: Base name for data files and output files
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    # Set up logging
    logger, timestamp = setup_logging(base_name)

    logger.info(f"Cross-Entropy Supervised Learning")
    logger.info(f"Base name: {base_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Load training data
    logger.info("Loading training data...")
    data = load_training_data(base_name, logger)

    # TODO: User implements
    # - Data preprocessing and batching
    # - Model architecture definition
    # - Loss function (cross-entropy)
    # - Optimizer setup
    # - Training loop
    # - Checkpoint saving

    logger.info("TODO: Implement training loop")
    logger.info("  1. Preprocess data into batches")
    logger.info("  2. Define neural network model")
    logger.info("  3. Define cross-entropy loss function")
    logger.info("  4. Create optimizer (e.g., Adam)")
    logger.info("  5. Implement training loop:")
    logger.info("     - Forward pass")
    logger.info("     - Compute loss")
    logger.info("     - Backward pass")
    logger.info("     - Optimizer step")
    logger.info("  6. Save checkpoints periodically")
    logger.info(
        f"  7. Save final model to nn_checkpoints/{base_name}_{timestamp}_final.pth"
    )

    # Placeholder return
    logger.info("")
    logger.info("Training scaffolding complete. User should implement algorithm.")


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network using supervised learning with cross-entropy loss"
    )

    # Required arguments
    parser.add_argument(
        "--base-name",
        type=str,
        required=True,
        help="Base name for data files (data/<base_name>_*.npz) and output files",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cpu)",
    )

    args = parser.parse_args()

    # Run training
    train_model(
        base_name=args.base_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
