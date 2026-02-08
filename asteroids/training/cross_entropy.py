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
from cmath import e
from datetime import datetime
from decimal import DefaultContext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add parent to path for asteroids imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asteroids.ai.neural import NNAIParameters


class ModelWrap(nn.Module):
    def __init__(self, model):
        # model is a tensor
        super().__init__()
        self.model = model

    def forward(self, x, y=None):
        logits = self.model(x)
        if y is not None:
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
        return logits, loss


class DataLoader:
    def __init__(self, base_name, logger, batch_size, device="cpu"):
        self.base_name = base_name
        self.logger = logger
        self.batch_size = batch_size
        self.device = device

        # Find all matching files
        pattern = f"data/{base_name}_*.npz"
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise FileNotFoundError(
                f"No data files found matching pattern: {pattern}\n"
                f"Please ensure data files exist in data/ directory."
            )

        logger.info(f"Found {len(self.files)} data files matching '{base_name}'")

        self.data = []
        self.file_index = 0
        self.epoch_number = 0
        self.load_data()

    def load_data(self):
        file = self.files[self.file_index]
        try:
            self.data = np.load(file)
        except Exception as e:
            self.logger.error(f"  Failed to load {file}: {e}")
            raise
        self.batch_index = 0

    def get_batch(self):
        if self.batch_index + self.batch_size >= len(self.data["states"]):
            self.file_index += 1
            if self.file_index >= len(self.files):
                self.file_index = 0
                self.epoch_number += 1
            self.load_data()
        start = self.batch_index * self.batch_size
        end = start + self.batch_size
        states = self.data["states"][start:end]
        labels = self.data["actions"][start:end]

        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        labels = torch.from_numpy(labels).long().to(self.device)

        self.batch_index += 1
        return states, labels


def setup_logging(base_name):
    """
    Set up dual logging (console + file).

    Args:
        base_name: Base name for log file

    """
    # Create nn_checkpoints directory
    os.makedirs("nn_checkpoints", exist_ok=True)

    log_file = os.path.join("nn_checkpoints", f"{base_name}_cross_entropy.log")

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

    return logger


def train_model(
    base_name,
    batch_size,
    learning_rate,
    epochs,
    print_interval,
    eval_interval,
    games_per_eval,
    device,
):
    """
    Train model using cross-entropy loss on supervised data.

    Args:
        base_name: Base name for data files and output files
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    logger = setup_logging(base_name)

    logger.info(f"Cross-Entropy Supervised Learning")
    logger.info(f"Base name: {base_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("")

    logger.info("Loading training data...")
    data_loader = DataLoader(base_name, logger, batch_size, device)
    raw_model = NNAIParameters(device)
    logger.info(f"Model architecture: {raw_model.model}")
    model_wrap = ModelWrap(raw_model.model)
    logger.info(f"Model parameters: {model_wrap.parameters()}")
    optimizer = torch.optim.AdamW(model_wrap.parameters(), lr=learning_rate)

    iter = 0
    while data_loader.epoch_number < epochs:
        # starting with 1 is off by one, but prevents printing or eval on zero
        iter += 1
        if iter % eval_interval == 0:
            # run games-per-eval games and print the average score
            pass

        states, labels = data_loader.get_batch()
        logits, loss = model_wrap(states, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % print_interval == 0:
            logger.info(
                f"Epoch {data_loader.epoch_number}, Iteration {iter}, Loss: {loss.item()}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network using supervised learning with cross-entropy loss"
    )

    # Required arguments
    parser.add_argument(
        "--base-name",
        type=str,
        default="training_data20k_combinded",
        help="Base name for data files (data/<base_name>_*.npz) and output files",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--print-interval",
        type=int,
        default=100,
        help="Number of iterations between prints (default: 100)",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Number of iterations between evaluations (default: 100)",
    )

    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=100,
        help="Number of games to play per evaluation (default: 100)",
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
        print_interval=args.print_interval,
        eval_interval=args.eval_interval,
        games_per_eval=args.games_per_eval,
        device=args.device,
    )


if __name__ == "__main__":
    main()
