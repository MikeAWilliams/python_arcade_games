"""
Cross-entropy supervised learning for Asteroids AI.

User implements: Training loop, loss function, model architecture
Agent provides: Argument parsing, logging, file I/O scaffolding
"""

import argparse
import glob
import logging
import math
import os
import sys
import time
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

from asteroids.ai.neural import NNAIParameters, validate_and_load_model
from asteroids.core.game import Action


class ModelWrap(nn.Module):
    def __init__(self, model):
        # model is a tensor
        super().__init__()
        self.model = model

    def forward(self, x, y=None):
        logits = self.model(x)
        if y is not None:
            # y is one-hot encoded, so compute cross-entropy manually
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.mean(torch.sum(y * log_probs, dim=1))
        else:
            loss = None
        return logits, loss


class DataLoader:
    def __init__(self, base_name, logger, batch_per_file, device="cpu"):
        self.base_name = base_name
        self.logger = logger
        self.batch_per_file = batch_per_file
        self.batch_size = 1
        self.device = device
        self.num_actions = len(Action)

        # Find all matching files
        pattern = f"data/{base_name}_*.npz"
        # Sort numerically by the number in filename (not alphabetically)
        self.files = sorted(
            glob.glob(pattern), key=lambda x: int(Path(x).stem.split("_")[-1])
        )

        if not self.files:
            raise FileNotFoundError(
                f"No data files found matching pattern: {pattern}\n"
                f"Please ensure data files exist in data/ directory."
            )

        logger.info(f"Found {len(self.files)} data files matching '{base_name}'")

        self.states = None
        self.actions = None
        self.file_index = 0
        self.epoch_number = 0
        self.load_data()

    def load_data(self):
        file = self.files[self.file_index]
        try:
            raw = np.load(file)
            self.states = raw["states"]
            self.actions = raw["actions"]
            self.batch_size = len(self.states) // self.batch_per_file
        except Exception as e:
            self.logger.error(f"  Failed to load {file}: {e}")
            raise
        self.batch_index = 0

    def get_batch(self):
        if self.batch_index * self.batch_size + self.batch_size >= len(self.states):
            self.file_index += 1
            if self.file_index >= len(self.files):
                self.file_index = 0
                self.epoch_number += 1
            self.load_data()
        start = self.batch_index * self.batch_size
        end = start + self.batch_size
        states = self.states[start:end]
        labels = self.actions[start:end]

        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        labels = torch.from_numpy(labels).long().to(self.device)

        # Convert labels to one-hot encoding
        labels = F.one_hot(labels, num_classes=self.num_actions).float()

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


def evaluate_model(raw_model, games_per_eval, eval_threads, width=1280, height=720):
    """
    Evaluate current model by running games in parallel.

    Args:
        raw_model: NNAIParameters instance with trained weights
        games_per_eval: Number of games to run
        eval_threads: Number of worker processes
        width/height: Game dimensions

    Returns:
        tuple of (avg_score, avg_time_alive)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from asteroids.core.game_runner import run_single_game

    # Serialize model for workers (must be on CPU)
    model_state_dict = {k: v.cpu() for k, v in raw_model.model.state_dict().items()}
    eval_params = NNAIParameters(device="cpu")
    validate_and_load_model(
        eval_params.model, model_state_dict, source_description="training checkpoint"
    )
    eval_params.model.eval()

    # Prepare game arguments
    game_args = [
        (game_id, width, height, "neural", None, eval_params, None)
        for game_id in range(games_per_eval)
    ]

    # Run games in parallel
    results = []
    with ProcessPoolExecutor(max_workers=eval_threads) as executor:
        futures = [executor.submit(run_single_game, args) for args in game_args]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Compute averages
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_time_alive = sum(r["time_alive"] for r in results) / len(results)

    return avg_score, avg_time_alive


def train_model(
    base_name,
    batch_per_file,
    learning_rate,
    max_iterations,
    print_interval,
    checkpoint_interval,
    eval_interval,
    games_per_eval,
    eval_threads,
    device,
):
    """
    Train model using cross-entropy loss on supervised data.

    Args:
        base_name: Base name for data files and output files
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        max_iterations: Number of training max_iterations
        device: Device to train on ('cpu' or 'cuda')
    """
    logger = setup_logging(base_name)

    logger.info(f"Cross-Entropy Supervised Learning")
    logger.info(f"Base name: {base_name}")
    logger.info(f"Batch per file: {batch_per_file}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"max_iterations: {max_iterations}")
    logger.info(f"Evaluation interval: {eval_interval}")
    logger.info(f"Games per evaluation: {games_per_eval}")
    logger.info(f"Evaluation threads: {eval_threads}")
    logger.info(f"Device: {device}")
    logger.info("")

    logger.info("Loading training data...")
    data_loader = DataLoader(base_name, logger, batch_per_file, device)
    raw_model = NNAIParameters(device)
    model_wrap = ModelWrap(raw_model.model)
    optimizer = torch.optim.AdamW(model_wrap.parameters(), lr=learning_rate)

    logger.info("starting training")
    start_time = time.time()
    iter = 0
    while iter < max_iterations:
        if iter % eval_interval == 0:
            avg_score, avg_time_alive = evaluate_model(
                raw_model=raw_model,
                games_per_eval=games_per_eval,
                eval_threads=eval_threads,
            )

            logger.info(
                f"Evaluation avg_score={avg_score:.1f}, avg_time_alive={avg_time_alive:.1f}s"
            )

        states, labels = data_loader.get_batch()
        _, loss = model_wrap(states, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter and iter % checkpoint_interval == 0:
            torch.save(
                raw_model.model.state_dict(),
                f"nn_checkpoints/{base_name}_checkpoint_{iter}.pth",
            )

        if iter % print_interval == 0:
            elapsed_time = time.time() - start_time
            time_per_iter = elapsed_time / (iter + 1)
            estimated_total = time_per_iter * max_iterations
            time_remaining = estimated_total - elapsed_time

            logger.info(
                f"Epoch {data_loader.epoch_number}, "
                f"Iteration {iter}/{max_iterations}, Loss: {loss.item():.4f} | "
                f"Elapsed: {elapsed_time:.1f}s, Per-iter: {time_per_iter:.3f}s, "
                f"Remaining: {time_remaining:.1f}s, Total: {estimated_total:.1f}s"
            )
            if math.isnan(loss.item()):
                logger.error("Loss is NaN, terminating")
                sys.exit(1)
        iter += 1

    # we made it
    elapsed_time = time.time() - start_time
    time_per_iter = elapsed_time / (iter + 1)
    logger.info(
        "Training completed successfully, "
        f"Elapsed: {elapsed_time:.1f}s, Per-iter: {time_per_iter:.3f}s"
    )

    torch.save(
        raw_model.model.state_dict(),
        f"nn_checkpoints/{base_name}_final.pth",
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
        "--batch-per-file",
        type=int,
        default=3,
        help="Divide each file into this number of batches (default: 3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=400,  # 400 for now as default batch per file is 3*115(files)=345
        help="Number of training iterations (default: 400)",
    )

    parser.add_argument(
        "--print-interval",
        type=int,
        default=10,
        help="Number of iterations between prints (default: 10)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Number of iterations between checkpoints (default: 50)",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Number of iterations between evaluations (default: 10)",
    )

    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=100,
        help="Number of games to play per evaluation (default: 100)",
    )

    parser.add_argument(
        "--eval-threads",
        type=int,
        default=os.cpu_count() // 2 if os.cpu_count() else 2,
        help=f"Number of threads for evaluation (default: half of CPU cores)",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cuda)",
    )

    args = parser.parse_args()

    # Run training
    train_model(
        base_name=args.base_name,
        batch_per_file=args.batch_per_file,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        print_interval=args.print_interval,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        games_per_eval=args.games_per_eval,
        eval_threads=args.eval_threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
