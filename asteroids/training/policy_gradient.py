"""
Trains a Neural Network AI Input Method using policy gradient
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Add parent directory to path so we can import asteroids package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asteroids.core.game import Action, Game
from asteroids.ai.neural import NNAIInputMethod, NNAIParameters, validate_and_load_model
from asteroids.core.game_runner import execute_action


def setup_logging(log_dir="nn_checkpoints"):
    """
    Set up dual logging (console + file).
    Returns: logger instance, timestamp for filenames
    """
    # Create directory if needed
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Simple format (no logger name prefix)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, timestamp


def discounted_rewards(rewards, gamma=0.99, normalize=True):
    eps = 0.0001
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret - np.mean(ret)) / (np.std(ret) + eps)
    return ret


def train_on_game_results(model, optimizer, states, actions, advantages, device):
    """
    Train model using REINFORCE policy gradient.

    Args:
        states: (N, state_dim) array of states
        actions: (N,) array of action indices taken
        advantages: (N,) array of advantage values (discounted rewards)
    """
    # Convert to tensors
    states = torch.from_numpy(states).float().to(device)
    actions = torch.from_numpy(actions).long().to(device)
    advantages = torch.from_numpy(advantages).float().to(device)

    optimizer.zero_grad()

    # Forward pass - get action logits
    logits = model(states)  # (N, num_actions)

    # Get log probability of actions that were actually taken
    log_probs = F.log_softmax(logits, dim=1)
    log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Policy gradient loss: -E[log π(a|s) * A(s,a)]
    loss = -torch.mean(log_probs * advantages)

    # Backpropagate
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def run_games_batch_worker(args):
    """
    Run multiple games sequentially in a single worker process.
    This reduces overhead by reusing the model and minimizing task submissions.
    Also computes discounted rewards in parallel to avoid main process bottleneck.

    Args:
        args: Tuple of (worker_id, num_games, width, height, model_state_dict)

    Returns:
        List of dicts, one per game with states, actions, discounted_rewards, score
    """
    worker_id, num_games, width, height, model_state_dict = args

    # Create parameters once and reuse for all games in this worker
    params = NNAIParameters(device="cpu")
    validate_and_load_model(
        params.model, model_state_dict, source_description="training checkpoint"
    )
    params.model.eval()

    results = []
    for game_id in range(num_games):
        # Run game
        game = Game(width, height)
        input_method = NNAIInputMethod(game=game, parameters=params, keep_data=True)
        dt = 1 / 60

        while game.player_alive:
            game.clear_turn()
            game.clear_acc()
            action = input_method.get_move()
            execute_action(game, action)
            game.update(dt)

        # Collect data
        states = np.array(input_method.states)
        actions = np.array(input_method.actions_taken)
        # Note: probs no longer needed for REINFORCE
        rewards = np.diff(input_method.scores, prepend=0)

        # Add small survival bonus per frame (0.001 per frame)
        # Heuristic AI: ~149s * 60fps = ~8960 frames → ~8.96 total bonus
        # Max score: ~236, so survival is ~3.8% of max score (keeps score primary)
        survival_bonus = 0.001
        rewards = rewards + survival_bonus

        # Compute discounted rewards HERE in the worker (parallel!)
        rewards_reshaped = np.vstack(rewards)
        dr = discounted_rewards(rewards_reshaped)

        results.append(
            {
                "worker_id": worker_id,
                "game_id": game_id,
                "states": states,
                "actions": actions,
                # probs removed - not needed for REINFORCE
                "discounted_rewards": dr,  # Already computed
                "score": np.sum(rewards),
            }
        )

    return results


def run_games_parallel(
    width, height, model_state_dict, batch_size, num_workers, executor
):
    """
    Run multiple games in parallel using a persistent ProcessPoolExecutor.
    Distributes games across workers to minimize task overhead.
    Returns concatenated training data from all games.

    Args:
        executor: Persistent ProcessPoolExecutor to reuse across epochs
    """
    # Distribute games across workers
    games_per_worker = batch_size // num_workers
    extra_games = batch_size % num_workers

    # Prepare arguments for each worker
    worker_args = []
    for worker_id in range(num_workers):
        # Give extra games to first few workers
        num_games = games_per_worker + (1 if worker_id < extra_games else 0)
        worker_args.append((worker_id, num_games, width, height, model_state_dict))

    all_states = []
    all_actions = []
    all_discounted_rewards = []
    total_score = 0

    # Submit to persistent executor
    futures = [executor.submit(run_games_batch_worker, args) for args in worker_args]

    for future in as_completed(futures):
        worker_results = future.result()  # List of game results from this worker

        for result in worker_results:
            # Discounted rewards already computed in worker!
            all_states.append(result["states"])
            all_actions.append(result["actions"])
            all_discounted_rewards.append(result["discounted_rewards"])
            total_score += result["score"]

    # Concatenate all games into single batches
    states_batch = np.concatenate(all_states, axis=0)
    actions_batch = np.concatenate(all_actions, axis=0)
    discounted_rewards_batch = np.concatenate(all_discounted_rewards, axis=0)

    return (
        states_batch,
        actions_batch,
        discounted_rewards_batch,
        total_score,
    )


def train_model(width, height, batch_size=32, num_workers=None):
    # Set up logging
    logger, timestamp = setup_logging()

    # Detect device (GPU if available, else CPU)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    device = "cpu"
    logger.info(f"Using device: {device}")

    # Default to all CPU cores
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    logger.info(f"Using {num_workers} worker processes for game simulation")
    logger.info(f"Batch size: {batch_size} games per training update")

    params = NNAIParameters(device=device)
    model = params.model
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    # alpha removed - not needed for REINFORCE
    max_score = 0
    total_epochs = 60000
    print_frequency = 500
    intermediate_save_frequency = total_epochs / 10
    start_time = time.time()

    # Create persistent process pool to avoid recreation overhead
    logger.info("Creating persistent worker pool...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for epoch in range(total_epochs):
            # Get model state dict for subprocess (CPU version)
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            # Run games in parallel using persistent executor
            sim_start = time.time()
            states, actions, dr, total_score = run_games_parallel(
                width, height, model_state_dict, batch_size, num_workers, executor
            )
            sim_time = time.time() - sim_start

            avg_score = total_score / batch_size
            if avg_score > max_score:
                max_score = avg_score

            # Training computation
            train_start = time.time()
            # Advantages are the discounted rewards (already computed per game)
            # Shape: actions is (N,), dr is (N, 1) -> squeeze to (N,)
            advantages = dr.squeeze()
            loss = train_on_game_results(
                model, opt, states, actions, advantages, device
            )
            train_time = time.time() - train_start
            if epoch % intermediate_save_frequency == 0:
                checkpoint_path = os.path.join(
                    "nn_checkpoints", f"checkpoint_{timestamp}_epoch_{epoch}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "max_score": float(max_score),
                        "loss": loss,
                    },
                    checkpoint_path,
                )
            if epoch % print_frequency == 0:
                elapsed_time = time.time() - start_time
                progress = (epoch + 1) / total_epochs
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                estimated_remaining_time = estimated_total_time - elapsed_time

                # Format times as HH:MM:SS
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                total_str = time.strftime("%H:%M:%S", time.gmtime(estimated_total_time))
                remaining_str = time.strftime(
                    "%H:%M:%S", time.gmtime(estimated_remaining_time)
                )

                logger.info(
                    f"{epoch}/{total_epochs} -> avg_score:{avg_score:.2f}, max:{max_score:.2f}, loss:{loss:.4f} | "
                    f"sim:{sim_time:.2f}s, train:{train_time:.2f}s | "
                    f"elapsed:{elapsed_str}, total:{total_str}, remaining:{remaining_str}"
                )

    # Save the trained model
    final_checkpoint_path = os.path.join(
        "nn_checkpoints", f"checkpoint_{timestamp}_final.pth"
    )
    torch.save(
        {
            "epoch": total_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "max_score": max_score,
            "loss": loss,
        },
        final_checkpoint_path,
    )
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    logger.info(f"Training completed in {total_time_str}")
    logger.info(f"Model saved to {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural Network AI using Policy Gradient"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Game width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Game height (default: 720)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of games to run per training update (default: 32)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: {os.cpu_count() or 4})",
    )
    args = parser.parse_args()
    train_model(args.width, args.height, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
