"""
Trains a Neural Network AI Input Method using policy gradient
"""

# Training run name — controls log and checkpoint file names.
# Change this for each new training run.
TRAINING_RUN_NAME = "polar2_pg"

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Add parent directory to path so we can import asteroids package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asteroids.core.game import Action, Game
from asteroids.ai.raw_geometry_nn import (
    RawGeometryNNInputMethod,
    RawGeometryNNParameters,
    validate_and_load_model,
)
from asteroids.ai.polar_nn import PolarNNInputMethod, PolarNNParameters
from asteroids.ai.polar2_nn import Polar2NNInputMethod, Polar2NNParameters
from asteroids.core.game_runner import execute_action

MODEL_TYPES = {
    "raw": {
        "params_class": RawGeometryNNParameters,
        "input_class": RawGeometryNNInputMethod,
    },
    "polar": {
        "params_class": PolarNNParameters,
        "input_class": PolarNNInputMethod,
    },
    "polar2": {
        "params_class": Polar2NNParameters,
        "input_class": Polar2NNInputMethod,
    },
}


def setup_logging(run_name):
    """
    Set up dual logging (console + file).

    Args:
        run_name: Name for this training run (controls log and checkpoint filenames)
    """
    os.makedirs("nn_checkpoints", exist_ok=True)

    log_file = os.path.join("nn_checkpoints", f"{run_name}_policy_gradient.log")

    # Configure logger
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)

    # Simple format (no logger name prefix)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


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


def train_on_game_results(
    model, optimizer, states, actions, advantages, device, entropy_coeff=0.0
):
    """
    Train model using REINFORCE policy gradient with entropy bonus.

    Args:
        states: (N, state_dim) array of states
        actions: (N,) array of action indices taken
        advantages: (N,) array of advantage values (discounted rewards)
        entropy_coeff: weight for entropy bonus (encourages exploration)
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
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Entropy bonus: -sum(p * log(p)) per sample
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)

    # Policy gradient loss with entropy bonus
    # Subtract entropy term to encourage exploration (maximizing entropy)
    loss = -torch.mean(selected_log_probs * advantages) - entropy_coeff * torch.mean(
        entropy
    )

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
        args: Tuple of (worker_id, num_games, width, height, model_state_dict,
              model_type, death_penalty, death_penalty_frames)

    Returns:
        List of dicts, one per game with states, actions, discounted_rewards, score
    """
    (
        worker_id,
        num_games,
        width,
        height,
        model_state_dict,
        model_type,
        death_penalty,
        death_penalty_frames,
    ) = args

    # Create parameters once and reuse for all games in this worker
    model_info = MODEL_TYPES[model_type]
    params = model_info["params_class"](device="cpu")
    validate_and_load_model(
        params.model, model_state_dict, source_description="training checkpoint"
    )
    params.model.eval()

    results = []
    for game_id in range(num_games):
        # Run game
        game = Game(width, height)
        input_method = model_info["input_class"](
            game=game, parameters=params, keep_data=True
        )
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

        # Death penalty: penalize last N frames before death with ramping penalty
        if death_penalty != 0 and death_penalty_frames > 0:
            for i in range(
                max(0, len(rewards) - death_penalty_frames), len(rewards)
            ):
                decay = (
                    i - (len(rewards) - death_penalty_frames)
                ) / death_penalty_frames  # 0→1
                rewards[i] += death_penalty * decay

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
    width,
    height,
    model_state_dict,
    batch_size,
    num_workers,
    executor,
    model_type,
    death_penalty=0.0,
    death_penalty_frames=0,
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
        worker_args.append(
            (
                worker_id,
                num_games,
                width,
                height,
                model_state_dict,
                model_type,
                death_penalty,
                death_penalty_frames,
            )
        )

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


def train_model(
    width,
    height,
    batch_size=32,
    num_workers=None,
    model_type="raw",
    run_name=TRAINING_RUN_NAME,
    checkpoint=None,
    entropy_coeff=0.0,
    death_penalty=0.0,
    death_penalty_frames=60,
):
    # Set up logging
    logger = setup_logging(run_name)

    # Detect device (GPU if available, else CPU)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    device = "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {model_type}")

    # Default to all CPU cores
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    logger.info(f"Using {num_workers} worker processes for game simulation")
    logger.info(f"Batch size: {batch_size} games per training update")
    logger.info(f"Entropy coefficient: {entropy_coeff}")
    if death_penalty != 0:
        logger.info(
            f"Death penalty: {death_penalty} over last {death_penalty_frames} frames"
        )

    model_info = MODEL_TYPES[model_type]
    params = model_info["params_class"](device=device)
    model = params.model
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    max_score = 0

    # Load from checkpoint if provided and exists
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        validate_and_load_model(
            model, ckpt["model_state_dict"], source_description=checkpoint
        )
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "max_score" in ckpt:
            max_score = float(ckpt["max_score"])
        logger.info(
            f"Resumed from checkpoint: {checkpoint} (max_score: {max_score:.2f})"
        )
    elif checkpoint:
        logger.info(f"No checkpoint found at {checkpoint}, starting from scratch")
    total_epochs = 60000
    print_frequency = 500
    intermediate_save_frequency = total_epochs / 10
    timed_save_interval = 30 * 60  # 30 minutes
    last_timed_save = time.time()
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
                width,
                height,
                model_state_dict,
                batch_size,
                num_workers,
                executor,
                model_type,
                death_penalty,
                death_penalty_frames,
            )
            sim_time = time.time() - sim_start

            avg_score = total_score / batch_size
            if avg_score > max_score:
                max_score = avg_score
                best_path = os.path.join("nn_checkpoints", f"{run_name}_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "max_score": float(max_score),
                    },
                    best_path,
                )
                logger.info(
                    f"{epoch}/{total_epochs} -> NEW BEST avg_score:{avg_score:.2f} | sim:{sim_time:.2f}s"
                )

            # Training computation
            train_start = time.time()
            # Advantages are the discounted rewards (already computed per game)
            # Shape: actions is (N,), dr is (N, 1) -> squeeze to (N,)
            advantages = dr.squeeze()
            loss = train_on_game_results(
                model, opt, states, actions, advantages, device, entropy_coeff
            )
            train_time = time.time() - train_start
            now = time.time()
            if epoch % intermediate_save_frequency == 0 or (
                now - last_timed_save >= timed_save_interval
            ):
                last_timed_save = now
                checkpoint_path = os.path.join(
                    "nn_checkpoints", f"{run_name}_checkpoint_{epoch}.pth"
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
    final_checkpoint_path = os.path.join("nn_checkpoints", f"{run_name}_final.pth")
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
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_TYPES.keys()),
        default="polar",
        help="Model architecture to train (default: polar)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=TRAINING_RUN_NAME,
        help=f"Name for this training run, controls log/checkpoint filenames (default: {TRAINING_RUN_NAME})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="nn_weights/polar_pg_best.pth",
        help="Path to checkpoint to resume from (default: nn_weights/polar_pg_best.pth)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient to encourage exploration (default: 0.0)",
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=0.0,
        help="Death penalty magnitude applied to last N frames before death (default: 0.0, try -0.5)",
    )
    parser.add_argument(
        "--death-penalty-frames",
        type=int,
        default=60,
        help="Number of frames before death to apply ramping penalty (default: 60)",
    )
    args = parser.parse_args()
    train_model(
        args.width,
        args.height,
        args.batch_size,
        args.workers,
        args.model_type,
        args.run_name,
        checkpoint=None if args.no_resume else args.checkpoint,
        entropy_coeff=args.entropy_coeff,
        death_penalty=args.death_penalty,
        death_penalty_frames=args.death_penalty_frames,
    )


if __name__ == "__main__":
    main()
