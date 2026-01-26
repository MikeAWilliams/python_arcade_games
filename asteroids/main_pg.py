"""
Trains a Neural Network AI Input Method using policy gradient
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn

from game import Action, Game
from nn_ai_input import NNAIInputMethod, NNAIParameters


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


def train_on_game_results(model, optimizer, x, y, device):
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions + 1e-8) * y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss


def run_games_batch_worker(args):
    """
    Run multiple games sequentially in a single worker process.
    This reduces overhead by reusing the model and minimizing task submissions.
    Also computes discounted rewards in parallel to avoid main process bottleneck.

    Args:
        args: Tuple of (worker_id, num_games, width, height, model_state_dict)

    Returns:
        List of dicts, one per game with states, actions, probs, discounted_rewards, score
    """
    worker_id, num_games, width, height, model_state_dict = args

    # Create parameters once and reuse for all games in this worker
    params = NNAIParameters(device="cpu")
    params.model.load_state_dict(model_state_dict)
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
        probs = np.array(input_method.probabilities)
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
                "probs": probs,
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
    all_probs = []
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
            all_probs.append(result["probs"])
            all_discounted_rewards.append(result["discounted_rewards"])
            total_score += result["score"]

    # Concatenate all games into single batches
    states_batch = np.concatenate(all_states, axis=0)
    actions_batch = np.concatenate(all_actions, axis=0)
    probs_batch = np.concatenate(all_probs, axis=0)
    discounted_rewards_batch = np.concatenate(all_discounted_rewards, axis=0)

    return (
        states_batch,
        actions_batch,
        probs_batch,
        discounted_rewards_batch,
        total_score,
    )


# need to refactor and use game_runner.py. But being lazy to get it to work now
def execute_action(game, action):
    """Execute action on game"""
    if action == Action.TURN_LEFT:
        game.turning_left()
    elif action == Action.TURN_RIGHT:
        game.turning_right()
    elif action == Action.ACCELERATE:
        game.accelerate()
    elif action == Action.DECELERATE:
        game.decelerate()
    elif action == Action.SHOOT:
        game.shoot()
    elif action == Action.NO_ACTION:
        game.no_action()


def train_model(width, height, batch_size=32, num_workers=None):
    # Detect device (GPU if available, else CPU)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    device = "cpu"
    print(f"Using device: {device}")

    # Default to all CPU cores
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    print(f"Using {num_workers} worker processes for game simulation")
    print(f"Batch size: {batch_size} games per training update")

    params = NNAIParameters(device=device)
    model = params.model
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    alpha = 1e-4
    max_score = 0
    total_epochs = 2000
    print_frequency = 100
    intermediate_save_frequency = total_epochs / 10
    start_time = time.time()

    # Create persistent process pool to avoid recreation overhead
    print("Creating persistent worker pool...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for epoch in range(total_epochs):
            # Get model state dict for subprocess (CPU version)
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

            # Run games in parallel using persistent executor
            sim_start = time.time()
            states, actions, probs, dr, total_score = run_games_parallel(
                width, height, model_state_dict, batch_size, num_workers, executor
            )
            sim_time = time.time() - sim_start

            avg_score = total_score / batch_size
            if avg_score > max_score:
                max_score = avg_score

            # Training computation
            train_start = time.time()
            # recall that an action is 0 or 1 based on the index of the model output selected by probability sample
            # we had an array of actions but we ran np.vstack(action) which makde it into a list of lists where each internal list had one element
            # T transposes the array making it into a list with one list inside and all the elements in that
            # taking out the 0 element grabs that internal list
            # np.eye(2) converts a numeric value with 2 possible values into a 2x2 matrix one hot encoded
            # value of 0 to [1, 0] and value of 1 to [0, 1]
            one_hot_actions = np.eye(params.num_actions)[actions.T][0]
            # a probability row is a list of two probabilities, after 1 hot encoding above we end up
            # with the action taken as a 1. so a single subtraction is like [1, 0] - [0.7, 0.3] = [0.3, -0.3]
            gradients = one_hot_actions - probs
            # weight the gradient by dicsounted rewards (already computed per game in run_games_batch)
            gradients *= dr
            # target here is not a labeled correct value, just a nudge to the model
            # because alpha is small it will take some big rewards to make target much different than the initial probs
            # so when the rewards are small we don't change thigns much
            target = alpha * np.vstack([gradients]) + probs
            train_on_game_results(model, opt, states, target, device)
            train_time = time.time() - train_start
            if epoch % intermediate_save_frequency == 0:
                torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
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

                print(
                    f"{epoch}/{total_epochs} -> avg_score:{avg_score:.2f}, max:{max_score:.2f} | "
                    f"sim:{sim_time:.2f}s, train:{train_time:.2f}s | "
                    f"elapsed:{elapsed_str}, total:{total_str}, remaining:{remaining_str}"
                )

    # Save the trained model
    torch.save(model.state_dict(), "nn_model.pth")
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"Training completed in {total_time_str}")
    print("Model saved to nn_model.pth")


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
