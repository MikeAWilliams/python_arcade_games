"""
Measure the entropy and action distribution of a trained NN model.

Runs a single game (up to --max-frames) and reports:
- Per-frame entropy statistics (mean, min, max, std)
- Average action probabilities across all frames

Usage:
    python tools/measure_entropy.py                                              # polar, default checkpoint
    python tools/measure_entropy.py --model-type polar2 --checkpoint nn_checkpoints/polar2_pg_best.pth
    python tools/measure_entropy.py --max-frames 10000
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asteroids.ai.polar_nn import PolarNNParameters, PolarNNInputMethod
from asteroids.ai.polar2_nn import Polar2NNParameters, Polar2NNInputMethod
from asteroids.ai.raw_geometry_nn import validate_and_load_model
from asteroids.core.game import Game
from asteroids.core.game_runner import execute_action

MODEL_TYPES = {
    "polar": {
        "params_class": PolarNNParameters,
        "input_class": PolarNNInputMethod,
        "default_checkpoint": "nn_checkpoints/polar_pg_best.pth",
    },
    "polar2": {
        "params_class": Polar2NNParameters,
        "input_class": Polar2NNInputMethod,
        "default_checkpoint": "nn_checkpoints/polar2_pg_best.pth",
    },
}


def measure_entropy(model_type: str, checkpoint: str, max_frames: int):
    model_info = MODEL_TYPES[model_type]

    # Load model
    params = model_info["params_class"](device="cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_and_load_model(
        params.model, ckpt["model_state_dict"], source_description=checkpoint
    )
    params.model.eval()

    # Run a game and collect action distributions
    game = Game(1280, 720)
    inp = model_info["input_class"](game=game, parameters=params, keep_data=True)
    dt = 1 / 60
    all_probs = []
    frames = 0

    while game.player_alive and frames < max_frames:
        game.clear_turn()
        game.clear_acc()

        state = inp.compute_state()
        with torch.no_grad():
            logits = params.model(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            )
            probs = F.softmax(logits, dim=1).numpy()[0]
        all_probs.append(probs)

        action = inp.get_move()
        execute_action(game, action)
        game.update(dt)
        frames += 1

    all_probs = np.array(all_probs)

    # Compute entropy per frame: -sum(p * log(p))
    entropies = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    # Max possible entropy for 6 actions = ln(6) = 1.79
    max_entropy = np.log(6)

    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Frames: {frames}")
    print(f"Score: {game.player_score}")
    print(f"Max possible entropy (uniform over 6 actions): {max_entropy:.4f}")
    print(
        f"Mean entropy: {np.mean(entropies):.4f} ({np.mean(entropies)/max_entropy*100:.1f}% of max)"
    )
    print(f"Min entropy:  {np.min(entropies):.4f}")
    print(f"Max entropy:  {np.max(entropies):.4f}")
    print(f"Std entropy:  {np.std(entropies):.4f}")
    print()
    print("Average action probabilities:")
    actions = [
        "TURN_LEFT",
        "TURN_RIGHT",
        "ACCELERATE",
        "DECELERATE",
        "SHOOT",
        "NO_ACTION",
    ]
    for i, name in enumerate(actions):
        print(f"  {name:15s}: {np.mean(all_probs[:, i]):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure entropy and action distribution of a trained NN model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_TYPES.keys()),
        default="polar",
        help="Model type (default: polar)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default depends on model type)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Maximum frames to run (default: 5000)",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint or MODEL_TYPES[args.model_type]["default_checkpoint"]
    measure_entropy(args.model_type, checkpoint, args.max_frames)


if __name__ == "__main__":
    main()
