"""
Generate a randomly-initialized neural network model weights file.

Useful for creating a default nn_model.pth when no trained model exists,
e.g. after an architecture change that invalidates old weights.

Usage:
    python tools/generate_random_model.py
    python tools/generate_random_model.py --output nn_weights/my_model.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from asteroids.ai.neural import NNAIParameters


def main():
    parser = argparse.ArgumentParser(
        description="Generate a randomly-initialized neural network model weights file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nn_weights/nn_model.pth",
        help="Output path for the model file (default: nn_weights/nn_model.pth)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    params = NNAIParameters()
    torch.save(params.model.state_dict(), args.output)
    print(f"Saved {params.num_inputs}-input model to {args.output}")


if __name__ == "__main__":
    main()
