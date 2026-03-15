"""
Generate a randomly-initialized neural network model weights file.

Useful for creating a default model when no trained model exists,
e.g. after an architecture change that invalidates old weights.

Usage:
    python tools/generate_random_model.py                          # raw_geometry (default)
    python tools/generate_random_model.py --model polar             # polar
    python tools/generate_random_model.py --model polar --output nn_weights/my_model.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from asteroids.ai.raw_geometry_nn import RawGeometryNNParameters
from asteroids.ai.polar_nn import PolarNNParameters
from asteroids.ai.polar2_nn import Polar2NNParameters

MODELS = {
    "raw_geometry": {
        "params_class": RawGeometryNNParameters,
        "default_output": "nn_weights/nn_model.pth",
    },
    "polar": {
        "params_class": PolarNNParameters,
        "default_output": "nn_weights/nn_polar.pth",
    },
    "polar2": {
        "params_class": Polar2NNParameters,
        "default_output": "nn_weights/nn_polar2.pth",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a randomly-initialized neural network model weights file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="raw_geometry",
        help="Model type to generate (default: raw_geometry)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the model file (default depends on model type)",
    )
    args = parser.parse_args()

    model_info = MODELS[args.model]
    output = args.output or model_info["default_output"]

    os.makedirs(os.path.dirname(output), exist_ok=True)

    params = model_info["params_class"]()
    torch.save(params.model.state_dict(), output)
    print(f"Saved {args.model} model ({params.num_inputs} inputs) to {output}")


if __name__ == "__main__":
    main()
