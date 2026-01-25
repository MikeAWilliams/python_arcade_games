"""
AI input method using neural network
"""

import math
import random
import statistics
from enum import Enum

from game import Action, InputMethod, Vec2d


class NNAIParameters:
    """Configuration parameters for NNAI"""

    def __init__(self):
        pass


# Genetic Algorithm Functions for NNAI Parameter Optimization


# Parameter bounds for NNAI genetic algorithm
# Format: {param_name: (min_value, max_value)}
NN_AI_PARAM_BOUNDS = {}


def nn_ai_random_params() -> NNAIParameters:
    """
    Generate random NNAI parameters within bounds.

    Returns:
        NNAIParameters with random values
    """
    pass


def crossover_parameter(parent1, parent2):
    alpha = random.random()
    offspring_value = alpha * parent1 + (1 - alpha) * parent2
    return offspring_value


def nn_ai_crossover(params1: NNAIParameters, params2: NNAIParameters) -> NNAIParameters:
    """
    Create offspring parameters by blending two parents.

    Uses blend crossover: for each parameter, offspring value is
    randomly chosen using: offspring = alpha * p1 + (1-alpha) * p2
    where alpha is random in range [0, 1]

    Args:
        params1: First parent parameters
        params2: Second parent parameters

    Returns:
        New NNAIParameters (offspring)
    """
    pass


def mutate_param(val, min_val, max_val, rate):
    if random.random() < rate:
        sigma = (max_val - min_val) * 0.1
        val += random.gauss(0, sigma)
        val = max(min_val, min(val, max_val))
    return val


def nn_ai_mutate(params: NNAIParameters, mutation_rate: float) -> NNAIParameters:
    """
    Mutate parameters with Gaussian noise.

    For each parameter, with probability mutation_rate:
    - Add Gaussian noise: param += N(0, sigma)
    - Clamp to parameter bounds
    - sigma = (max - min) * 0.1  # 10% of range

    Args:
        params: Parameters to mutate
        mutation_rate: Probability of mutating each parameter

    Returns:
        New NNAIParameters (mutated)
    """
    pass


def nn_ai_calculate_diversity(params_list: list[NNAIParameters]) -> float:
    """
    Calculate population diversity for NNAI parameters.

    Computes normalized standard deviation across all parameters
    and returns the average as a diversity metric.

    Args:
        params_list: List of NNAIParameters from population

    Returns:
        Diversity metric (0-1, higher = more diverse)
    """
    # Need at least 2 individuals to calculate diversity
    if len(params_list) < 2:
        return 0.0

    pass


class NNAIInputMethod(InputMethod):
    """
    A Neural Network based AI.
    """

    def __init__(self, game, parameters: NNAIParameters = None):
        self.game = game

        # Use default parameters if none provided
        if parameters is None:
            parameters = NNAIParameters()

    def get_move(self) -> Action:
        """
        Return the game action predicted by the Neural Network
        """
        return Action.NO_ACTION
