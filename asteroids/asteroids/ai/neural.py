"""
AI input method using neural network
"""

import math
import random
import statistics
from asyncio.unix_events import SelectorEventLoop
from enum import Enum

import numpy as np
import torch
from torch import nn

from asteroids.core.game import Action, InputMethod, SHOOT_COOLDOWN


def compute_state(game):
    """
    Compute the game state vector for neural network input.

    State vector has the form:
    - player_state: x, y, vx, vy, theta, shot_cooldown
    - per_asteroid_state: x, y, vx, vy, active (for up to 27 asteroids)

    Total possible asteroids = 27 (3 + 6 + 18 for each generation)

    Args:
        game: Game instance

    Returns:
        List of floats representing the game state
    """
    result = []

    # Encode the player state
    result.append(float(game.player.geometry.pos.x / game.width))
    result.append(float(game.player.geometry.pos.y / game.height))
    result.append(float(game.player.vel.x / game.width))
    result.append(float(game.player.vel.y / game.height))
    result.append(float(game.player.geometry.angle / (2 * math.pi)))
    result.append(float(game.shoot_cooldown / SHOOT_COOLDOWN))

    # Encode the asteroid state
    asteroid_id_map = {}
    for asteroid in game.asteroids:
        asteroid_id_map[asteroid.id] = asteroid

    for id in range(27):
        if id in asteroid_id_map:
            asteroid = asteroid_id_map[id]
            result.append(float(asteroid.geometry.pos.x / game.width))
            result.append(float(asteroid.geometry.pos.y / game.height))
            result.append(float(asteroid.vel.x / game.width))
            result.append(float(asteroid.vel.y / game.height))
            result.append(float(1))
        else:
            result.extend([float(0), float(0), float(0), float(0), float(0)])

    return result


class NNAIParameters:
    """Configuration parameters for NNAI"""

    def __init__(self, device=None):
        if device is None:
            device = "cpu"
        self.device = device
        player_state_count = 6  # x,y, vx,vy,theta, shot_cooldown
        per_asteroid_count = 5  # x,y, vx,vy, active
        possible_asteroids = 27  # 3+6+18 for each generation
        self.num_inputs = player_state_count + per_asteroid_count * possible_asteroids
        self.num_actions = len(Action)
        middle_dim = 128
        # for now try a single hidden layer
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_inputs, middle_dim, bias=False, dtype=torch.float32
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                middle_dim, self.num_actions, bias=False, dtype=torch.float32
            ),
        ).to(device)


class NNAIInputMethod(InputMethod):
    """
    A Neural Network based AI.
    """

    def __init__(self, game, parameters: NNAIParameters = None, keep_data=False):
        self.game = game
        self.keep_data = keep_data
        if parameters is None:
            parameters = NNAIParameters()
        self.parameters = parameters

        self.states = [] if keep_data else None
        self.actions_taken = [] if keep_data else None
        self.probabilities = [] if keep_data else None
        self.scores = [] if keep_data else None

    def compute_state(self):
        """Compute state vector using the free function"""
        result = compute_state(self.game)
        assert len(result) == self.parameters.num_inputs
        return result

    def compute_action(self, state):
        logits = self.parameters.model(
            torch.from_numpy(np.expand_dims(state, 0))
            .float()
            .to(self.parameters.device)
        )
        # Apply softmax manually to convert logits to probabilities
        action_probs = torch.nn.functional.softmax(logits, dim=1)[0]
        action_probs_cpu = action_probs.detach().cpu().numpy()
        action = np.random.choice(
            self.parameters.num_actions, p=np.squeeze(action_probs_cpu)
        )
        return Action(action), action_probs_cpu

    def get_move(self) -> Action:
        """
        Return the game action predicted by the Neural Network
        """
        state = self.compute_state()
        action, prob = self.compute_action(state)
        if self.keep_data:
            self.states.append(state)
            self.actions_taken.append(action.value)
            self.probabilities.append(prob)
            self.scores.append(self.game.player_score)
        return action
