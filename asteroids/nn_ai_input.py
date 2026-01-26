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

from game import SHOOT_COOLDOWN, Action, InputMethod


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
            # dim=1 because batch size is 1
            torch.nn.Softmax(dim=1),
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

    def get_asteroid_id_map(self):
        asteroid_id_map = {}
        for asteroid in self.game.asteroids:
            asteroid_id_map[asteroid.id] = asteroid
        return asteroid_id_map

    def compute_state(self):
        # state vecotor should have the form
        # player_state x,y, vx,vy,theta, shot_cooldown
        # the for each asteroid
        # per_asteroid_state x,y, vx,vy, active
        # with a total_possible_asteroids = 27  # 3+6+18 for each generation
        # Note this is totally hard coded to the generation rules and will need to change if I change that
        result = []
        # encode the player state
        result.append(float(self.game.player.geometry.pos.x / self.game.width))
        result.append(float(self.game.player.geometry.pos.y / self.game.height))
        result.append(float(self.game.player.vel.x / self.game.width))
        result.append(float(self.game.player.vel.y / self.game.height))
        result.append(float(self.game.player.geometry.angle / (2 * math.pi)))
        result.append(float(self.game.shoot_cooldown / SHOOT_COOLDOWN))

        # encode the asteroid state
        id_map = self.get_asteroid_id_map()
        for id in range(27):
            if id in id_map:
                asteroid = id_map[id]
                result.append(float(asteroid.geometry.pos.x / self.game.width))
                result.append(float(asteroid.geometry.pos.y / self.game.height))
                result.append(float(asteroid.vel.x / self.game.width))
                result.append(float(asteroid.vel.y / self.game.height))
                result.append(float(1))
            else:
                result.extend([float(0), float(0), float(0), float(0), float(0)])

        assert len(result) == self.parameters.num_inputs
        return result

    def compute_action(self, state):
        action_probs = self.parameters.model(
            torch.from_numpy(np.expand_dims(state, 0))
            .float()
            .to(self.parameters.device)
        )[0]
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
