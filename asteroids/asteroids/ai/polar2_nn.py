"""
AI input method using neural network with improved polar-coordinate features (v2).

Changes from polar_nn.py:
- Edge-to-edge distance (subtracts player + asteroid radii) instead of center-to-center
- Time-to-impact instead of closing speed
- Lateral velocity per asteroid (for lead aiming)
- Player velocity direction relative to bearing (for dodging)
- Size category removed (folded into edge-to-edge distance)
"""

import math

import numpy as np
import torch

from asteroids.core.game import (
    PLAYER_RADIUS,
    SHOOT_COOLDOWN,
    Action,
    InputMethod,
)
from asteroids.ai.polar_nn import _normalize_angle

MAX_ASTEROID_SLOTS = 9
MAX_TIME_TO_IMPACT = 10.0  # Cap for receding or very far asteroids


def compute_state(game) -> list[float]:
    """
    Compute the polar2 state vector for neural network input.

    Global state (4 values):
        - player speed magnitude (normalized by screen diagonal)
        - player velocity direction relative to bearing (normalized to [-1, 1])
        - shot cooldown (normalized 0-1)
        - asteroid count (normalized by MAX_ASTEROID_SLOTS)

    Per asteroid, sorted by edge distance, top MAX_ASTEROID_SLOTS (4 values each):
        - edge-to-edge distance (center dist minus both radii, normalized by screen diagonal)
        - relative angle from player bearing (normalized to [-1, 1] by dividing by pi)
        - time-to-impact (edge distance / closing speed, clamped to MAX_TIME_TO_IMPACT)
        - lateral velocity (perpendicular component, normalized by screen diagonal)

    Args:
        game: Game instance

    Returns:
        List of floats representing the state vector
    """
    max_dist = math.sqrt(game.width**2 + game.height**2)

    px = game.player.geometry.pos.x
    py = game.player.geometry.pos.y
    pvx = game.player.vel.x
    pvy = game.player.vel.y
    bearing = game.player.geometry.angle
    player_speed = math.sqrt(pvx**2 + pvy**2)

    # Player velocity direction relative to bearing
    if player_speed > 0:
        vel_angle = math.atan2(pvy, pvx)
        vel_dir_relative = _normalize_angle(vel_angle - bearing) / math.pi
    else:
        vel_dir_relative = 0.0

    # Compute per-asteroid features
    asteroid_features = []
    for asteroid in game.asteroids:
        ax = asteroid.geometry.pos.x
        ay = asteroid.geometry.pos.y
        avx = asteroid.vel.x
        avy = asteroid.vel.y

        # Center-to-center distance
        dx = ax - px
        dy = ay - py
        center_dist = math.sqrt(dx**2 + dy**2)

        # Edge-to-edge distance
        edge_dist = max(0.0, center_dist - PLAYER_RADIUS - asteroid.geometry.radius)

        # Relative angle from player bearing
        angle_to_asteroid = math.atan2(dy, dx)
        relative_angle = _normalize_angle(angle_to_asteroid - bearing)

        # Direction vector from player to asteroid
        if center_dist > 0:
            dir_x = dx / center_dist
            dir_y = dy / center_dist
        else:
            dir_x = 0.0
            dir_y = 0.0

        # Relative velocity
        rel_vx = avx - pvx
        rel_vy = avy - pvy

        # Closing speed (radial component, positive = approaching)
        closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)

        # Time-to-impact
        if closing_speed > 0:
            tti = min(edge_dist / closing_speed, MAX_TIME_TO_IMPACT)
        else:
            tti = MAX_TIME_TO_IMPACT

        # Lateral velocity (perpendicular component)
        lateral_speed = -rel_vx * dir_y + rel_vy * dir_x

        asteroid_features.append((edge_dist, relative_angle, tti, lateral_speed))

    # Sort by edge distance (nearest first)
    asteroid_features.sort(key=lambda x: x[0])

    # Build state vector
    result = []

    # Global state
    result.append(float(player_speed / max_dist))
    result.append(float(vel_dir_relative))
    result.append(float(game.shoot_cooldown / SHOOT_COOLDOWN))
    result.append(
        float(min(len(asteroid_features), MAX_ASTEROID_SLOTS) / MAX_ASTEROID_SLOTS)
    )

    # Per-asteroid state (pad with zeros if fewer than MAX_ASTEROID_SLOTS)
    for i in range(MAX_ASTEROID_SLOTS):
        if i < len(asteroid_features):
            edge_dist, rel_angle, tti, lateral = asteroid_features[i]
            result.append(float(edge_dist / max_dist))
            result.append(float(rel_angle / math.pi))
            result.append(float(tti / MAX_TIME_TO_IMPACT))
            result.append(float(lateral / max_dist))
        else:
            result.extend([0.0, 0.0, 0.0, 0.0])

    return result


class Polar2NNParameters:
    """Configuration parameters for Polar2NN"""

    def __init__(self, device=None):
        if device is None:
            device = "cpu"
        self.device = device
        global_state_count = 4  # speed, vel_direction, cooldown, asteroid_count
        per_asteroid_count = 4  # edge_distance, relative_angle, tti, lateral_velocity
        self.num_inputs = global_state_count + per_asteroid_count * MAX_ASTEROID_SLOTS
        self.num_actions = len(Action)
        hidden1 = 128
        hidden2 = 64
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, hidden1, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden1, hidden2, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden2, self.num_actions, dtype=torch.float32),
        ).to(device)


class Polar2NNInputMethod(InputMethod):
    """
    A Neural Network based AI using improved polar-coordinate features (v2).
    """

    def __init__(self, game, parameters: Polar2NNParameters = None, keep_data=False):
        self.game = game
        self.keep_data = keep_data
        if parameters is None:
            parameters = Polar2NNParameters()
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
