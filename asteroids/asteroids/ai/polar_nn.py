"""
AI input method using neural network with polar-coordinate features.

Instead of raw geometry, this approach pre-computes player-relative features
for each asteroid: distance, relative angle from bearing, and closing speed.
Asteroids are sorted by distance and only the nearest N are kept.
"""

import math

import numpy as np
import torch

from asteroids.core.game import (
    BIG_ASTEROID_RADIUS,
    MEDIUM_ASTEROID_RADIUS,
    SHOOT_COOLDOWN,
    SMALL_ASTEROID_RADIUS,
    Action,
    InputMethod,
)

MAX_ASTEROID_SLOTS = 9


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _asteroid_size_category(radius: float) -> float:
    """Map asteroid radius to normalized size: small=0.33, medium=0.66, large=1.0."""
    if radius >= BIG_ASTEROID_RADIUS:
        return 1.0
    elif radius >= MEDIUM_ASTEROID_RADIUS:
        return 0.66
    else:
        return 0.33


def compute_state(game) -> list[float]:
    """
    Compute the polar-coordinate state vector for neural network input.

    Global state (3 values):
        - player speed magnitude (normalized by screen diagonal)
        - shot cooldown (normalized 0-1)
        - asteroid count (normalized by MAX_ASTEROID_SLOTS)

    Per asteroid, sorted by distance, top MAX_ASTEROID_SLOTS (4 values each):
        - distance from player (normalized 0-1 by screen diagonal)
        - relative angle from player bearing (normalized to [-1, 1] by dividing by pi)
        - closing speed (normalized by screen diagonal)
        - size category (0.33 / 0.66 / 1.0)

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

    # Compute per-asteroid features
    asteroid_features = []
    for asteroid in game.asteroids:
        ax = asteroid.geometry.pos.x
        ay = asteroid.geometry.pos.y
        avx = asteroid.vel.x
        avy = asteroid.vel.y

        # Distance
        dx = ax - px
        dy = ay - py
        dist = math.sqrt(dx**2 + dy**2)

        # Relative angle from player bearing
        angle_to_asteroid = math.atan2(dy, dx)
        relative_angle = _normalize_angle(angle_to_asteroid - bearing)

        # Closing speed: positive means approaching
        # Project relative velocity onto the direction vector
        if dist > 0:
            dir_x = dx / dist
            dir_y = dy / dist
        else:
            dir_x = 0.0
            dir_y = 0.0
        rel_vx = avx - pvx
        rel_vy = avy - pvy
        closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)

        size = _asteroid_size_category(asteroid.geometry.radius)

        asteroid_features.append((dist, relative_angle, closing_speed, size))

    # Sort by distance (nearest first)
    asteroid_features.sort(key=lambda x: x[0])

    # Build state vector
    result = []

    # Global state
    result.append(float(player_speed / max_dist))
    result.append(float(game.shoot_cooldown / SHOOT_COOLDOWN))
    # asteroid count, normalized. 0  no asteroids to 1 MAX_ASTEROID_SLOTS
    result.append(
        float(min(len(asteroid_features), MAX_ASTEROID_SLOTS) / MAX_ASTEROID_SLOTS)
    )

    # Per-asteroid state (pad with zeros if fewer than MAX_ASTEROID_SLOTS)
    for i in range(MAX_ASTEROID_SLOTS):
        if i < len(asteroid_features):
            dist, rel_angle, closing, size = asteroid_features[i]
            result.append(float(dist / max_dist))
            result.append(float(rel_angle / math.pi))
            result.append(float(closing / max_dist))
            result.append(float(size))
        else:
            result.extend([0.0, 0.0, 0.0, 0.0])

    return result


def convert_raw_geometry_state(
    raw_state: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    Convert a RawGeometryNN state vector to a PolarNN state vector.

    This allows reuse of training data recorded in the raw geometry format.

    Args:
        raw_state: 1D array of 142 floats from RawGeometryNN compute_state()
        width: Game world width used during recording
        height: Game world height used during recording

    Returns:
        1D array of floats in PolarNN format
    """
    max_dist = math.sqrt(width**2 + height**2)

    # Decode player state from raw format
    px = raw_state[0] * width
    py = raw_state[1] * height
    pvx = raw_state[2] * width
    pvy = raw_state[3] * height
    bearing_cos = raw_state[4]
    bearing_sin = raw_state[5]
    shot_cooldown = raw_state[6]  # already normalized
    bearing = math.atan2(bearing_sin, bearing_cos)
    player_speed = math.sqrt(pvx**2 + pvy**2)

    # Decode asteroids (27 slots, 5 values each starting at index 7)
    asteroid_features = []
    for i in range(27):
        base = 7 + i * 5
        active = raw_state[base + 4]
        if active < 0.5:
            continue

        ax = raw_state[base + 0] * width
        ay = raw_state[base + 1] * height
        avx = raw_state[base + 2] * width
        avy = raw_state[base + 3] * height

        dx = ax - px
        dy = ay - py
        dist = math.sqrt(dx**2 + dy**2)

        angle_to_asteroid = math.atan2(dy, dx)
        relative_angle = _normalize_angle(angle_to_asteroid - bearing)

        if dist > 0:
            dir_x = dx / dist
            dir_y = dy / dist
        else:
            dir_x = 0.0
            dir_y = 0.0
        rel_vx = avx - pvx
        rel_vy = avy - pvy
        closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)

        # Raw format doesn't store radius, so infer from asteroid ID:
        # IDs 0-2: big, 3-8: medium, 9-26: small
        if i < 3:
            size = 1.0
        elif i < 9:
            size = 0.66
        else:
            size = 0.33

        asteroid_features.append((dist, relative_angle, closing_speed, size))

    # Sort by distance
    asteroid_features.sort(key=lambda x: x[0])

    # Build state vector
    result = []

    # Global state
    result.append(float(player_speed / max_dist))
    result.append(float(shot_cooldown))
    result.append(
        float(min(len(asteroid_features), MAX_ASTEROID_SLOTS) / MAX_ASTEROID_SLOTS)
    )

    # Per-asteroid state
    for i in range(MAX_ASTEROID_SLOTS):
        if i < len(asteroid_features):
            dist, rel_angle, closing, size = asteroid_features[i]
            result.append(float(dist / max_dist))
            result.append(float(rel_angle / math.pi))
            result.append(float(closing / max_dist))
            result.append(float(size))
        else:
            result.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(result, dtype=np.float32)


class PolarNNParameters:
    """Configuration parameters for PolarNN"""

    def __init__(self, device=None):
        if device is None:
            device = "cpu"
        self.device = device
        global_state_count = 3  # speed, cooldown, asteroid_count
        per_asteroid_count = 4  # distance, relative_angle, closing_speed, size
        self.num_inputs = global_state_count + per_asteroid_count * MAX_ASTEROID_SLOTS
        self.num_actions = len(Action)
        middle_dim = 128
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_inputs, middle_dim, bias=False, dtype=torch.float32
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                middle_dim, self.num_actions, bias=False, dtype=torch.float32
            ),
        ).to(device)


class PolarNNInputMethod(InputMethod):
    """
    A Neural Network based AI using polar-coordinate features.
    """

    def __init__(self, game, parameters: PolarNNParameters = None, keep_data=False):
        self.game = game
        self.keep_data = keep_data
        if parameters is None:
            parameters = PolarNNParameters()
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
