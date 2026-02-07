"""Core game logic and non-AI input."""

from asteroids.core.game import (
    Action,
    InputMethod,
    Game,
    Vec2d,
    GeometryObject,
    GeometryState,
    Player,
    Asteroid,
    Bullet,
    PLAYER_TURN_RATE,
    PLAYER_ACCELERATION,
    PLAYER_RADIUS,
    PLAYER_EXCLUSION_RADIUS,
    BULLET_SPEED,
    ASTEROID_BASE_SPEED,
    ASTEROID_SPEED_INCREMENT,
    BIG_ASTEROID_RADIUS,
    MEDIUM_ASTEROID_RADIUS,
    SMALL_ASTEROID_RADIUS,
    SHOOT_COOLDOWN,
)
from asteroids.core.keyboard import KeyboardInput

# Note: game_runner imports are not included here to avoid circular imports
# Import them directly: from asteroids.core.game_runner import run_single_game, execute_action

__all__ = [
    "Action",
    "InputMethod",
    "Game",
    "Vec2d",
    "GeometryObject",
    "GeometryState",
    "Player",
    "Asteroid",
    "Bullet",
    "KeyboardInput",
    "PLAYER_TURN_RATE",
    "PLAYER_ACCELERATION",
    "PLAYER_RADIUS",
    "PLAYER_EXCLUSION_RADIUS",
    "BULLET_SPEED",
    "ASTEROID_BASE_SPEED",
    "ASTEROID_SPEED_INCREMENT",
    "BIG_ASTEROID_RADIUS",
    "MEDIUM_ASTEROID_RADIUS",
    "SMALL_ASTEROID_RADIUS",
    "SHOOT_COOLDOWN",
]
