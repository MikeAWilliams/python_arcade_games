"""
Example AI input method implementation.

This demonstrates how to create an alternative input method
that doesn't rely on keyboard input.
"""

import random
import math
from game import InputMethod, Action


class RandomAIInput(InputMethod):
    """
    Simple AI input that makes random decisions.
    This is just a demonstration - a real AI would analyze game state.
    """

    def __init__(self):
        self.current_action = Action.NO_ACTION
        self.action_duration = 0

    def get_move(self) -> Action:
        """
        Returns a random action periodically.
        In a real AI, this would analyze the game state and make intelligent decisions.
        """
        # Continue current action if it has duration remaining
        if self.action_duration > 0:
            self.action_duration -= 1
            if self.action_duration == 0:
                self.current_action = Action.NO_ACTION
            return self.current_action

        # Randomly decide whether to take a new action
        if random.random() < 0.1:
            # Random chance to shoot
            if random.random() < 0.3:
                return Action.SHOOT

            # Choose a continuous action
            actions = [
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.ACCELERATE,
                Action.DECELERATE,
            ]
            self.current_action = random.choice(actions)
            self.action_duration = random.randint(10, 60)
            return self.current_action

        return self.current_action


class SmartAIInput(InputMethod):
    """
    A more intelligent AI that analyzes game state.
    This implementation aims at the nearest asteroid.
    """

    def __init__(self, game):
        self.game = game
        self.current_action = Action.NO_ACTION

    def get_move(self) -> Action:
        """
        Analyze game state and return intelligent action.
        """
        player_pos = self.game.player.geometry.pos
        player_angle = self.game.player.geometry.angle
        asteroids = self.game.asteroids

        # Find closest asteroid
        if asteroids:
            closest_asteroid = min(asteroids,
                                  key=lambda a: ((a.geometry.pos.x - player_pos.x)**2 +
                                                (a.geometry.pos.y - player_pos.y)**2))

            # Calculate angle to closest asteroid
            dx = closest_asteroid.geometry.pos.x - player_pos.x
            dy = closest_asteroid.geometry.pos.y - player_pos.y
            angle_to_asteroid = math.atan2(dy, dx)

            # Normalize angle difference
            angle_diff = (angle_to_asteroid - player_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            # Simple strategy: turn to face asteroid and shoot
            if abs(angle_diff) < 0.1:
                # Facing asteroid, shoot!
                return Action.SHOOT
            elif angle_diff > 0:
                return Action.TURN_LEFT
            else:
                return Action.TURN_RIGHT

        return Action.NO_ACTION
