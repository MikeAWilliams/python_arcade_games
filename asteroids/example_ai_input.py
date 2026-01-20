"""
Example AI input method implementation.

This demonstrates how to create an alternative input method
that doesn't rely on keyboard input.
"""

import random
import math
from game import InputMethod, Action, BULLET_SPEED


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
    This implementation predicts asteroid position based on velocity and aims ahead.
    """

    def __init__(self, game):
        self.game = game
        self.current_action = Action.NO_ACTION
        self.shoot_cooldown = 0

    def predict_intercept(self, player_pos, asteroid_pos, asteroid_vel):
        """
        Calculate the intercept point for a moving asteroid.
        Returns the angle to aim at, or None if no solution.
        """
        # Relative position
        dx = asteroid_pos.x - player_pos.x
        dy = asteroid_pos.y - player_pos.y

        # Solve quadratic equation for intercept time
        # |asteroid_pos + asteroid_vel * t - player_pos| = BULLET_SPEED * t

        a = asteroid_vel.x**2 + asteroid_vel.y**2 - BULLET_SPEED**2
        b = 2 * (dx * asteroid_vel.x + dy * asteroid_vel.y)
        c = dx**2 + dy**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            # No solution, aim at current position
            return math.atan2(dy, dx)

        if abs(a) < 1e-6:
            # Linear case
            if abs(b) < 1e-6:
                return math.atan2(dy, dx)
            t = -c / b
        else:
            # Quadratic case - take the smaller positive root
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2*a)
            t2 = (-b - sqrt_disc) / (2*a)

            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                # No positive solution, aim at current position
                return math.atan2(dy, dx)

        # Calculate predicted position
        pred_x = asteroid_pos.x + asteroid_vel.x * t
        pred_y = asteroid_pos.y + asteroid_vel.y * t

        # Return angle to predicted position
        return math.atan2(pred_y - player_pos.y, pred_x - player_pos.x)

    def get_move(self) -> Action:
        """
        Analyze game state and return intelligent action.
        """
        # Decrement cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        player_pos = self.game.player.geometry.pos
        player_angle = self.game.player.geometry.angle
        asteroids = self.game.asteroids

        # Find closest asteroid
        if asteroids:
            closest_asteroid = min(asteroids,
                                  key=lambda a: ((a.geometry.pos.x - player_pos.x)**2 +
                                                (a.geometry.pos.y - player_pos.y)**2))

            # Calculate predicted intercept angle
            target_angle = self.predict_intercept(
                player_pos,
                closest_asteroid.geometry.pos,
                closest_asteroid.vel
            )

            # Normalize angle difference
            angle_diff = (target_angle - player_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            # Strategy: turn to face predicted position and shoot
            if abs(angle_diff) < 0.05 and self.shoot_cooldown == 0:
                # Aimed at predicted position, shoot!
                self.shoot_cooldown = 30
                return Action.SHOOT
            elif angle_diff > 0:
                return Action.TURN_LEFT
            else:
                return Action.TURN_RIGHT

        return Action.NO_ACTION
