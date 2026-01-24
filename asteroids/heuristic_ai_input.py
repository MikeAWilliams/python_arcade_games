"""
Example AI input method implementation.

This demonstrates how to create an alternative input method
that doesn't rely on keyboard input.
"""

import random
import math
from game import InputMethod, Action, BULLET_SPEED, SHOOT_COOLDOWN


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
    It does not account for the time it would take to turn and point at the asteroid
    It cannot move
    """

    def __init__(self, game):
        self.game = game
        self.current_action = Action.NO_ACTION
        self.shoot_cooldown = 0

    def predict_intercept(self, player_pos, asteroid_pos, asteroid_vel):
        """
        Calculate the intercept point for a moving asteroid using naive time estimate.
        Returns the angle to aim at.
        """
        # Calculate distance to asteroid
        dx = asteroid_pos.x - player_pos.x
        dy = asteroid_pos.y - player_pos.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Naive time estimate: time for bullet to reach current asteroid position
        time = distance / BULLET_SPEED
        
        # Predict where asteroid will be at that time
        pred_x = asteroid_pos.x + asteroid_vel.x * time
        pred_y = asteroid_pos.y + asteroid_vel.y * time
        
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
                self.shoot_cooldown = SHOOT_COOLDOWN
                return Action.SHOOT
            elif angle_diff > 0:
                return Action.TURN_LEFT
            else:
                return Action.TURN_RIGHT

        return Action.NO_ACTION
