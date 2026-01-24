"""
Example AI input method implementation.

This demonstrates how to create an alternative input method
that doesn't rely on keyboard input.
"""

import random
import math
from enum import Enum
from game import InputMethod, Action, BULLET_SPEED, SHOOT_COOLDOWN, PLAYER_RADIUS, vec2d


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


class Strategy(Enum):
    """Enumeration of AI strategies"""
    EVASIVE_ACTION = "evasive_action"
    SPEED_CONTROL = "speed_control"
    SHOOT_NEAREST = "shoot_nearest"


class SmartAIInput(InputMethod):
    """
    A more intelligent AI that analyzes game state.
    This implementation predicts asteroid position based on velocity and aims ahead.
    It does not account for the time it would take to turn and point at the asteroid
    It cannot move
    """

    # Class-level constants for evasive action
    DANGER_RADIUS = PLAYER_RADIUS * 5
    EVASION_MAX_DISTANCE = 300  # Maximum distance to consider for evasion weighting
    MAX_SPEED = 100  # Maximum velocity magnitude before speed control activates

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

    def get_strategy(self) -> Strategy:
        """
        Determine which strategy to use based on game state.
        Priority: EVASIVE_ACTION > SPEED_CONTROL > SHOOT_NEAREST
        """
        # Check if any asteroid is within danger radius
        for asteroid in self.game.asteroids:
            distance = math.dist(self.game.player.geometry.pos, asteroid.geometry.pos)

            if distance < self.DANGER_RADIUS:
                return Strategy.EVASIVE_ACTION

        # Check if player velocity is too high
        velocity_magnitude = self.game.player.vel.size()
        if velocity_magnitude > self.MAX_SPEED:
            return Strategy.SPEED_CONTROL

        return Strategy.SHOOT_NEAREST

    def shoot_nearest(self) -> Action:
        """
        Strategy: Find the nearest asteroid, predict its position, and shoot at it.
        Turns to face the predicted intercept point.
        """
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

    def evasive_action(self) -> Action:
        """
        Strategy: Compute weighted average threat vector from nearby asteroids
        and turn to face that direction, then decelerate to move away from danger.

        Weight calculation: For asteroids within EVASION_MAX_DISTANCE,
        weight = max(0, EVASION_MAX_DISTANCE - distance)
        """
        if not self.game.asteroids:
            return Action.NO_ACTION

        # Compute weighted threat vector using vec2d
        weighted_vector = vec2d(0.0, 0.0)
        total_weight = 0.0

        for asteroid in self.game.asteroids:
            distance = math.dist(self.game.player.geometry.pos, asteroid.geometry.pos)

            # Only consider asteroids within max distance
            if distance < self.EVASION_MAX_DISTANCE and distance > 0:
                # Weight is inversely proportional to distance
                weight = self.EVASION_MAX_DISTANCE - distance

                # Direction vector from player to asteroid (normalized)
                direction = vec2d(
                    asteroid.geometry.pos.x - self.game.player.geometry.pos.x,
                    asteroid.geometry.pos.y - self.game.player.geometry.pos.y
                )
                normalized_direction = direction.multiply(1.0 / distance)

                # Accumulate weighted direction vectors
                weighted_vector.x += normalized_direction.x * weight
                weighted_vector.y += normalized_direction.y * weight
                total_weight += weight

        # If no asteroids within range, fall back to no action
        if total_weight == 0:
            return Action.NO_ACTION

        # Compute average threat direction
        threat_vector = vec2d(
            weighted_vector.x / total_weight,
            weighted_vector.y / total_weight
        )
        threat_angle = math.atan2(threat_vector.y, threat_vector.x)

        # Calculate angle difference to face the threat
        angle_diff = (threat_angle - self.game.player.geometry.angle) % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi

        # Turn to face the threat direction, then decelerate (move away)
        if abs(angle_diff) < 0.1:  # Facing threat direction
            return Action.DECELERATE
        elif angle_diff > 0:
            return Action.TURN_LEFT
        else:
            return Action.TURN_RIGHT

    def speed_control(self) -> Action:
        """
        Strategy: Manage player velocity by turning to align or oppose velocity vector.

        Algorithm:
        1. Calculate velocity angle (direction of movement)
        2. Calculate angle difference between velocity and player orientation
        3. Determine shorter turn direction (clockwise or counterclockwise)
        4. If turning toward velocity: DECELERATE (slow down)
        5. If turning away from velocity: ACCELERATE (change direction while slowing)
        """
        velocity = self.game.player.vel

        # If velocity is near zero, no speed control needed
        if velocity.size() < 1:
            return Action.NO_ACTION

        # Calculate velocity angle (direction we're currently moving)
        velocity_angle = math.atan2(velocity.y, velocity.x)

        # Calculate angle difference between orientation and velocity
        player_angle = self.game.player.geometry.angle
        angle_diff = (velocity_angle - player_angle) % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi

        # Determine which turn is shorter: toward velocity or away from it
        # angle_diff: positive = velocity is counterclockwise from orientation
        #            negative = velocity is clockwise from orientation

        abs_angle_diff = abs(angle_diff)

        # If close to aligned with velocity (within ~6 degrees)
        if abs_angle_diff < 0.1:
            # Facing velocity direction, decelerate to slow down
            return Action.DECELERATE

        # If close to opposite of velocity (within ~6 degrees of 180°)
        elif abs_angle_diff > math.pi - 0.1:
            # Facing opposite of velocity, accelerate to slow down
            return Action.ACCELERATE

        # Determine shorter turn direction
        # If angle_diff is between -90° and +90°, turning toward velocity is shorter
        # If angle_diff is beyond ±90°, turning away from velocity is shorter

        if abs_angle_diff < math.pi / 2:
            # Turning toward velocity is shorter
            # Once aligned, we'll DECELERATE
            if angle_diff > 0:
                return Action.TURN_LEFT  # Velocity is counterclockwise
            else:
                return Action.TURN_RIGHT  # Velocity is clockwise
        else:
            # Turning away from velocity is shorter (>90° difference)
            # Once aligned opposite, we'll ACCELERATE
            if angle_diff > 0:
                return Action.TURN_RIGHT  # Turn away (clockwise)
            else:
                return Action.TURN_LEFT  # Turn away (counterclockwise)

    def get_move(self) -> Action:
        """
        Analyze game state and return intelligent action.
        """
        # Decrement cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        strategy = self.get_strategy()
        match strategy:
            case Strategy.EVASIVE_ACTION:
                return self.evasive_action()
            case Strategy.SPEED_CONTROL:
                return self.speed_control()
            case Strategy.SHOOT_NEAREST:
                return self.shoot_nearest()
            case _:
                return Action.NO_ACTION
