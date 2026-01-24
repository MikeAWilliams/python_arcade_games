"""
Example AI input method implementation.

This demonstrates how to create an alternative input method
that doesn't rely on keyboard input.
"""

import math
import random
from enum import Enum

from pyglet.math import Vec2

from game import BULLET_SPEED, PLAYER_RADIUS, SHOOT_COOLDOWN, Action, InputMethod, Vec2d


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
    """

    # Constants that need not be optimized
    MIN_DISTANCE_EPSILON = 0.0001
    TURN_DIRECTION_THRESHOLD = math.pi / 2
    TICK_DURATION = 1 / 60
    MIN_VELOCITY_THRESHOLD = 1

    def __init__(
        self,
        game,
        evasion_max_distance: float = 550,
        max_speed: float = 100,
        evasion_lookahead_ticks: int = 60,
        shoot_angle_tolerance: float = 0.05,
        movement_angle_tolerance: float = 0.1,
    ):
        self.game = game
        self.EVASION_MAX_DISTANCE = evasion_max_distance
        self.MAX_SPEED = max_speed
        self.EVASION_LOOKAHEAD_TICKS = evasion_lookahead_ticks
        self.SHOOT_ANGLE_TOLERANCE = shoot_angle_tolerance
        self.MOVEMENT_ANGLE_TOLERANCE = movement_angle_tolerance

    def predict_intercept(self, player_pos, asteroid_pos, asteroid_vel):
        """
        Calculate the intercept point for a moving asteroid using naive time estimate.
        Returns the angle to aim at.
        """
        # Calculate distance to asteroid
        distance = player_pos.distance(asteroid_pos)

        # Naive time estimate: time for bullet to reach current asteroid position
        time = distance / BULLET_SPEED

        # Predict where asteroid will be at that time
        pred_x = asteroid_pos.x + asteroid_vel.x * time
        pred_y = asteroid_pos.y + asteroid_vel.y * time

        # Return angle to predicted position
        return math.atan2(pred_y - player_pos.y, pred_x - player_pos.x)

    def get_strategy(self) -> tuple[Strategy, list]:
        """
        Determine which strategy to use based on game state.
        Uses tick-by-tick collision prediction to identify dangerous asteroids.
        Priority: EVASIVE_ACTION > SPEED_CONTROL > SHOOT_NEAREST

        Returns:
            tuple[Strategy, list]: Strategy to use and list of dangerous asteroids
        """
        if not self.game.asteroids:
            return Strategy.SHOOT_NEAREST, []

        # Use set for O(1) lookup instead of list O(n)
        dangerous_set = set()

        # Pre-calculate player data to avoid repeated attribute access
        player_x = self.game.player.geometry.pos.x
        player_y = self.game.player.geometry.pos.y
        player_vx = self.game.player.vel.x
        player_vy = self.game.player.vel.y
        player_radius = self.game.player.geometry.radius

        # Loop asteroids on outside so we can break early per asteroid
        for asteroid in self.game.asteroids:
            # Pre-calculate asteroid data
            ast_x = asteroid.geometry.pos.x
            ast_y = asteroid.geometry.pos.y
            ast_vx = asteroid.vel.x
            ast_vy = asteroid.vel.y
            ast_radius = asteroid.geometry.radius

            # Pre-calculate combined radius squared (constant for this asteroid)
            r_sum_sq = (player_radius + ast_radius) ** 2

            # Check each tick for this asteroid
            for tick in range(1, self.EVASION_LOOKAHEAD_TICKS + 1):
                look_ahead_t = self.TICK_DURATION * tick

                # Project positions (inline, no Vec2d creation)
                future_px = player_x + player_vx * look_ahead_t
                future_py = player_y + player_vy * look_ahead_t
                future_ax = ast_x + ast_vx * look_ahead_t
                future_ay = ast_y + ast_vy * look_ahead_t

                # Check collision using squared distance
                dx = future_px - future_ax
                dy = future_py - future_ay
                dist_sq = dx * dx + dy * dy

                if dist_sq <= r_sum_sq:
                    dangerous_set.add(asteroid)
                    break  # Found collision, check next asteroid

        # Convert set to list
        dangerous_asteroids = list(dangerous_set)

        # Return strategy based on prediction
        if dangerous_asteroids:
            return Strategy.EVASIVE_ACTION, dangerous_asteroids

        # Check if player velocity is too high
        velocity_magnitude = self.game.player.vel.size()
        if velocity_magnitude > self.MAX_SPEED:
            return Strategy.SPEED_CONTROL, []

        return Strategy.SHOOT_NEAREST, []

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
            closest_asteroid = min(
                asteroids,
                key=lambda a: a.geometry.pos.distance2(player_pos),
            )

            # Calculate predicted intercept angle
            target_angle = self.predict_intercept(
                player_pos, closest_asteroid.geometry.pos, closest_asteroid.vel
            )

            # Normalize angle difference
            angle_diff = (target_angle - player_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            # Strategy: turn to face predicted position and shoot
            if (
                abs(angle_diff) < self.SHOOT_ANGLE_TOLERANCE
                and self.game.shoot_cooldown == 0
            ):
                # Aimed at predicted position, shoot!
                return Action.SHOOT
            elif angle_diff > 0:
                return Action.TURN_LEFT
            else:
                return Action.TURN_RIGHT

        return Action.NO_ACTION

    def evasive_action(self, dangerous_asteroids: list) -> Action:
        """
        Strategy: Compute weighted average threat vector from dangerous asteroids
        and choose the shorter turn to evade (either face toward and decelerate,
        or face away and accelerate).

        Args:
            dangerous_asteroids: Pre-identified asteroids that will collide with player

        Weight calculation: Weight is inversely proportional to distance
        (closer asteroids have higher weight).

        Evasion logic:
        - If threat is <90° away: Turn toward threat, then DECELERATE
        - If threat is >90° away: Turn away from threat, then ACCELERATE
        - Chooses the shorter turn for faster evasion response
        """
        if not dangerous_asteroids:
            return Action.NO_ACTION

        # Compute weighted threat vector using Vec2d
        weighted_vector = Vec2d(0.0, 0.0)
        total_weight = 0.0
        look_ahead_t = self.TICK_DURATION * self.EVASION_LOOKAHEAD_TICKS
        player_pos = Vec2d(
            self.game.player.geometry.pos.x + self.game.player.vel.x * look_ahead_t,
            self.game.player.geometry.pos.y + self.game.player.vel.y * look_ahead_t,
        )

        for asteroid in dangerous_asteroids:
            asteroid_pos = Vec2d(
                asteroid.geometry.pos.x + asteroid.vel.x * look_ahead_t,
                asteroid.geometry.pos.y + asteroid.vel.y * look_ahead_t,
            )
            distance = player_pos.distance(asteroid_pos)
            distance -= self.game.player.geometry.radius
            distance -= asteroid.geometry.radius
            if distance <= 0:
                distance = self.MIN_DISTANCE_EPSILON

            # Weight is inversely proportional to distance
            # Use EVASION_MAX_DISTANCE as reference for relative weighting
            weight = max(0, self.EVASION_MAX_DISTANCE - distance)

            # Direction vector from player to asteroid (normalized)
            direction = Vec2d(
                asteroid.geometry.pos.x - self.game.player.geometry.pos.x,
                asteroid.geometry.pos.y - self.game.player.geometry.pos.y,
            )
            normalized_direction = direction.normalize()

            # Accumulate weighted direction vectors
            weighted_vector.x += normalized_direction.x * weight
            weighted_vector.y += normalized_direction.y * weight
            total_weight += weight

        # If no weight accumulated (shouldn't happen with dangerous asteroids)
        if total_weight == 0:
            return Action.NO_ACTION

        # Compute average threat direction
        threat_vector = Vec2d(
            weighted_vector.x / total_weight, weighted_vector.y / total_weight
        )
        threat_angle = math.atan2(threat_vector.y, threat_vector.x)

        # Calculate angle difference to face the threat
        angle_diff = (threat_angle - self.game.player.geometry.angle) % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi

        # Determine which turn is shorter: toward threat or away from threat
        # angle_diff: positive = threat is counterclockwise from orientation
        #            negative = threat is clockwise from orientation

        abs_angle_diff = abs(angle_diff)

        # If close to aligned with threat (within ~6 degrees)
        if abs_angle_diff < self.MOVEMENT_ANGLE_TOLERANCE:
            # Facing threat direction, decelerate to move away
            return Action.DECELERATE

        # If close to opposite of threat (within ~6 degrees of 180°)
        elif abs_angle_diff > math.pi - self.MOVEMENT_ANGLE_TOLERANCE:
            # Facing opposite of threat, accelerate to move away
            return Action.ACCELERATE

        # Determine shorter turn direction
        # If angle_diff is between -90° and +90°, turning toward threat is shorter
        # If angle_diff is beyond ±90°, turning away from threat is shorter

        if abs_angle_diff < self.TURN_DIRECTION_THRESHOLD:
            # Turning toward threat is shorter
            # Once aligned, we'll DECELERATE
            if angle_diff > 0:
                return Action.TURN_LEFT  # Threat is counterclockwise
            else:
                return Action.TURN_RIGHT  # Threat is clockwise
        else:
            # Turning away from threat is shorter (>90° difference)
            # Once aligned opposite, we'll ACCELERATE
            if angle_diff > 0:
                return Action.TURN_RIGHT  # Turn away (clockwise)
            else:
                return Action.TURN_LEFT  # Turn away (counterclockwise)

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
        if velocity.size() < self.MIN_VELOCITY_THRESHOLD:
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
        if abs_angle_diff < self.MOVEMENT_ANGLE_TOLERANCE:
            # Facing velocity direction, decelerate to slow down
            return Action.DECELERATE

        # If close to opposite of velocity (within ~6 degrees of 180°)
        elif abs_angle_diff > math.pi - self.MOVEMENT_ANGLE_TOLERANCE:
            # Facing opposite of velocity, accelerate to slow down
            return Action.ACCELERATE

        # Determine shorter turn direction
        # If angle_diff is between -90° and +90°, turning toward velocity is shorter
        # If angle_diff is beyond ±90°, turning away from velocity is shorter

        if abs_angle_diff < self.TURN_DIRECTION_THRESHOLD:
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
        strategy, dangerous_asteroids = self.get_strategy()
        match strategy:
            case Strategy.EVASIVE_ACTION:
                return self.evasive_action(dangerous_asteroids)
            case Strategy.SPEED_CONTROL:
                return self.speed_control()
            case Strategy.SHOOT_NEAREST:
                return self.shoot_nearest()
            case _:
                return Action.NO_ACTION
