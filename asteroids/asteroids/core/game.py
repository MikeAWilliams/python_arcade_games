import math
import random
from abc import ABC, abstractmethod
from enum import Enum
from gc import callbacks


class Action(Enum):
    """Enumeration of possible game actions"""

    TURN_LEFT = 0
    TURN_RIGHT = 1
    ACCELERATE = 2
    DECELERATE = 3
    SHOOT = 4
    NO_ACTION = 5


class InputMethod(ABC):
    """Abstract base class for input methods"""

    @abstractmethod
    def get_move(self) -> Action:
        """
        Returns the next action to take.
        Should return Action.NO_ACTION if no action should be taken.
        """
        pass


PLAYER_TURN_RATE = math.pi
PLAYER_ACCELERATION = 180
PLAYER_RADIUS = 20
PLAYER_EXCLUSION_RADIUS = PLAYER_RADIUS * 5
BULLET_SPEED = 500
ASTEROID_BASE_SPEED = 100
ASTEROID_SPEED_INCREMENT = 1.1
BIG_ASTEROID_RADIUS = 90
MEDIUM_ASTEROID_RADIUS = 60
SMALL_ASTEROID_RADIUS = 30
SHOOT_COOLDOWN = 20


class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((self.x, self.y))

    def size(self):
        return math.sqrt(self.x**2 + self.y**2)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance2(self, other):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def normalize(self):
        size = self.size()
        if size == 0:
            return Vec2d(0, 0)
        return Vec2d(self.x / size, self.y / size)

    def multiply(self, scalar):
        return Vec2d(self.x * scalar, self.y * scalar)

    def copy(self):
        return Vec2d(self.x, self.y)

    @staticmethod
    def random_size(size):
        result = Vec2d(random.uniform(-1, 1), random.uniform(-1, 1))
        result = result.normalize()
        result = result.multiply(size)
        return result


class GeometryObject:
    def __init__(self, pos: Vec2d, radius: int, angle: float = 0):
        self.pos = pos
        self.radius = radius
        self.angle = angle

    def intersects(self, other):
        r = self.radius + other.radius
        return self.pos.distance2(other.pos) <= r * r

    def copy(self):
        return GeometryObject(self.pos.copy(), self.radius, self.angle)


class GeometryState:
    def __init__(
        self,
        player: GeometryObject,
        asteroids: list[GeometryObject],
        bullets: list[GeometryObject],
    ):
        self.player = player
        self.asteroids = asteroids
        self.bullets = bullets


def bounce(obj, width, height):
    if obj.geometry.pos.x - obj.geometry.radius < 0:
        obj.geometry.pos.x = obj.geometry.radius
        obj.vel.x *= -1
    elif obj.geometry.pos.x + obj.geometry.radius > width:
        obj.geometry.pos.x = width - obj.geometry.radius
        obj.vel.x *= -1
    if obj.geometry.pos.y - obj.geometry.radius < 0:
        obj.geometry.pos.y = obj.geometry.radius
        obj.vel.y *= -1
    elif obj.geometry.pos.y + obj.geometry.radius > height:
        obj.geometry.pos.y = height - obj.geometry.radius
        obj.vel.y *= -1


def generate_fair_asteroid_starting_geometry(
    width: int, height: int, num: int, radius: int, exclusion_geometry: GeometryObject
):
    result = []
    while len(result) < num:
        geo = GeometryObject(
            Vec2d(
                random.randint(radius, width - radius),
                random.randint(radius, height - radius),
            ),
            radius,
        )
        if not geo.intersects(exclusion_geometry):
            too_close = False
            for other in result:
                if geo.intersects(other):
                    too_close = True
                    break

            if not too_close:
                result.append(geo)
    return result


def generate_crisis_asteroids(
    width: int,
    height: int,
    player_pos: Vec2d,
    player_angle: float,
    num_asteroids: int,
    asteroid_speed: float,
) -> list:
    """Generate asteroid placements for crisis mode.

    Each asteroid is placed at a distance such that it arrives at the player
    in exactly (num_asteroids * 90 degrees) / PLAYER_TURN_RATE seconds.
    For 1 asteroid, placement excludes the 90-degree forward arc.
    For 2+, placement is at random angles.
    Trajectories are offset to graze the player, not aimed dead center.
    """
    combined_radii = PLAYER_RADIUS + SMALL_ASTEROID_RADIUS
    time_to_arrive = num_asteroids * (math.pi / 2) / PLAYER_TURN_RATE
    distance = asteroid_speed * time_to_arrive + combined_radii

    asteroids = []
    for i in range(num_asteroids):
        # Choose placement angle
        if num_asteroids == 1:
            # Exclude ±45 degrees from player heading
            excluded_half = math.pi / 4
            offset = random.uniform(excluded_half, 2 * math.pi - excluded_half)
            placement_angle = player_angle + offset
        else:
            placement_angle = random.uniform(0, 2 * math.pi)

        # Position the asteroid
        ax = player_pos.x + distance * math.cos(placement_angle)
        ay = player_pos.y + distance * math.sin(placement_angle)

        # Clamp to screen bounds and recalculate from actual position
        ax = max(SMALL_ASTEROID_RADIUS, min(width - SMALL_ASTEROID_RADIUS, ax))
        ay = max(SMALL_ASTEROID_RADIUS, min(height - SMALL_ASTEROID_RADIUS, ay))

        # Velocity direction: toward player but offset to graze
        angle_to_player = math.atan2(player_pos.y - ay, player_pos.x - ax)
        actual_distance = math.sqrt((player_pos.x - ax) ** 2 + (player_pos.y - ay) ** 2)

        # Miss distance between PLAYER_RADIUS and SMALL_ASTEROID_RADIUS
        # so the asteroid grazes rather than hitting dead-on
        miss_distance = random.uniform(PLAYER_RADIUS, SMALL_ASTEROID_RADIUS)
        if actual_distance > 0:
            offset_angle = math.asin(min(miss_distance / actual_distance, 1.0))
        else:
            offset_angle = 0

        # Random side
        if random.random() < 0.5:
            offset_angle = -offset_angle

        vel_angle = angle_to_player + offset_angle
        vel = Vec2d(
            math.cos(vel_angle) * asteroid_speed,
            math.sin(vel_angle) * asteroid_speed,
        )

        geo = GeometryObject(Vec2d(ax, ay), SMALL_ASTEROID_RADIUS)
        asteroids.append(Asteroid(geo, vel, id=i))

    return asteroids


class Player:
    def __init__(self, geometry: GeometryObject):
        self.geometry = geometry
        self.geometry.angle = math.pi / 2
        self.vel = Vec2d(0, 0)
        self.accel = Vec2d(0, 0)
        self.angle_vel = 0

    def update(self, dt):
        self.vel.x += self.accel.x * dt
        self.vel.y += self.accel.y * dt
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt
        self.geometry.angle += self.angle_vel * dt
        self.geometry.angle %= 2 * math.pi

    def copy_geometry(self):
        return self.geometry.copy()

    def turning_left(self):
        self.angle_vel = PLAYER_TURN_RATE

    def turning_right(self):
        self.angle_vel = -PLAYER_TURN_RATE

    def clear_turn(self):
        self.angle_vel = 0

    def accelerate(self):
        self.accel.x = math.cos(self.geometry.angle) * PLAYER_ACCELERATION
        self.accel.y = math.sin(self.geometry.angle) * PLAYER_ACCELERATION

    def decelerate(self):
        self.accel.x = -math.cos(self.geometry.angle) * PLAYER_ACCELERATION
        self.accel.y = -math.sin(self.geometry.angle) * PLAYER_ACCELERATION

    def clear_acc(self):
        self.accel.x = 0
        self.accel.y = 0

    def no_action(self):
        pass


def calc_asteroid_id(parent_id: int, child_number: int):
    """
    Calculate the unique ID for an asteroid.
    The 3 in first 2x in second and 3x in third generation is hard coded here
    """
    if parent_id < 3:
        children_per_parent = 2
    else:
        children_per_parent = 3

    if parent_id < 3:
        offset = 2 * parent_id
    else:
        offset = 6 + 3 * (parent_id - 3)

    if child_number < 0 or child_number > children_per_parent:
        raise ValueError("Invalid child_number for this parent")

    return offset + child_number + 3


class Asteroid:
    def __init__(self, geometry: GeometryObject, vel: Vec2d, id: int):
        self.geometry = geometry
        self.vel = vel
        # cary a unique id for the sake of encoding state in the nn ai
        self.id = id

    def update(self, dt):
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt

    def copy_geometry(self):
        return self.geometry.copy()


class Bullet:
    def __init__(self, geometry: GeometryObject, vel: Vec2d):
        self.geometry = geometry
        self.vel = vel

    def update(self, dt):
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt

    def copy_geometry(self):
        return self.geometry.copy()


class Game:
    def __init__(
        self, width, height, starting_wave: int = 1, crisis_mode: bool = False
    ):
        self.width = width
        self.height = height
        self.crisis_mode = crisis_mode
        self.wave_number = starting_wave
        self.player = Player(
            GeometryObject(Vec2d(width // 2, height // 2), PLAYER_RADIUS)
        )
        self.bullets = []
        self.asteroid_speed_multiplier = ASTEROID_SPEED_INCREMENT ** (
            starting_wave - 1
        )  # Increases with each wave
        if crisis_mode:
            self.player.geometry.angle = random.uniform(0, 2 * math.pi)
            num_asteroids = random.randint(1, 5)
            current_speed = ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier
            self.asteroids = generate_crisis_asteroids(
                width,
                height,
                self.player.geometry.pos,
                self.player.geometry.angle,
                num_asteroids,
                current_speed,
            )
        else:
            asteroid_centers = generate_fair_asteroid_starting_geometry(
                width,
                height,
                3,
                BIG_ASTEROID_RADIUS,
                GeometryObject(self.player.geometry.pos, PLAYER_EXCLUSION_RADIUS),
            )
            self.asteroids = [
                Asteroid(
                    geo,
                    Vec2d.random_size(
                        ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier
                    ),
                    id=i,
                )
                for i, geo in enumerate(asteroid_centers)
            ]
        self.time_alive = 0
        self.player_alive = True
        self.player_score = 0
        self.shoot_cooldown = 0

    def prune_bullets(self):
        self.bullets = [
            bullet
            for bullet in self.bullets
            if bullet.geometry.pos.x >= 0
            and bullet.geometry.pos.x <= self.width
            and bullet.geometry.pos.y >= 0
            and bullet.geometry.pos.y <= self.height
        ]

    def check_player_asteroid_collision(self):
        for asteroid in self.asteroids:
            if asteroid.geometry.intersects(self.player.geometry):
                self.player_alive = False

    def spawn_new_asteroid_if_needed(self, parent):
        current_speed = ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier
        if parent.geometry.radius == BIG_ASTEROID_RADIUS:
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), MEDIUM_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                    calc_asteroid_id(parent.id, 0),
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), MEDIUM_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                    calc_asteroid_id(parent.id, 1),
                )
            )
        elif parent.geometry.radius == MEDIUM_ASTEROID_RADIUS:
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                    calc_asteroid_id(parent.id, 0),
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                    calc_asteroid_id(parent.id, 1),
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                    calc_asteroid_id(parent.id, 2),
                )
            )

    def check_bullet_asteroid_collision(self):
        for asteroid in self.asteroids:
            for bullet in self.bullets:
                if asteroid.geometry.intersects(bullet.geometry):
                    if not self.crisis_mode:
                        self.player_score += 1
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    self.spawn_new_asteroid_if_needed(asteroid)
                    break

        # Check if all asteroids are destroyed and start new wave
        if len(self.asteroids) == 0:
            if self.crisis_mode:
                self.player_score += 27
            self.start_new_wave()

    def update(self, dt):
        self.time_alive += dt
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        self.player.update(dt)
        for bullet in self.bullets:
            bullet.update(dt)

        bounce(self.player, self.width, self.height)
        self.prune_bullets()

        for a in self.asteroids:
            a.update(dt)
            bounce(a, self.width, self.height)
        self.check_player_asteroid_collision()
        self.check_bullet_asteroid_collision()

    def geometry_state(self) -> GeometryState:
        asteroids = [a.geometry.copy() for a in self.asteroids]
        bullets = [b.geometry.copy() for b in self.bullets]
        return GeometryState(self.player.geometry.copy(), asteroids, bullets)

    def turning_left(self):
        self.player.turning_left()

    def turning_right(self):
        self.player.turning_right()

    def clear_turn(self):
        self.player.clear_turn()

    def accelerate(self):
        self.player.accelerate()

    def decelerate(self):
        self.player.decelerate()

    def clear_acc(self):
        self.player.clear_acc()

    def shoot(self):
        if self.shoot_cooldown > 0:
            return
        dir_vec = Vec2d(
            math.cos(self.player.geometry.angle), math.sin(self.player.geometry.angle)
        )
        bullet = Bullet(
            GeometryObject(
                Vec2d(
                    self.player.geometry.pos.x
                    + dir_vec.x * self.player.geometry.radius,
                    self.player.geometry.pos.y
                    + dir_vec.y * self.player.geometry.radius,
                ),
                1,
            ),
            Vec2d(dir_vec.x * BULLET_SPEED, dir_vec.y * BULLET_SPEED),
        )
        self.bullets.append(bullet)
        self.shoot_cooldown = SHOOT_COOLDOWN

    def no_action(self):
        pass

    def start_new_wave(self):
        """
        Start a new wave of asteroids with increased difficulty.
        Resets player position and velocity, increases asteroid speed.
        """
        # Increase difficulty
        self.asteroid_speed_multiplier *= ASTEROID_SPEED_INCREMENT
        self.wave_number += 1

        # Reset player to center with zero velocity
        self.player.geometry.pos.x = self.width // 2
        self.player.geometry.pos.y = self.height // 2
        self.player.vel.x = 0
        self.player.vel.y = 0
        self.player.accel.x = 0
        self.player.accel.y = 0
        self.player.angle_vel = 0

        current_speed = ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier

        if self.crisis_mode:
            self.player.geometry.angle = random.uniform(0, 2 * math.pi)
            num_asteroids = random.randint(1, 5)
            self.asteroids = generate_crisis_asteroids(
                self.width,
                self.height,
                self.player.geometry.pos,
                self.player.geometry.angle,
                num_asteroids,
                current_speed,
            )
        else:
            self.player.geometry.angle = math.pi / 2  # Facing up
            asteroid_centers = generate_fair_asteroid_starting_geometry(
                self.width,
                self.height,
                3,
                BIG_ASTEROID_RADIUS,
                GeometryObject(self.player.geometry.pos, PLAYER_EXCLUSION_RADIUS),
            )
            self.asteroids = [
                Asteroid(geo, Vec2d.random_size(current_speed), id)
                for id, geo in enumerate(asteroid_centers)
            ]

        # Clear bullets
        self.bullets = []
