import math
import random
from abc import ABC, abstractmethod
from enum import Enum


class Action(Enum):
    """Enumeration of possible game actions"""

    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    SHOOT = "shoot"
    NO_ACTION = "no_action"


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
        return math.dist(self.pos, other.pos) <= self.radius + other.radius

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


class Asteroid:
    def __init__(self, geometry: GeometryObject, vel: Vec2d):
        self.geometry = geometry
        self.vel = vel

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
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.player = Player(
            GeometryObject(Vec2d(width // 2, height // 2), PLAYER_RADIUS)
        )
        self.bullets = []
        self.asteroid_speed_multiplier = 1.0  # Increases with each wave
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
                Vec2d.random_size(ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier),
            )
            for geo in asteroid_centers
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
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), MEDIUM_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                )
            )
        elif parent.geometry.radius == MEDIUM_ASTEROID_RADIUS:
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                )
            )
            self.asteroids.append(
                Asteroid(
                    GeometryObject(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS),
                    Vec2d.random_size(current_speed),
                )
            )

    def check_bullet_asteroid_collision(self):
        for asteroid in self.asteroids:
            for bullet in self.bullets:
                if asteroid.geometry.intersects(bullet.geometry):
                    self.player_score += 1
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    self.spawn_new_asteroid_if_needed(asteroid)
                    break

        # Check if all asteroids are destroyed and start new wave
        if len(self.asteroids) == 0:
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

        # Reset player to center with zero velocity
        self.player.geometry.pos.x = self.width // 2
        self.player.geometry.pos.y = self.height // 2
        self.player.geometry.angle = math.pi / 2  # Facing up
        self.player.vel.x = 0
        self.player.vel.y = 0
        self.player.accel.x = 0
        self.player.accel.y = 0
        self.player.angle_vel = 0

        # Spawn 3 new large asteroids
        current_speed = ASTEROID_BASE_SPEED * self.asteroid_speed_multiplier
        asteroid_centers = generate_fair_asteroid_starting_geometry(
            self.width,
            self.height,
            3,
            BIG_ASTEROID_RADIUS,
            GeometryObject(self.player.geometry.pos, PLAYER_EXCLUSION_RADIUS),
        )
        self.asteroids = [
            Asteroid(geo, Vec2d.random_size(current_speed)) for geo in asteroid_centers
        ]

        # Clear bullets
        self.bullets = []
