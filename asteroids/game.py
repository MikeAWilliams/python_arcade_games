import random
import math

PLAYER_TURN_RATE = math.pi/3
PLAYER_ACCELERATION = 20
PLAYER_RADIUS = 20
PLAYER_EXCLUSION_RADIUS = PLAYER_RADIUS*5
BULLET_SPEED = 500
ASTEROID_SPEED = 50
BIG_ASTEROID_RADIUS = 90
MEDIUM_ASTEROID_RADIUS = 60
SMALL_ASTEROID_RADIUS = 30

class vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((self.x, self.y))

    def size(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        size = self.size()
        if size == 0:
            return vec2d(0, 0)
        return vec2d(self.x / size, self.y / size)

    def multiply(self, scalar):
        return vec2d(self.x * scalar, self.y * scalar)

    def copy(self):
        return vec2d(self.x, self.y)

    @staticmethod
    def random_size(size):
        result = vec2d(random.uniform(-1, 1), random.uniform(-1, 1))
        result.normalize()
        result = result.multiply(size)
        return result

class geometry_object:
    def __init__(self, pos: vec2d, radius: int, angle: float=0):
        self.pos = pos
        self.radius = radius
        self.angle = angle

    def intersects(self, other):
        return math.dist(self.pos, other.pos) <= self.radius + other.radius

    def copy(self):
        return geometry_object(self.pos.copy(), self.radius, self.angle)

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

def generate_fair_asteroid_starting_geometry(width: int, height: int, num: int, radius: int, exclusion_geometry: geometry_object):
    result = []
    while len(result) < num:
        geo = geometry_object(vec2d(random.randint(radius, width-radius), random.randint(radius, height-radius)), radius)
        if not geo.intersects(exclusion_geometry):
            to_close = False
            for other in result:
                if geo.intersects(other):
                    to_close = True
                    break

            if not to_close:
                result.append(geo)
    return result

class Player():
    def __init__(self, geometry :geometry_object):
        self.geometry = geometry
        self.geometry.angle = math.pi/2
        self.vel = vec2d(0, 0)
        self.accel= vec2d(0, 0)
        self.angle_vel = 0

    def update(self, dt):
        self.vel.x += self.accel.x * dt
        self.vel.y += self.accel.y * dt
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt
        self.geometry.angle += self.angle_vel * dt


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

class Asteroid():
    def __init__(self, geometer: geometry_object, vel: vec2d):
        self.geometry = geometer
        self.vel = vel

    def update(self, dt):
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt

class Bullet():
    def __init__(self, geometry: geometry_object, vel: vec2d):
        self.geometry = geometry
        self.vel = vel

    def update(self, dt):
        self.geometry.pos.x += self.vel.x * dt
        self.geometry.pos.y += self.vel.y * dt

class Game():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.player = Player(geometry_object(vec2d(width//2, height//2), PLAYER_RADIUS))
        self.bullets = []
        asteroid_centers = generate_fair_asteroid_starting_geometry(width, height, 3, BIG_ASTEROID_RADIUS, geometry_object(self.player.geometry.pos, PLAYER_EXCLUSION_RADIUS))
        self.asteroids = [Asteroid(geo, vec2d.random_size(ASTEROID_SPEED)) for geo in asteroid_centers]
        self.time_alive = 0
        self.player_alive = True
        self.player_score = 0

    def prune_bullets(self):
        self.bullets = [bullet for bullet in self.bullets if bullet.geometry.pos.x >= 0 and bullet.geometry.pos.x <= self.width and bullet.geometry.pos.y >= 0 and bullet.geometry.pos.y <= self.height]

    def check_player_asteroid_collision(self):
        for asteroid in self.asteroids:
            if asteroid.geometry.intersects(self.player.geometry):
                self.player_alive = False

    def spawn_new_asteroid_if_needed(self, parent):
        if parent.geometry.radius == BIG_ASTEROID_RADIUS:
            self.asteroids.append(Asteroid(geometry_object(parent.geometry.pos.copy(), MEDIUM_ASTEROID_RADIUS), vec2d.random_size(ASTEROID_SPEED)))
            self.asteroids.append(Asteroid(geometry_object(parent.geometry.pos.copy(), MEDIUM_ASTEROID_RADIUS), vec2d.random_size(ASTEROID_SPEED)))
        elif parent.geometry.radius == MEDIUM_ASTEROID_RADIUS:
            self.asteroids.append(Asteroid(geometry_object(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS), vec2d.random_size(ASTEROID_SPEED)))
            self.asteroids.append(Asteroid(geometry_object(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS), vec2d.random_size(ASTEROID_SPEED)))
            self.asteroids.append(Asteroid(geometry_object(parent.geometry.pos.copy(), SMALL_ASTEROID_RADIUS), vec2d.random_size(ASTEROID_SPEED)))

    def check_bullet_asteroid_collision(self):
        for asteroid in self.asteroids:
            for bullet in self.bullets:
                if asteroid.geometry.intersects(bullet.geometry):
                    self.player_score += 1
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    self.spawn_new_asteroid_if_needed(asteroid)
                    break

    def update(self, dt):
        self.time_alive += dt
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
        dir_vec = vec2d(math.cos(self.player.geometry.angle), math.sin(self.player.geometry.angle))
        bullet = Bullet(geometry_object(vec2d(self.player.geometry.pos.x + dir_vec.x * self.player.geometry.radius, self.player.geometry.pos.y + dir_vec.y * self.player.geometry.radius),1), vec2d(dir_vec.x * BULLET_SPEED, dir_vec.y * BULLET_SPEED))
        self.bullets.append(bullet)

    def no_action(self):
        pass
