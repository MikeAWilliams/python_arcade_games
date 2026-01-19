import random
import math

PLAYER_TURN_RATE = math.pi/3
PLAYER_ACCELERATION = 20
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
        result = vec2d(random.randint(-1, 1), random.randint(-1, 1))
        result.normalize()
        result = result.multiply(size)
        return result

def bounce(obj, width, height):
    if obj.pos.x - obj.radius < 0:
        obj.pos.x = obj.radius
        obj.vel.x *= -1
    elif obj.pos.x + obj.radius > width:
        obj.pos.x = width - obj.radius
        obj.vel.x *= -1
    if obj.pos.y - obj.radius < 0:
        obj.pos.y = obj.radius
        obj.vel.y *= -1
    elif obj.pos.y + obj.radius > height:
        obj.pos.y = height - obj.radius
        obj.vel.y *= -1

def generate_fair_asteroid_starting_positions(width, height, num, radius, exclusion_center, exclusion_radius):
    result = []
    while len(result) < num:
        pos = vec2d(random.randint(0, width), random.randint(0, height))
        if math.dist(pos, exclusion_center) > exclusion_radius:
            result.append(pos)
    return result

class Player():
    def __init__(self, pos: vec2d, radius: int):
        self.pos = pos
        self.vel = vec2d(0, 0)
        self.accel= vec2d(0, 0)
        self.radius = radius
        self.angle = math.pi/2
        self.angle_vel = 0
        self.bullets = []

    def update(self, dt):
        self.vel.x += self.accel.x * dt
        self.vel.y += self.accel.y * dt
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt
        self.angle += self.angle_vel * dt

        # Update bullets
        for bullet in self.bullets:
            bullet.update(dt)

    def turning_left(self):
        self.angle_vel = PLAYER_TURN_RATE

    def turning_right(self):
        self.angle_vel = -PLAYER_TURN_RATE

    def clear_turn(self):
        self.angle_vel = 0

    def accelerate(self):
        self.accel.x = math.cos(self.angle) * PLAYER_ACCELERATION
        self.accel.y = math.sin(self.angle) * PLAYER_ACCELERATION

    def decelerate(self):
        self.accel.x = -math.cos(self.angle) * PLAYER_ACCELERATION
        self.accel.y = -math.sin(self.angle) * PLAYER_ACCELERATION

    def clear_acc(self):
        self.accel.x = 0
        self.accel.y = 0

    def shoot(self):
        dir_vec = vec2d(math.cos(self.angle), math.sin(self.angle))
        bullet = Bullet(vec2d(self.pos.x + dir_vec.x * self.radius, self.pos.y + dir_vec.y * self.radius), vec2d(dir_vec.x * BULLET_SPEED, dir_vec.y * BULLET_SPEED), 1)
        self.bullets.append(bullet)

class Asteroid():
    def __init__(self, pos: vec2d, vel: vec2d, radius: int):
        self.pos = pos
        self.vel = vel
        self.radius = radius

    def update(self, dt):
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

class Bullet():
    def __init__(self, pos: vec2d, vel: vec2d, radius: int):
        self.pos = pos
        self.vel = vel
        self.radius = radius

    def update(self, dt):
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

class Game():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.player = Player(vec2d(width//2, height//2), 20)
        asteroid_centers = generate_fair_asteroid_starting_positions(width, height, 3, BIG_ASTEROID_RADIUS, self.player.pos, 100)
        self.asteroids = [Asteroid(center, vec2d.random_size(ASTEROID_SPEED), 90) for center in asteroid_centers]
        self.time_alive = 0
        self.player_alive = True
        self.player_score = 0

    def prune_bullets(self):
        self.player.bullets = [bullet for bullet in self.player.bullets if bullet.pos.x >= 0 and bullet.pos.x <= self.width and bullet.pos.y >= 0 and bullet.pos.y <= self.height]

    def check_player_asteroid_collision(self):
        for asteroid in self.asteroids:
            if math.dist(self.player.pos, asteroid.pos) <= self.player.radius + asteroid.radius:
                self.player_alive = False

    def spawn_new_asteroid_if_needed(self, parent):
        if parent.radius == BIG_ASTEROID_RADIUS:
            self.asteroids.append(Asteroid(parent.pos.copy(), vec2d.random_size(ASTEROID_SPEED), MEDIUM_ASTEROID_RADIUS))
            self.asteroids.append(Asteroid(parent.pos.copy(), vec2d.random_size(ASTEROID_SPEED), MEDIUM_ASTEROID_RADIUS))
        elif parent.radius == MEDIUM_ASTEROID_RADIUS:
            self.asteroids.append(Asteroid(parent.pos.copy(), vec2d.random_size(ASTEROID_SPEED), SMALL_ASTEROID_RADIUS))
            self.asteroids.append(Asteroid(parent.pos.copy(), vec2d.random_size(ASTEROID_SPEED), SMALL_ASTEROID_RADIUS))
            self.asteroids.append(Asteroid(parent.pos.copy(), vec2d.random_size(ASTEROID_SPEED), SMALL_ASTEROID_RADIUS))

    def check_bullet_asteroid_collision(self):
        for asteroid in self.asteroids:
            for bullet in self.player.bullets:
                if math.dist(asteroid.pos, bullet.pos) <= asteroid.radius + bullet.radius:
                    self.player_score += 1
                    self.asteroids.remove(asteroid)
                    self.player.bullets.remove(bullet)
                    self.spawn_new_asteroid_if_needed(asteroid)
                    break

    def update(self, dt):
        self.time_alive += dt
        self.player.update(dt)
        bounce(self.player, self.width, self.height)
        self.prune_bullets()

        for a in self.asteroids:
            a.update(dt)
            bounce(a, self.width, self.height)
        self.check_player_asteroid_collision()
        self.check_bullet_asteroid_collision()
