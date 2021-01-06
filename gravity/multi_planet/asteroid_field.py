import arcade
import random
import math
import numpy as np

import vector

ASTEROID_FILE_PATH = "../assets/asteroid.png"
ACCELERATION_GRAVITY = 10

ASTEROID_MASS_MEAN = 10000
ASTEROID_MASS_STD =   2000
ASTEROID_MASS_UPPER_OUTLIER = ASTEROID_MASS_MEAN + 3 * ASTEROID_MASS_STD
ASTEROID_MASS_LOWER_OUTLIER = ASTEROID_MASS_MEAN - 3 * ASTEROID_MASS_STD
ASTEROID_MASS_RANGE = ASTEROID_MASS_UPPER_OUTLIER - ASTEROID_MASS_LOWER_OUTLIER
ASTEROID_MAX_SCALE = 0.1
ASTEROID_MIN_SCALE = 0.025

class AsteroidData():
    def __init__(self, center, mass):
        self.center = center
        self.mass = mass

def MapMassToScale(mass):
    if mass > ASTEROID_MASS_UPPER_OUTLIER:
        mass = ASTEROID_MASS_UPPER_OUTLIER
    if mass < ASTEROID_MASS_LOWER_OUTLIER:
        mass = ASTEROID_MASS_LOWER_OUTLIER
    return np.interp(mass, [ASTEROID_MASS_LOWER_OUTLIER, ASTEROID_MASS_UPPER_OUTLIER], [ASTEROID_MIN_SCALE, ASTEROID_MAX_SCALE])

def create_random_asteroid(width, height):
    data = AsteroidData(vector.Vector2D(random.randint(1, width), \
        random.randint(1, height)), \
            np.random.normal(ASTEROID_MASS_MEAN, ASTEROID_MASS_STD))
    sprite = arcade.Sprite(ASTEROID_FILE_PATH, MapMassToScale(data.mass))
    sprite.center_x = data.center.x
    sprite.center_y = data.center.y
    sprite.radians = random.uniform(0, 2 * math.pi)
    return data, sprite

class AsteroidField():
    def __init__(self, width, height, count):
        self.asteroids = arcade.SpriteList()
        self.generate_random_non_intersecting_asteroids(width, height, count)
        

    def generate_random_non_intersecting_asteroids(self, width, height, count):
        self.asteroid_data = []
        for _ in range(count):
            collision_list = [3]
            while len(collision_list) > 0:
                data, sprite = create_random_asteroid(width, height)
                collision_list = arcade.check_for_collision_with_list(sprite, self.asteroids)

            self.asteroids.append(sprite)
            self.asteroid_data.append(data)

    def get_collision_sprites(self):
        return self.asteroids

    def draw(self):
        self.asteroids.draw()

    def compute_asteriod_to_point_gravity(self, position):
        result = vector.Vector2D(0, 0)
        for dp in self.asteroid_data:
            direction = vector.Add(result, vector.Subtract(dp.center, position))
            length = direction.length()
            # prevent division by small numebers from blowing stuff up
            if length > 50:
                direction.make_unit()
            else:
                length = 50
                direction = vector.Multipy(direction, 1.0 / length)

            result = vector.Add(result, vector.Multipy(direction, dp.mass * ACCELERATION_GRAVITY / (length * length)))

        return result
