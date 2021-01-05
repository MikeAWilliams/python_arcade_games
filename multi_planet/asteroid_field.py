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

class AsteroidField():
    def __init__(self, width, height, count):
        self.asteroids = arcade.SpriteList()
        self.generate_random_asteroids(width, height, count)
        self.create_sprites() 
        
    def create_sprites(self):
        for dp in self.asteroid_data:
            asteroid = arcade.Sprite(ASTEROID_FILE_PATH, MapMassToScale(dp.mass))
            asteroid.center_x = dp.center.x
            asteroid.center_y = dp.center.y
            asteroid.radians = random.uniform(0, 2 * math.pi)
            self.asteroids.append(asteroid)

    def generate_random_asteroids(self, width, height, count):
        self.asteroid_data = []
        for _ in range(count):
            self.asteroid_data.append( \
                AsteroidData(vector.Vector2D(random.randint(1, width), random.randint(1, height)), \
                    np.random.normal(ASTEROID_MASS_MEAN, ASTEROID_MASS_STD)))

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
