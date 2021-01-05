import arcade

import vector

ASTEROID_FILE_PATH = "../assets/asteroid.png"
ACCELERATION_GRAVITY = 1000000

class AsteroidField():
    def __init__(self, width, height):
        self.asteroids = arcade.SpriteList()
        self.asteroid_centers = []
        self.asteroid_centers.append(vector.Vector2D(width / 4, height / 2))
        self.asteroid_centers.append(vector.Vector2D(width - width / 4, height / 2))

        for center in self.asteroid_centers:
            asteroid = arcade.Sprite(ASTEROID_FILE_PATH, 0.2)
            asteroid.center_x = center.x
            asteroid.center_y = center.y
            self.asteroids.append(asteroid)


    def draw(self):
        self.asteroids.draw()

    def compute_asteriod_to_point_gravity(self, position):
        result = vector.Vector2D(0, 0)
        for center in self.asteroid_centers:
            direction = vector.Add(result, vector.Subtract(center, position))
            length = direction.length()
            # prevent division by small numebers from blowing stuff up
            if length > 50:
                direction.make_unit()
            else:
                length = 50
                direction = vector.Multipy(direction, 1.0 / length)

            result = vector.Add(result, vector.Multipy(direction, ACCELERATION_GRAVITY / (length * length)))

        return result