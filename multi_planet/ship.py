import arcade
import math
import random

import vector

SHIP_FILE_WIDTH = 840
SHIP_FILE_HEIGTH = 1510
SHIP_PATH = "../assets/rocket_white_fire.png"
SHIP_FIRE_PATH = "../assets/rocket_red_fire.png" 
SHIP_CRASHED_PATH = "../assets/rocket_crash.png"

ACCELERATION_ROCKET = 300
SHIP_CRASH_VELOCITY = -200
ANGULAR_SPEED = 2 * math.pi / 300

class Ship():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scale = 0.05
        self.init_position = vector.Vector2D(random.randint(1, width), random.randint(1, height))
        self.position = self.init_position.copy()

        self.ship = arcade.Sprite(SHIP_PATH, self.scale)
        self.ship_fire = arcade.Sprite(SHIP_FIRE_PATH, self.scale)
    
    def setup(self):
        # ship state
        self.position = self.init_position.copy()
        self.velocity = vector.Vector2D(0, 0) 
        self.gravity_acceleration = vector.Vector2D(0, 0)
        self.rocket_acceleration = vector.Vector2D(0, 0)
        self.ship_engine_on = False
        self.radians = 0
        self.radian_speed = 0

        # create the ship off sprite
        self.ship.radians = 0
        self.ship.center_x = self.position.x 
        self.ship.bottom = self.position.y

        # create the ship on sprite
        self.ship_fire.center_x = self.position.x 
        self.ship_fire.top = self.position.y

        # keep the ship sprites in a sprite list which is faster later
        self.ship_list = arcade.SpriteList()
        self.ship_list.append(self.ship)
    
    def get_collision_sprite(self):
        return self.ship

    def draw(self):
        self.ship_list.draw()
    
    def on_key_press(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.rocket_acceleration = vector.Vector2D(-math.sin(self.radians), math.cos(self.radians))
            self.rocket_acceleration = vector.Multipy(self.rocket_acceleration, ACCELERATION_ROCKET)
            self.ship_list.append(self.ship_fire)
            self.ship.remove_from_sprite_lists()
            self.ship_engine_on = True
        
        if arcade.key.A == symbol:
            self.radian_speed = ANGULAR_SPEED

        if arcade.key.D == symbol:
            self.radian_speed = -ANGULAR_SPEED

    def on_key_release(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.rocket_acceleration = vector.Vector2D(0, 0)
            self.ship_fire.remove_from_sprite_lists()
            self.ship_list.append(self.ship)
            self.ship_engine_on = False

        if arcade.key.A == symbol:
            self.radian_speed = 0

        if arcade.key.D == symbol:
            self.radian_speed = 0
    
    def on_crash(self):
        # don't draw the good ships any more
        self.ship.remove_from_sprite_lists()
        self.ship_fire.remove_from_sprite_lists()
        #draw the crashed ship
        self.ship_crashed = arcade.Sprite(SHIP_CRASHED_PATH, self.scale)
        self.ship_crashed.center_x = self.position.x
        self.ship_crashed.bottom = self.position.y
        self.ship_list.append(self.ship_crashed)

        self.on_land()

    def on_land(self):
        self.velocity = vector.Vector2D(0, 0)
    
    def bounce(self):
        if self.position.x >= self.width and self.velocity.x > 0:
            self.velocity.x *= -1
        if self.position.x <= 0 and self.velocity.x < 0:
            self.velocity.x *= -1
        if self.position.y >= self.height and self.velocity.y > 0:
            self.velocity.y *= -1
        if self.position.y <= 0 and self.velocity.y < 0:
            self.velocity.y *= -1

    def on_update(self, delta_time: float, force):
        acceleration = vector.Add(force, self.rocket_acceleration)

        self.velocity = vector.Add(self.velocity, vector.Multipy(acceleration, delta_time))
        self.bounce()
        self.position = vector.Add(self.position, vector.Multipy(self.velocity, delta_time))

        self.radians += self.radian_speed

        self.ship.radians = self.radians
        self.ship.center_x = self.position.x
        self.ship.center_y = self.position.y
        self.ship_fire.radians = self.radians
        self.ship_fire.center_x = self.position.x
        self.ship_fire.center_y = self.position.y
        self.ship_list.update()



