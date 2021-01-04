import arcade

import vector

SHIP_FILE_WIDTH = 840
SHIP_FILE_HEIGTH = 1510
SHIP_PATH = "../assets/rocket_off.png"
SHIP_FIRE_PATH = "../assets/rocket_engine_fire.png" 
SHIP_CRASHED_PATH = "../assets/rocket_crash.png"

ACCELERATION_GRAVITY = -100
ACCELERATION_ROCKET = 50
SHIP_CRASH_VELOCITY = -200

class Ship():
    def __init__(self, init_x, init_y):
        self.scale = 0.1
        self.init_position = vector.Vector2D(init_x, init_y)
        self.position = self.init_position

        self.ship = arcade.Sprite(SHIP_PATH, self.scale)
        self.ship_fire = arcade.Sprite(SHIP_FIRE_PATH, self.scale)
    
    def setup(self):
        # ship state
        self.position = self.init_position.copy()
        self.velocity = 0
        self.acceleration = ACCELERATION_GRAVITY
        self.ship_engine_on = False

        # create the ship off sprite
        self.ship.center_x = self.position.x 
        self.ship.bottom = self.position.y

        # create the ship on sprite
        self.ship_fire.center_x = self.position.x 
        self.ship_fire.top = self.position.y

        # keep the ship sprites in a sprite list which is faster later
        self.ship_list = arcade.SpriteList()
        self.ship_list.append(self.ship)

    def draw(self):
        self.ship_list.draw()
    
    def on_key_press(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.acceleration = ACCELERATION_ROCKET
            self.ship_list.append(self.ship_fire)
            self.ship_engine_on = True

    def on_key_release(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.acceleration = ACCELERATION_GRAVITY
            self.ship_fire.remove_from_sprite_lists()
            self.ship_engine_on = False
    
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
        self.velocity = 0

    def on_update(self, delta_time: float):
        # check for on the ground
        #move the ship
        self.velocity += self.acceleration * delta_time
        self.position.y += self.velocity * delta_time

        self.ship.bottom = self.position.y
        self.ship_fire.top = self.position.y
        self.ship_list.update()



