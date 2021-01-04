import arcade

SHIP_FILE_WIDTH = 840
SHIP_FILE_HEIGTH = 1510
SHIP_PATH = "../assets/rocket_off.png"
SHIP_FIRE_PATH = "../assets/rocket_engine_fire.png" 
SHIP_CRASHED_PATH = "../assets/rocket_crash.png"

ACCELERATION_GRAVITY = -100
ACCELERATION_ROCKET = 50
SHIP_CRASH_VELOCITY = -200

class Ship():
    def __init__(self, center_x, top_y, bottom_y):
        self.scale = 0.1
        self.center_x = center_x
        self.top_y = top_y
        self.bottom_y = bottom_y
    
    def setup(self):
        # ship state
        self.ship_y = self.top_y - SHIP_FILE_HEIGTH * self.scale
        self.ship_y_velocity = 0
        self.ship_acceleration = ACCELERATION_GRAVITY
        self.ship_engine_on = False

        # create the ship off sprite
        self.ship = arcade.Sprite(SHIP_PATH, self.scale)
        self.ship.center_x = self.center_x 
        self.ship.bottom = self.ship_y

        # create the ship on sprite
        self.ship_fire = arcade.Sprite(SHIP_FIRE_PATH, self.scale)
        self.ship_fire.center_x = self.center_x 
        self.ship_fire.top = self.ship_y

        # keep the ship sprites in a sprite list which is faster later
        self.ship_list = arcade.SpriteList()
        self.ship_list.append(self.ship)

    def draw(self):
        self.ship_list.draw()
    
    def on_key_press(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_ROCKET
            self.ship_list.append(self.ship_fire)
            self.ship_engine_on = True

    def on_key_release(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_GRAVITY
            self.ship_fire.remove_from_sprite_lists()
            self.ship_engine_on = False
    
    def on_crash(self):
        # don't draw the good ships any more
        self.ship.remove_from_sprite_lists()
        self.ship_fire.remove_from_sprite_lists()
        #draw the crashed ship
        self.ship_crashed = arcade.Sprite(SHIP_CRASHED_PATH, self.scale)
        self.ship_crashed.center_x = self.center_x
        self.ship_crashed.bottom = self.ship_y
        self.ship_list.append(self.ship_crashed)

        self.on_land()

    def on_land(self):
        self.ship_y_velocity = 0
        self.ship_y = self.bottom_y

    def on_update(self, delta_time: float):
        # check for on the ground
        if self.bottom_y >= self.ship_y:
            if self.ship_y_velocity < SHIP_CRASH_VELOCITY:
                self.game_over_message = "GAME OVER\nYou crashed at velocity\n" + str(round(self.ship_y_velocity,1))
                self.game_over = True

        #move the ship
        self.ship_y_velocity += self.ship_acceleration * delta_time
        self.ship_y += self.ship_y_velocity * delta_time

        self.ship.bottom = self.ship_y
        self.ship_fire.top = self.ship_y
        self.ship_list.update()



