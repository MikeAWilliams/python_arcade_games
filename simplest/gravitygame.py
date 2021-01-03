import arcade
import os

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
EARTH_HEIGHT = 100

SHIP_FILE_WIDTH = 840
SHIP_FILE_HEIGTH = 1510
SHIP_SCALE = 0.1
SHIP_OFF_PATH = "../assets/rocket_off.png"
SHIP_ON_PATH = "../assets/rocket_on.png"

ACCELERATION_GRAVITY = -5
ACCELERATION_ROCKET = 3

class MyGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)


    def setup(self):
        # create the ship one for engine off one onn
        self.ship_off = arcade.Sprite(SHIP_OFF_PATH, SHIP_SCALE)
        self.ship_off.center_x = SCREEN_WIDTH / 2
        self.ship_off.bottom = SCREEN_HEIGHT - SHIP_FILE_HEIGTH * SHIP_SCALE
        self.ship_off.alpha = 255

        self.ship_on = arcade.Sprite(SHIP_ON_PATH, SHIP_SCALE)
        self.ship_on.center_x = SCREEN_WIDTH / 2
        self.ship_on.bottom = SCREEN_HEIGHT - SHIP_FILE_HEIGTH * SHIP_SCALE
        self.ship_on.alpha = 0

        self.ship_list = arcade.SpriteList()
        self.ship_list.append(self.ship_on)
        self.ship_list.append(self.ship_off)

        self.ship_y_velocity = 0
        self.ship_acceleration = ACCELERATION_GRAVITY

        self.paused = False

    def on_draw(self):
        arcade.start_render()

        # draw the ground as two blue rectangles with a green in the  middle
        earth_x_cursor = 0
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
        earth_x_cursor += OCEAN_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
        earth_x_cursor += LAND_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)

        self.ship_list.draw()

    def on_key_press(self, symbol, modifiers):
        if arcade.key.ESCAPE == symbol:
            arcade.close_window()
        
        if arcade.key.ENTER == symbol:
            self.paused = not self.paused
        
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_ROCKET
            self.ship_on.alpha = 255
            self.ship_off.alpha = 0

    def on_key_release(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_GRAVITY
            self.ship_on.alpha = 0
            self.ship_off.alpha = 255


    def on_update(self, delta_time: float):
        if self.paused:
            return
        
        #move the ship
        self.ship_y_velocity += self.ship_acceleration * delta_time

        self.ship_on.change_y = self.ship_y_velocity
        self.ship_off.change_y = self.ship_y_velocity
        self.ship_list.update()



    
def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
