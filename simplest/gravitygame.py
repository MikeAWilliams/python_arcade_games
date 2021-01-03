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
SHIP_PATH = "../assets/rocket_off.png"

ACCELERATION_GRAVITY = -5
ACCELERATION_ROCKET = 3

class MyGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)

        self.paused = False
        self.ship_acceleration = ACCELERATION_GRAVITY

    def setup(self):
        # create the ship
        self.ship = arcade.Sprite(SHIP_PATH, SHIP_SCALE)
        self.ship.center_x = SCREEN_WIDTH / 2
        self.ship.bottom = SCREEN_HEIGHT - SHIP_FILE_HEIGTH * SHIP_SCALE

    def on_draw(self):
        arcade.start_render()

        # draw the ground as two blue rectangles with a green in the  middle
        earth_x_cursor = 0
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
        earth_x_cursor += OCEAN_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
        earth_x_cursor += LAND_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)

        self.ship.draw()

    def on_key_press(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_ROCKET

    def on_update(self, delta_time: float):
        if self.paused:
            return
        
        #move the ship
        self.ship.change_y += self.ship_acceleration * delta_time
        self.ship.update()


    
def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
