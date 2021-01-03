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

class MyGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)

    def setup(self):
        pass

    def on_draw(self):
        arcade.start_render()

        # draw the ground as two blue rectangles with a green in the  middle
        earth_x_cursor = 0
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
        earth_x_cursor += OCEAN_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
        earth_x_cursor += LAND_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)

        ship_list = arcade.SpriteList()
        ship = arcade.Sprite(SHIP_PATH, SHIP_SCALE)
        ship.center_x = SCREEN_WIDTH / 2
        ship.bottom = SCREEN_HEIGHT - SHIP_FILE_HEIGTH * SHIP_SCALE
        ship_list.append(ship)
        ship_list.draw()

        ship.change_y = -100

def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
