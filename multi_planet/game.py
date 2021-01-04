import arcade
import os

import vector
import ship

# Constants
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
EARTH_HEIGHT = 100
GAME_OVER_FONT_SIZE = 50




class GravityGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.ship = ship.Ship(width/2, height)


    def setup(self):
        #game state
        self.paused = False
        self.game_over = False
        self.game_over_message = ""

        self.ship.setup()


    def on_draw(self):
        arcade.start_render()

        if self.game_over:
            arcade.draw_text(self.game_over_message, self.width/2, self.height/2, arcade.color.BLACK, GAME_OVER_FONT_SIZE, align="center", anchor_x="center", anchor_y="center")

        # draw the ground as two blue rectangles with a green in the  middle
        earth_x_cursor = 0
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
        earth_x_cursor += OCEAN_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
        earth_x_cursor += LAND_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)

        self.ship.draw()

        arcade.finish_render()

    def on_key_press(self, symbol, modifiers):
        if arcade.key.ESCAPE == symbol:
            arcade.close_window()

        if arcade.key.N == symbol:
            self.setup()

        # all the controls that are valid during game over need to be above this
        if self.game_over:
            return

        if arcade.key.ENTER == symbol:
            self.paused = not self.paused

        self.ship.on_key_press(symbol, modifiers)
    
    def on_key_release(self, symbol, modifiers):
        self.ship.on_key_release(symbol, modifiers)

    def detect_colisions(self):
        if EARTH_HEIGHT >= self.ship.position.y:
            if self.ship.velocity.y < ship.SHIP_CRASH_VELOCITY:
                self.game_over_message = "GAME OVER\nYou crashed at velocity\n" + str(round(self.ship.velocity.y,1))
                self.ship.on_crash()
                self.game_over = True
            else:
                self.ship.on_land()
            return True
        return False


    def on_update(self, delta_time: float):
        if self.game_over:
            return

        if self.paused:
            return

        self.detect_colisions()

        if not self.game_over:
            self.ship.on_update(delta_time)



    