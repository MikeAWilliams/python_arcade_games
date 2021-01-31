import arcade
import os
import random

import grid

# Constants
SCREEN_TITLE = "Gravity Game"

GAME_OVER_FONT_SIZE = 50




class WompusGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.grid = grid.Grid(self.width, self.height)

    def setup(self):
        #game state
        self.game_over = False
        self.game_over_message = ""


    def on_draw(self):
        arcade.start_render()

        if self.game_over:
            arcade.draw_text(self.game_over_message, self.width/2, self.height/2, arcade.color.BLACK, GAME_OVER_FONT_SIZE, align="center", anchor_x="center", anchor_y="center")

        self.grid.Draw()

    def on_key_press(self, symbol, modifiers):
        if arcade.key.ESCAPE == symbol:
            arcade.close_window()

        if arcade.key.N == symbol:
            self.setup()

        # all the controls that are valid during game over need to be above this
        if self.game_over:
            return

    def on_key_release(self, symbol, modifiers):
        pass

    def on_update(self, delta_time: float):
        if self.game_over:
            return