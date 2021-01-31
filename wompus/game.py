import arcade
import os
import random

import grid
import player

# Constants
SCREEN_TITLE = "Gravity Game"

GAME_OVER_FONT_SIZE = 50

GRID_COLUMNS = 10
GRID_ROWS = 5

class WompusGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.grid = grid.Grid(width, height, GRID_COLUMNS, GRID_ROWS)
        self.player = player.Player(self.grid.column_width, self.grid.row_height)
        self.playerI = 0
        self.playerJ = 0

    def setup(self):
        #game state
        self.game_over = False
        self.game_over_message = ""
        self.playerI = random.randint(0, GRID_COLUMNS - 1)
        self.playerJ = random.randint(0, GRID_ROWS - 1)
        self.player.SetIJ(self.playerI, self.playerJ)


    def on_draw(self):
        arcade.start_render()

        if self.game_over:
            arcade.draw_text(self.game_over_message, self.width/2, self.height/2, arcade.color.BLACK, GAME_OVER_FONT_SIZE, align="center", anchor_x="center", anchor_y="center")

        self.grid.Draw()
        self.player.Draw()

    def on_key_press(self, symbol, modifiers):
        if arcade.key.ESCAPE == symbol:
            arcade.close_window()

        if arcade.key.N == symbol:
            self.setup()

        # all the controls that are valid during game over need to be above this
        if self.game_over:
            return

    def on_key_release(self, symbol, modifiers):
        if self.game_over:
            return
        if arcade.key.A == symbol:
            self.TryToMoveLeft()
        elif arcade.key.W == symbol:
            self.TryToMoveUp()
        elif arcade.key.D == symbol:
            self.TryToMoveRight()
        elif arcade.key.S == symbol:
            self.TryToMoveDown()
    
    def TryToMoveDown(self):
        if self.playerJ <= 0:
            return
        self.playerJ-=1
        self.player.SetIJ(self.playerI, self.playerJ)

    def TryToMoveUp(self):
        if self.playerJ >= GRID_ROWS - 1:
            return
        self.playerJ+=1
        self.player.SetIJ(self.playerI, self.playerJ)

    def TryToMoveRight(self):
        if self.playerI >= GRID_COLUMNS - 1:
            return
        self.playerI+=1
        self.player.SetIJ(self.playerI, self.playerJ)

    def TryToMoveLeft(self):
        if self.playerI <= 0:
            return
        self.playerI-=1
        self.player.SetIJ(self.playerI, self.playerJ)

    def on_update(self, delta_time: float):
        if self.game_over:
            return

    def on_update(self, delta_time: float):
        if self.game_over:
            return