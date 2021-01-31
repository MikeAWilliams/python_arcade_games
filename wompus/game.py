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

class Position():
    def __init__(self, i, j):
        self.I = i
        self.J = j
    def Intersecrts(self, other):
        return self.I == other.I and self.J == other.J

def GetRandomUnocupiedPosition(ocupied):
    while True:
        position = Position(random.randint(0, GRID_COLUMNS - 1), random.randint(0, GRID_ROWS - 1))
        good = True
        for other in ocupied:
            if position.Intersecrts(other):
                good = False
                break
        if good:
            break
    return position


class WompusGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.grid = grid.Grid(width, height, GRID_COLUMNS, GRID_ROWS)
        self.player = player.Player(self.grid.column_width, self.grid.row_height)
        self.playerPosition = Position(0, 0)

    def setup(self):
        #game state
        self.game_over = False
        self.game_over_message = ""
        self.playerPosition = GetRandomUnocupiedPosition([])
        self.player.SetIJ(self.playerPosition.I, self.playerPosition.J)
        self.wompusPosition = GetRandomUnocupiedPosition([self.playerPosition])
        self.exitPosition = GetRandomUnocupiedPosition([self.playerPosition, self.wompusPosition])


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
        self.CheckEndGame()
    
    def CheckEndGame(self):
        if self.playerPosition.Intersecrts(self.wompusPosition):
            self.game_over = True
            self.game_over_message = "You were killed by the wompus"

        if self.playerPosition.Intersecrts(self.exitPosition):
            self.game_over = True
            self.game_over_message = "You escaped"

    def TryToMoveDown(self):
        if self.playerPosition.J <= 0:
            return
        self.playerPosition.J-=1
        self.player.SetIJ(self.playerPosition.I, self.playerPosition.J)

    def TryToMoveUp(self):
        if self.playerPosition.J >= GRID_ROWS - 1:
            return
        self.playerPosition.J+=1
        self.player.SetIJ(self.playerPosition.I, self.playerPosition.J)

    def TryToMoveRight(self):
        if self.playerPosition.I >= GRID_COLUMNS - 1:
            return
        self.playerPosition.I+=1
        self.player.SetIJ(self.playerPosition.I, self.playerPosition.J)

    def TryToMoveLeft(self):
        if self.playerPosition.I <= 0:
            return
        self.playerPosition.I-=1
        self.player.SetIJ(self.playerPosition.I, self.playerPosition.J)

    def on_update(self, delta_time: float):
        if self.game_over:
            return

    def on_update(self, delta_time: float):
        if self.game_over:
            return