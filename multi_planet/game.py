import arcade
import os

import vector
import ship
import asteroid_field

# Constants
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
GAME_OVER_FONT_SIZE = 50
NUMBER_OF_ASTEROIDS = 6




class GravityGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.ship = ship.Ship(width/2, height)
        self.asteroids = asteroid_field.AsteroidField(self.width, self.height, NUMBER_OF_ASTEROIDS)
        

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

        self.asteroids.draw()
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
        return False


    def on_update(self, delta_time: float):
        if self.game_over:
            return

        if self.paused:
            return

        self.detect_colisions()

        if not self.game_over:
            self.ship.on_update(delta_time, self.asteroids.compute_asteriod_to_point_gravity(self.ship.position))



    