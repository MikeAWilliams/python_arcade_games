import arcade
import os

import vector
import ship

# Constants
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
ACCELERATION_GRAVITY = 1000000
GAME_OVER_FONT_SIZE = 50




class GravityGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.ship = ship.Ship(width/2, height)
        self.asteroids = arcade.SpriteList()
        
        self.asteroid_centers = []
        self.asteroid_centers.append(vector.Vector2D(self.width / 4, self.height / 2))
        self.asteroid_centers.append(vector.Vector2D(self.width - self.width / 4, self.height / 2))

        for center in self.asteroid_centers:
            asteroid = arcade.Sprite("../assets/asteroid.png", 0.2)
            asteroid.center_x = center.x
            asteroid.center_y = center.y
            self.asteroids.append(asteroid)



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

    def compute_asteriod_ship_gravity(self):
        result = vector.Vector2D(0, 0)
        for center in self.asteroid_centers:
            direction = vector.Add(result, vector.Subtract(center, self.ship.position))
            length = direction.length()
            # prevent division by small numebers from blowing stuff up
            if length > 50:
                direction.make_unit()
            else:
                length = 50
                direction = vector.Multipy(direction, 1.0 / length)

            result = vector.Add(result, vector.Multipy(direction, ACCELERATION_GRAVITY / (length * length)))

        return result

    def on_update(self, delta_time: float):
        if self.game_over:
            return

        if self.paused:
            return

        self.detect_colisions()

        if not self.game_over:
            self.ship.on_update(delta_time, self.compute_asteriod_ship_gravity())



    