import arcade
import os

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
EARTH_HEIGHT = 100
GAME_OVER_FONT_SIZE = 50

SHIP_FILE_WIDTH = 840
SHIP_FILE_HEIGTH = 1510
SHIP_SCALE = 0.1
SHIP_PATH = "../assets/rocket_off.png"
SHIP_FIRE_PATH = "../assets/rocket_engine_fire.png" 
SHIP_CRASHED_PATH = "../assets/rocket_crash.png"

ACCELERATION_GRAVITY = -100
ACCELERATION_ROCKET = 50

SHIP_CRASH_VELOCITY = -200

class MyGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)


    def setup(self):
        # ship state
        self.ship_y = SCREEN_HEIGHT - SHIP_FILE_HEIGTH * SHIP_SCALE
        self.ship_y_velocity = 0
        self.ship_acceleration = ACCELERATION_GRAVITY
        self.ship_engine_on = False

        #game state
        self.paused = False
        self.game_over = False
        self.game_over_message = ""

        # create the ship off sprite
        self.ship = arcade.Sprite(SHIP_PATH, SHIP_SCALE)
        self.ship.center_x = SCREEN_WIDTH / 2
        self.ship.bottom = self.ship_y

        # create the ship on sprite
        self.ship_fire = arcade.Sprite(SHIP_FIRE_PATH, SHIP_SCALE)
        self.ship_fire.center_x = SCREEN_WIDTH / 2
        self.ship_fire.top = self.ship_y

        # keep the ship sprites in a sprite list which is faster later
        self.ship_list = arcade.SpriteList()
        self.ship_list.append(self.ship)


    def on_draw(self):
        arcade.start_render()

        if self.game_over:
            arcade.draw_text(self.game_over_message, SCREEN_WIDTH/2, SCREEN_HEIGHT/2, arcade.color.BLACK, GAME_OVER_FONT_SIZE, align="center", anchor_x="center", anchor_y="center")

        # draw the ground as two blue rectangles with a green in the  middle
        earth_x_cursor = 0
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
        earth_x_cursor += OCEAN_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
        earth_x_cursor += LAND_WIDTH
        arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)

        self.ship_list.draw()
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

        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_ROCKET
            self.ship_list.append(self.ship_fire)
            self.ship_engine_on = True

    def on_key_release(self, symbol, modifiers):
        if arcade.key.W == symbol:
            self.ship_acceleration = ACCELERATION_GRAVITY
            self.ship_fire.remove_from_sprite_lists()
            self.ship_engine_on = False


    def on_update(self, delta_time: float):
        if self.game_over:
            return

        if self.paused:
            return

        # check for on the ground
        if EARTH_HEIGHT >= self.ship_y:
            if self.ship_y_velocity < SHIP_CRASH_VELOCITY:
                self.game_over_message = "GAME OVER\nYou crashed at velocity\n" + str(round(self.ship_y_velocity,1))
                self.game_over = True
                # don't draw the good ships any more
                self.ship.remove_from_sprite_lists()
                self.ship_fire.remove_from_sprite_lists()
                #draw the crashed ship
                self.ship_crashed = arcade.Sprite(SHIP_CRASHED_PATH, SHIP_SCALE)
                self.ship_crashed.center_x = SCREEN_WIDTH / 2
                self.ship_crashed.bottom = self.ship_y
                self.ship_list.append(self.ship_crashed)

            self.ship_y_velocity = 0
            self.ship_y = EARTH_HEIGHT
            if not self.ship_engine_on:
                return

        #move the ship
        self.ship_y_velocity += self.ship_acceleration * delta_time
        self.ship_y += self.ship_y_velocity * delta_time

        self.ship.bottom = self.ship_y
        self.ship_fire.top = self.ship_y
        self.ship_list.update()



    
def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
