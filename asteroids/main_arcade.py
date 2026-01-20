import arcade
import game
import math
import sys
from game_input_keyboard import KeyboardInput

# constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Asteroids!"


class GameView(arcade.View):
    """ Main application class. """

    def __init__(self, game, input_method: game.InputMethod):
        super().__init__()
        self.game = game
        self.input_method = input_method

    def on_update(self, dt):
        """ Move everything """
        # Clear turn and acceleration every frame
        self.game.clear_turn()
        self.game.clear_acc()

        # Get and execute the action from input method
        action = self.input_method.get_move()
        self.execute_action(action)

        self.game.update(dt)
        if not self.game.player_alive:
            print(f"Your score was {self.game.player_score}")
            print(f"You lived for {self.game.time_alive:.2f} seconds")
            print("Game Over!")
            sys.exit(0)
        if len(self.game.asteroids) == 0:
            print(f"Your score was {self.game.player_score}")
            print(f"You lived for {self.game.time_alive:.2f} seconds")
            print("You Win!")
            sys.exit(0)

    def execute_action(self, action: game.Action):
        """Execute the given action on the game"""
        if action == game.Action.TURN_LEFT:
            self.game.turning_left()
        elif action == game.Action.TURN_RIGHT:
            self.game.turning_right()
        elif action == game.Action.ACCELERATE:
            self.game.accelerate()
        elif action == game.Action.DECELERATE:
            self.game.decelerate()
        elif action == game.Action.SHOOT:
            self.game.shoot()
        elif action == game.Action.NO_ACTION:
            self.game.no_action()

    def on_key_press(self, key, modifiers):
        """Delegate key press to input method if it supports it"""
        if hasattr(self.input_method, 'on_key_press'):
            self.input_method.on_key_press(key, modifiers)

    def on_key_release(self, key, modifiers):
        """Delegate key release to input method if it supports it"""
        if hasattr(self.input_method, 'on_key_release'):
            self.input_method.on_key_release(key, modifiers)

    def draw_player(self, player_geometry):
        cx = player_geometry.pos.x
        cy = player_geometry.pos.y
        r = player_geometry.radius
        p_angle = player_geometry.angle

        # outline for collision detection
        arcade.draw_circle_outline(cx, cy, r, arcade.color.WHITE, border_width=1)

        # the triangle we expect

        theta = p_angle
        x1 = cx + r * math.cos(theta)
        y1 = cy + r * math.sin(theta)

        theta = 0.8*math.pi+p_angle
        x2 = cx + r * math.cos(theta)
        y2 = cy + r * math.sin(theta)

        theta = 1.2*math.pi+p_angle
        x3 = cx + r * math.cos(theta)
        y3 = cy + r * math.sin(theta)

        # Draw the filled triangle using the calculated points
        arcade.draw_triangle_filled(
            x1, y1, x2, y2, x3, y3, arcade.color.WHITE
        )

    def draw_bullets(self, bullets):
        self.draw_circles(bullets)

    def draw_asteroids(self, asteroids):
        self.draw_circles(asteroids)

    def draw_circles(self, geometries):
        for g in geometries:
            arcade.draw_circle_filled(g.pos.x, g.pos.y, g.radius, arcade.color.WHITE)

    def draw_geometry(self, geometry: game.geometry_state):
        self.draw_player(geometry.player)
        self.draw_asteroids(geometry.asteroids)
        self.draw_bullets(geometry.bullets)

    def on_draw(self):
        """ Render the screen. """
        self.clear()
        self.draw_geometry(self.game.geometry_state())

def main():
    g = game.Game(WINDOW_WIDTH, WINDOW_HEIGHT)
    input_method = KeyboardInput()

    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game_view = GameView(g, input_method)
    window.show_view(game_view)
    arcade.run()

if __name__ == "__main__":
    main()
