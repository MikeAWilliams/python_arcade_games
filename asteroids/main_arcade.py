import arcade
import game
import math
import sys

# constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Asteroids!"

class GameView(arcade.View):
    """ Main application class. """

    def __init__(self, game):
        # Call the parent __init__
        super().__init__()
        self.game = game


    def on_update(self, dt):
        """ Move everything """
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

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.game.player.turning_left()
        elif key == arcade.key.RIGHT:
            self.game.player.turning_right()
        elif key == arcade.key.SPACE:
            self.game.player.shoot()
        elif key == arcade.key.UP:
            self.game.player.accelerate()
        elif key == arcade.key.DOWN:
            self.game.player.decelerate()

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.game.player.clear_turn()
        elif key == arcade.key.UP or key == arcade.key.DOWN:
            self.game.player.clear_acc()

    def draw_player(self):
        cx = self.game.player.pos.x
        cy = self.game.player.pos.y
        r = self.game.player.radius
        p_angle = self.game.player.angle

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

        # Draw bullets
        for bullet in self.game.player.bullets:
            arcade.draw_circle_filled(bullet.pos.x, bullet.pos.y, bullet.radius, arcade.color.WHITE)

    def on_draw(self):
        """ Render the screen. """
        self.clear()
        self.draw_player()
        for a in self.game.asteroids:
            arcade.draw_circle_filled(a.pos.x, a.pos.y, a.radius, arcade.color.WHITE)


def main():
    g = game.Game(WINDOW_WIDTH, WINDOW_HEIGHT)

    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game_view = GameView(g)
    window.show_view(game_view)
    arcade.run()
if __name__ == "__main__":
    main()
