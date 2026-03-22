import argparse
import math
import sys

import arcade
import torch

from asteroids.ai import (
    HeuristicAIInput,
    RawGeometryNNInputMethod,
    RawGeometryNNParameters,
    validate_and_load_model,
)
from asteroids.ai.polar_nn import PolarNNInputMethod, PolarNNParameters
from asteroids.ai.polar2_nn import Polar2NNInputMethod, Polar2NNParameters
from asteroids.core import Action, Game, InputMethod, KeyboardInput

# constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Asteroids!"


class GameView(arcade.View):
    """Main application class."""

    def __init__(self, game, input_method: InputMethod, pause_on_wave: bool = False):
        super().__init__()
        self.game = game
        self.input_method = input_method
        self.last_action = Action.NO_ACTION
        self.pause_on_wave = pause_on_wave
        self.waiting_for_input = pause_on_wave
        self.last_wave_number = game.wave_number

    def on_update(self, dt):
        """Move everything"""
        if self.waiting_for_input:
            return

        # Detect new wave for crisis pause (human player)
        if self.pause_on_wave and self.game.wave_number != self.last_wave_number:
            self.last_wave_number = self.game.wave_number
            self.waiting_for_input = True
            return

        # Clear turn and acceleration every frame
        self.game.clear_turn()
        self.game.clear_acc()

        # Get and execute the action from input method
        action = self.input_method.get_move()
        self.last_action = action
        bullet_count = len(self.game.bullets)
        self.execute_action(action)
        self.shot_blocked = (
            action == Action.SHOOT and len(self.game.bullets) == bullet_count
        )

        self.game.update(dt)
        if not self.game.player_alive:
            print(f"Your score was {self.game.player_score}")
            print(f"You lived for {self.game.time_alive:.2f} seconds")
            print("Game Over!")
            sys.exit(0)

    def execute_action(self, action: Action):
        """Execute the given action on the game"""
        if action == Action.TURN_LEFT:
            self.game.turning_left()
        elif action == Action.TURN_RIGHT:
            self.game.turning_right()
        elif action == Action.ACCELERATE:
            self.game.accelerate()
        elif action == Action.DECELERATE:
            self.game.decelerate()
        elif action == Action.SHOOT:
            self.game.shoot()
        elif action == Action.NO_ACTION:
            self.game.no_action()

    def on_key_press(self, key, modifiers):
        """Delegate key press to input method if it supports it"""
        if self.waiting_for_input:
            self.waiting_for_input = False
        if hasattr(self.input_method, "on_key_press"):
            self.input_method.on_key_press(key, modifiers)

    def on_key_release(self, key, modifiers):
        """Delegate key release to input method if it supports it"""
        if hasattr(self.input_method, "on_key_release"):
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

        theta = 0.8 * math.pi + p_angle
        x2 = cx + r * math.cos(theta)
        y2 = cy + r * math.sin(theta)

        theta = 1.2 * math.pi + p_angle
        x3 = cx + r * math.cos(theta)
        y3 = cy + r * math.sin(theta)

        # Draw the filled triangle using the calculated points
        arcade.draw_triangle_filled(x1, y1, x2, y2, x3, y3, arcade.color.WHITE)

        # Thrust/brake indicator
        if self.last_action == Action.ACCELERATE:
            # Flame behind the ship (opposite of heading)
            back_angle = p_angle + math.pi
            flame_len = r * 1.5
            ft = back_angle  # flame tip
            fl = back_angle + 0.3  # flame left
            fr = back_angle - 0.3  # flame right
            base_dist = r * 0.6
            arcade.draw_triangle_filled(
                cx + flame_len * math.cos(ft),
                cy + flame_len * math.sin(ft),
                cx + base_dist * math.cos(fl),
                cy + base_dist * math.sin(fl),
                cx + base_dist * math.cos(fr),
                cy + base_dist * math.sin(fr),
                arcade.color.ORANGE,
            )
        elif self.last_action == Action.DECELERATE:
            # Flame in front of the ship (direction of heading)
            flame_len = r * 1.5
            ft = p_angle
            fl = p_angle + 0.3
            fr = p_angle - 0.3
            base_dist = r * 0.6
            arcade.draw_triangle_filled(
                cx + flame_len * math.cos(ft),
                cy + flame_len * math.sin(ft),
                cx + base_dist * math.cos(fl),
                cy + base_dist * math.sin(fl),
                cx + base_dist * math.cos(fr),
                cy + base_dist * math.sin(fr),
                arcade.color.RED,
            )
        elif self.last_action == Action.TURN_LEFT:
            # Triangle on the right side of the ship (reaction flame)
            side_angle = p_angle - math.pi / 2
            flame_len = r * 1.5
            ft = side_angle
            fl = side_angle + 0.3
            fr = side_angle - 0.3
            base_dist = r * 0.6
            arcade.draw_triangle_filled(
                cx + flame_len * math.cos(ft),
                cy + flame_len * math.sin(ft),
                cx + base_dist * math.cos(fl),
                cy + base_dist * math.sin(fl),
                cx + base_dist * math.cos(fr),
                cy + base_dist * math.sin(fr),
                arcade.color.YELLOW,
            )
        elif self.last_action == Action.TURN_RIGHT:
            # Triangle on the left side of the ship (reaction flame)
            side_angle = p_angle + math.pi / 2
            flame_len = r * 1.5
            ft = side_angle
            fl = side_angle + 0.3
            fr = side_angle - 0.3
            base_dist = r * 0.6
            arcade.draw_triangle_filled(
                cx + flame_len * math.cos(ft),
                cy + flame_len * math.sin(ft),
                cx + base_dist * math.cos(fl),
                cy + base_dist * math.sin(fl),
                cx + base_dist * math.cos(fr),
                cy + base_dist * math.sin(fr),
                arcade.color.YELLOW,
            )
        elif self.last_action == Action.SHOOT and self.shot_blocked:
            # Shoot on cooldown — small X in front of ship
            front_dist = r * 1.3
            fx = cx + front_dist * math.cos(p_angle)
            fy = cy + front_dist * math.sin(p_angle)
            s = r * 0.25
            arcade.draw_line(fx - s, fy - s, fx + s, fy + s, arcade.color.RED, 2)
            arcade.draw_line(fx - s, fy + s, fx + s, fy - s, arcade.color.RED, 2)
        elif self.last_action == Action.NO_ACTION:
            arcade.draw_circle_filled(cx, cy, r * 0.15, arcade.color.GRAY)

    def draw_bullets(self, bullets):
        self.draw_circles(bullets)

    def draw_asteroids(self, asteroids):
        self.draw_circles(asteroids)

    def draw_circles(self, geometries):
        for g in geometries:
            arcade.draw_circle_filled(g.pos.x, g.pos.y, g.radius, arcade.color.WHITE)

    def on_draw(self):
        self.clear()
        geometry = self.game.geometry_state()
        self.draw_player(geometry.player)
        self.draw_asteroids(geometry.asteroids)
        self.draw_bullets(geometry.bullets)

        # HUD: wave and score
        arcade.draw_text(
            f"Wave: {self.game.wave_number}  Score: {self.game.player_score}",
            WINDOW_WIDTH - 10,
            WINDOW_HEIGHT - 30,
            arcade.color.WHITE,
            14,
            anchor_x="right",
        )

        if self.waiting_for_input:
            arcade.draw_text(
                "Press any key to start",
                WINDOW_WIDTH // 2,
                100,
                arcade.color.YELLOW,
                20,
                anchor_x="center",
            )


def main():
    parser = argparse.ArgumentParser(description="Asteroids Game")
    parser.add_argument(
        "--aih", action="store_true", help="Use Heuristic AI input instead of keyboard"
    )
    parser.add_argument(
        "--air",
        type=str,
        nargs="?",
        const="nn_model.pth",
        help="Use RawGeometry Neural Network AI, optionally specify model file path (default: nn_model.pth)",
    )
    parser.add_argument(
        "--aip",
        type=str,
        nargs="?",
        const="polar_pg_best.pth",
        help="Use Polar Neural Network AI, optionally specify model file path (default: polar_pg_best.pth)",
    )
    parser.add_argument(
        "--aip2",
        type=str,
        nargs="?",
        const="nn_polar2.pth",
        help="Use Polar2 Neural Network AI, optionally specify model file path (default: nn_polar2.pth)",
    )
    parser.add_argument(
        "--crisis",
        action="store_true",
        help="Enable crisis mode: small asteroid dodge scenarios",
    )
    args = parser.parse_args()

    g = Game(WINDOW_WIDTH, WINDOW_HEIGHT, crisis_mode=args.crisis)

    if args.aih:
        input_method = HeuristicAIInput(g)
    elif args.air or args.aip or args.aip2:
        # Detect device (GPU if available, else CPU)
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        if args.aip2:
            model_file = args.aip2
            params = Polar2NNParameters(device=device)
            input_class = Polar2NNInputMethod
        elif args.aip:
            model_file = args.aip
            params = PolarNNParameters(device=device)
            input_class = PolarNNInputMethod
        else:
            model_file = args.air
            params = RawGeometryNNParameters(device=device)
            input_class = RawGeometryNNInputMethod
        validate_and_load_model(
            params.model,
            torch.load(
                "nn_weights/" + model_file, map_location=device, weights_only=False
            ),
            source_description=model_file,
        )
        params.model.eval()
        print(f"Model loaded from {model_file} on device: {device}")
        input_method = input_class(g, parameters=params)
    else:
        input_method = KeyboardInput()

    is_human = not (args.aih or args.air or args.aip or args.aip2)
    pause_on_wave = args.crisis and is_human

    window = arcade.Window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    game_view = GameView(g, input_method, pause_on_wave=pause_on_wave)
    window.show_view(game_view)
    arcade.run()


if __name__ == "__main__":
    main()
