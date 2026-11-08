import math
from dataclasses import dataclass
from typing import NamedTuple

import arcade

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


@dataclass
class WaveParam:
    frequency: float
    amplitude: float
    phase: float


class Point(NamedTuple):
    x: float
    y: float


class GameView(arcade.View):
    def __init__(self) -> None:
        super().__init__()
        # range is centered on the origin. So range of 4pi is -2pi--2pi
        self.x_reange = math.pi * 4
        self.y_range = 4
        # sample spacing is 2pi/sampe_rate
        self.sample_rate = 100

        self.wave = WaveParam(1, 1, 0)

    def graph_to_game(self, p: Point) -> Point:
        result = Point(
            p.x / self.x_reange * WINDOW_WIDTH + WINDOW_WIDTH,
            p.y / self.y_range * WINDOW_HEIGHT + WINDOW_HEIGHT,
        )
        print(result.x, result.y)
        return result

    def draw_some_test_points(self) -> None:
        point_list = (
            self.graph_to_game(Point(-math.pi, -1)),
            self.graph_to_game(Point(-math.pi, 1)),
        )
        arcade.draw_lines(point_list, arcade.color.BLUE, 3)

    def draw_axies(self) -> None:
        point_list = (
            (0, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH / 2, 0),
            (WINDOW_WIDTH / 2, WINDOW_HEIGHT),
        )
        arcade.draw_lines(point_list, arcade.color.BLUE, 3)

    def draw_sample_strip(self) -> None:
        point_list = (
            (510, 450),
            (570, 450),
            (510, 480),
            (570, 480),
            (510, 510),
            (570, 510),
        )
        arcade.draw_line_strip(point_list, arcade.color.TROPICAL_RAIN_FOREST, 3)

    def on_draw(self) -> bool | None:
        self.draw_axies()
        self.draw_sample_strip()
        self.draw_some_test_points()

    def on_update(self, delta_time: float) -> bool | None:
        pass


def main():
    window = arcade.Window(
        WINDOW_WIDTH, WINDOW_HEIGHT, "Show Me A Sound", resizable=True
    )
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
