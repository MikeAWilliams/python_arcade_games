import math
from dataclasses import dataclass
from typing import NamedTuple

import arcade

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


@dataclass
class WaveParam:
    frequency: float
    amplitude: float
    phase: float
    color: arcade.types.Color

    def value_at_t(self, t) -> float:
        return self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phase)


class Point(NamedTuple):
    x: float
    y: float


class GameView(arcade.View):
    def __init__(self) -> None:
        super().__init__()
        # range is centered on the origin. So range of 4 is -2 - 2
        self.x_range = 0.005
        self.y_range = 4
        # sample spacing is number of samples in the full x range
        self.sample_rate = 1000

        self.waves = [
            WaveParam(4000, 1, 0, arcade.color.GREEN),
            WaveParam(20, 1, 0, arcade.color.RED),
        ]

    def graph_to_game(self, p: Point) -> Point:
        result = Point(
            p.x / self.x_range * WINDOW_WIDTH + WINDOW_WIDTH / 2,
            p.y / self.y_range * WINDOW_HEIGHT + WINDOW_HEIGHT / 2,
        )
        return result

    def draw_axies(self) -> None:
        point_list = (
            (0, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH / 2, 0),
            (WINDOW_WIDTH / 2, WINDOW_HEIGHT),
        )
        arcade.draw_lines(point_list, arcade.color.BLUE, 3)

    def draw_wave(self) -> None:
        dx = self.x_range / self.sample_rate
        point_lists = [[] for _ in self.waves]
        for i in range(self.sample_rate + 1):
            x = -self.x_range / 2 + i * dx
            for index, w in enumerate(self.waves):
                p = Point(x, w.value_at_t(x))
                point_lists[index].append(self.graph_to_game(p))
        for i in range(len(self.waves)):
            arcade.draw_line_strip(point_lists[i], self.waves[i].color)

    def on_draw(self) -> bool | None:
        self.draw_axies()
        self.draw_wave()

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
