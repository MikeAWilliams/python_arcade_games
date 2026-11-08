import math
from dataclasses import dataclass
from typing import NamedTuple

import arcade
import numpy as np
import sounddevice as sd

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160


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
        self.x_range = 0.025
        self.y_range = 4
        # sample spacing is number of samples in the full x range
        self.num_samples = 1000

        self.sum_color = arcade.color.RED

        self.audio_sample_rate = 44100
        self.audio_volume = 0.4
        self.audio_time = 1

        self.waves = [
            # WaveParam(4000, 1, 0, arcade.color.GREEN),
            WaveParam(400, 1, 0, arcade.color.GREEN),
            WaveParam(400, 1, math.pi / 2, arcade.color.GREEN),
        ]
        self.generate_data()
        sd.play(self.audio_samples, self.audio_sample_rate, loop=True)

    def generate_data(self) -> None:
        self.generate_axes()
        self.generate_wave_points()
        self.generate_audio_sample()

    def generate_axes(self) -> None:
        self.axes_points = (
            (0, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH, WINDOW_HEIGHT / 2),
            (WINDOW_WIDTH / 2, 0),
            (WINDOW_WIDTH / 2, WINDOW_HEIGHT),
        )

    def generate_wave_points(self) -> None:
        dx = self.x_range / self.num_samples
        self.wave_point_lists = [[] for _ in self.waves]
        self.wave_sum_list = []
        for i in range(self.num_samples + 1):
            x = -self.x_range / 2 + i * dx
            sum = 0
            for index, w in enumerate(self.waves):
                p = Point(x, w.value_at_t(x))
                sum += p.y
                self.wave_point_lists[index].append(self.graph_to_game(p))
            # sum /= len(self.waves)
            self.wave_sum_list.append(self.graph_to_game(Point(x, sum)))

    def generate_audio_sample(self) -> None:
        t = (
            np.arange(int(self.audio_sample_rate * self.audio_time))
            / self.audio_sample_rate
        )

        total = np.zeros_like(t)
        for w in self.waves:
            total += w.amplitude * np.sin(2 * np.pi * w.frequency * t + w.phase)
        peak = np.max(np.abs(total))
        if peak > 0:
            total /= peak  # normalise so the sum never clips
        self.audio_samples = (total * self.audio_volume).astype(np.float32)

    def graph_to_game(self, p: Point) -> Point:
        result = Point(
            p.x / self.x_range * WINDOW_WIDTH + WINDOW_WIDTH / 2,
            p.y / self.y_range * WINDOW_HEIGHT + WINDOW_HEIGHT / 2,
        )
        return result

    def draw_axies(self) -> None:
        arcade.draw_lines(self.axes_points, arcade.color.BLUE, 3)

    def draw_wave(self) -> None:
        for i in range(len(self.waves)):
            arcade.draw_line_strip(self.wave_point_lists[i], self.waves[i].color)
        arcade.draw_line_strip(self.wave_sum_list, self.sum_color)

    def on_draw(self) -> bool | None:
        self.clear()
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
