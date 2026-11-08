import colorsys
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

        self.setup_simple_waves()
        # self.setup_piano_waves()

        self.generate_data()
        sd.play(self.audio_samples, self.audio_sample_rate, loop=True)

    def setup_simple_waves(self) -> None:
        self.waves = [
            # WaveParam(4000, 1, 0, arcade.color.GREEN),
            WaveParam(261, 1, 0, arcade.color.GREEN),
            # WaveParam(261, 1, math.pi / 2, arcade.color.GREEN),
        ]

    def setup_piano_waves(
        self,
        fundamental: float = 261.63,  # middle C (about 262 Hz)
        rolloff: float = 1.0,
        max_frequency: float = 8000,
    ) -> None:
        """Build a static approximation of a piano tone out of sine waves.

        A piano note is a stack of partials (sine waves) at whole-number
        multiples of the fundamental, getting quieter as they go up.

        fundamental    pitch of the note in Hz.
        rolloff        partial n has amplitude 1 / n**rolloff. Higher = duller
                       (like a soft felt hammer); lower = brighter/buzzier.
                       Keep it modest for low notes: small speakers can't play
                       the fundamental, and your ear infers the pitch from the
                       upper partials, which must be strong enough to do that.
        max_frequency  no partial above this many Hz is added. Low notes get
                       many partials, high notes only one or two. It also keeps
                       partials below the Nyquist limit (half the sample rate),
                       where they would alias into the wrong pitch.

        Each frequency is rounded to a whole number of Hz. The audio buffer
        loops and is a whole number of seconds long, so whole-Hz partials finish
        a complete cycle and loop without a click. (The pitch error is largest
        on the lowest notes: 27.5 Hz becomes 28 Hz.)

        Not modelled: the attack and decay of a real key press. This is a
        steady sustained loop, so only the timbre is piano-ish.
        """
        num_harmonics = max(1, int(max_frequency // fundamental))
        self.waves = []
        for n in range(1, num_harmonics + 1):
            freq = round(n * fundamental)
            amplitude = 1 / n**rolloff

            # spread hues around the colour wheel so each partial is distinct
            r, g, b = colorsys.hsv_to_rgb((n - 1) / num_harmonics, 0.8, 1.0)
            color = arcade.types.Color(int(r * 255), int(g * 255), int(b * 255), 255)
            self.waves.append(WaveParam(freq, amplitude, 0, color))

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
