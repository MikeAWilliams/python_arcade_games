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

        #self.setup_simple_waves()
        self.setup_piano_waves()

        self.generate_data()
        sd.play(self.audio_samples, self.audio_sample_rate, loop=True)

    def setup_simple_waves(self) -> None:
        self.waves = [
            # WaveParam(4000, 1, 0, arcade.color.GREEN),
            WaveParam(400, 1, 0, arcade.color.GREEN),
            WaveParam(400, 1, math.pi / 2, arcade.color.GREEN),
        ]

    def setup_piano_waves(
        self,
        fundamental: float = 261.63,  # middle C (about 262 Hz)
        num_harmonics: int = 10,
        rolloff: float = 1.2,
        strike_position: float = 1 / 8,
        inharmonicity: float = 0.0004,
        snap_to_whole_hz: bool = True,
    ) -> None:
        """Build a static approximation of a piano tone out of sine waves.

        A real piano note is a stack of partials (sine waves) near whole-number
        multiples of the fundamental. Three things make it sound piano-like
        rather than like an organ or a plain beep:

        fundamental      pitch of the note in Hz.
        num_harmonics    how many partials to stack. More = brighter, but the
                         high ones are quiet anyway.
        rolloff          partial n has amplitude 1 / n**rolloff. Higher = duller
                         (more like a soft felt hammer); lower = brighter/buzzier.
        strike_position  where the hammer hits the string as a fraction of its
                         length (pianos use roughly 1/7 to 1/9). A string can't
                         vibrate with a node where it is struck, so harmonic n
                         is scaled by |sin(pi * n * strike_position)|; with 1/8
                         the 8th harmonic vanishes entirely.
        inharmonicity    stiff piano strings make partials slightly sharp:
                         f_n = n * f0 * sqrt(1 + B * n**2). B is about 0.0001
                         to 0.001 for real pianos. 0 gives perfect harmonics.
        snap_to_whole_hz round each frequency to a whole number of Hz. The audio
                         buffer is looped and is audio_time seconds long, so
                         whole-Hz partials (with a whole-second buffer) finish
                         a complete cycle and loop without a click.

        Not modelled: the decaying envelope (attack then fade) of a real
        key press. This tone is a steady sustained loop, so it is the *timbre*
        that is piano-ish, not the dynamics.
        """
        self.waves = []
        for n in range(1, num_harmonics + 1):
            freq = n * fundamental * math.sqrt(1 + inharmonicity * n * n)
            if snap_to_whole_hz:
                freq = round(freq)

            amplitude = (1 / n**rolloff) * abs(math.sin(math.pi * n * strike_position))
            if amplitude < 1e-3:
                continue  # an (almost) absent harmonic; nothing to draw or hear

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
