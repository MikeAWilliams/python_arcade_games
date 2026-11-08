import colorsys
import math
from dataclasses import dataclass
from typing import NamedTuple

import arcade
import numpy as np
import sounddevice as sd

WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160


def midi_num_to_freq(midi: int) -> float:
    """given a midi number compute the frequency
    classically you use f = 440*2**(n/12) where n  is the number of semitones above or below A4
    a4 is midi 69
    """
    return 440 * 2 ** ((midi - 69) / 12)


# pitch class is the number of semitones above C within an octave (C is 0)
# fmt: off
PITCH_CLASS = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11,
}
# fmt: on


def midi_from_note_name(note_name):
    pitch_name = note_name.rstrip("0123456789")  # "Bb0" -> "Bb"
    octave = int(note_name[len(pitch_name) :])  # "Bb0" -> 0
    # pitch class is the count from c (c is zero) within an octive
    # it is the number of simi tones above c0
    pitch_class = PITCH_CLASS[pitch_name]
    return 12 * (octave + 1) + pitch_class


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
        self.generate_axes()
        # sample spacing is number of samples in the full x range
        self.num_samples = 1000

        self.sum_color = arcade.color.RED

        self.audio_sample_rate = 44100
        self.audio_volume = 0.4
        self.audio_time = 1

        # 21 is A0, adding one goes up one semitone
        self.mini_number = 60
        # if true use piano_waves
        self.is_piano = True
        self.update_waves_and_data()

    def update_waves_and_data(self):
        sd.stop()
        frequence = midi_num_to_freq(self.mini_number)
        if self.is_piano:
            self.setup_piano_waves(fundamental=frequence, rolloff=2.5)
        else:
            self.setup_simple_waves(freq=frequence)
        self.generate_data()
        sd.play(self.audio_samples, self.audio_sample_rate, loop=True)

    def setup_simple_waves(self, freq: float) -> None:
        self.waves = [
            WaveParam(freq, 1, 0, arcade.color.GREEN),
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

    def on_key_press(self, symbol, modifiers) -> None:
        change = False
        if symbol == arcade.key.DOWN and self.mini_number > 21:
            change = True
            self.mini_number -= 1
        if symbol == arcade.key.UP and self.mini_number < 108:
            change = True
            self.mini_number += 1

        if change:
            self.update_waves_and_data()

    def on_update(self, delta_time: float) -> bool | None:
        pass


def dump_note_frequencies():
    # fmt: off
    notes = [
    "A0", "A#0", "Bb0", "B0",
    "C1", "C#1", "Db1", "D1", "D#1", "Eb1", "E1", "F1", "F#1", "Gb1", "G1", "G#1", "Ab1", "A1", "A#1", "Bb1", "B1",
    "C2", "C#2", "Db2", "D2", "D#2", "Eb2", "E2", "F2", "F#2", "Gb2", "G2", "G#2", "Ab2", "A2", "A#2", "Bb2", "B2",
    "C3", "C#3", "Db3", "D3", "D#3", "Eb3", "E3", "F3", "F#3", "Gb3", "G3", "G#3", "Ab3", "A3", "A#3", "Bb3", "B3",
    "C4", "C#4", "Db4", "D4", "D#4", "Eb4", "E4", "F4", "F#4", "Gb4", "G4", "G#4", "Ab4", "A4", "A#4", "Bb4", "B4",
    "C5", "C#5", "Db5", "D5", "D#5", "Eb5", "E5", "F5", "F#5", "Gb5", "G5", "G#5", "Ab5", "A5", "A#5", "Bb5", "B5",
    "C6", "C#6", "Db6", "D6", "D#6", "Eb6", "E6", "F6", "F#6", "Gb6", "G6", "G#6", "Ab6", "A6", "A#6", "Bb6", "B6",
    "C7", "C#7", "Db7", "D7", "D#7", "Eb7", "E7", "F7", "F#7", "Gb7", "G7", "G#7", "Ab7", "A7", "A#7", "Bb7", "B7",
    "C8",
]
    # fmt: on
    for n in notes:
        midi = midi_from_note_name(n)
        print(n, midi, midi_num_to_freq(midi))


def main():
    window = arcade.Window(
        WINDOW_WIDTH, WINDOW_HEIGHT, "Show Me A Sound", resizable=True
    )
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
