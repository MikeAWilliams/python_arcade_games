#!/usr/bin/env python3
"""
Visualize PolarNN state data converted from raw geometry recordings.
Loads a .npz file in RawGeometryNN format, converts each frame to PolarNN
format, and renders the player-relative view.

The player is always at center, facing right. Asteroids are drawn at their
relative distance and angle from the player's bearing.

Controls:
  SPACE      – pause / unpause
  RIGHT      – skip to next game

Usage:
    python tools/visualize_state_polar.py
    python tools/visualize_state_polar.py data/recording.npz
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import arcade
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asteroids.ai.polar_nn import MAX_ASTEROID_SLOTS, convert_raw_geometry_state
from asteroids.core.game import Action, PLAYER_RADIUS

# ── tuneable constants ─────────────────────────────────────────────────────────
DISPLAY_SIZE = 720  # square window side in pixels
GAME_WIDTH = 1280
GAME_HEIGHT = 720
INDICATOR_TICKS = 6
TRANSITION_TICKS = 8
# ──────────────────────────────────────────────────────────────────────────────


def _size_to_radius(size_val: float) -> float:
    """Map size category back to a display radius."""
    if size_val > 0.8:
        return 90.0
    elif size_val > 0.5:
        return 60.0
    else:
        return 30.0


class PolarVisualizerView(arcade.View):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        game_ids: np.ndarray,
        tick_nums: np.ndarray,
        filename: str,
    ):
        super().__init__()

        # Sort by (game_id, tick_num)
        order = np.lexsort((tick_nums, game_ids))
        self.raw_states = states[order]
        self.actions = actions[order]
        self.game_ids = game_ids[order]

        self.filename = filename
        self.total_games = int(len(np.unique(game_ids)))

        # Playback state
        self.frame_ptr = 0
        self.display_frame = 0
        self.current_game_id = None
        self.game_number = 0
        self.paused = False

        # Transition flash
        self.transitioning = False
        self.transition_ticks = 0

        # Action indicator
        self.indicator = None
        self.indicator_ticks = 0

        # Pre-built Text objects
        self.hud_text = arcade.Text(
            "",
            6,
            6,
            arcade.color.WHITE,
            font_size=11,
        )
        self.transition_text = arcade.Text(
            "",
            DISPLAY_SIZE / 2,
            DISPLAY_SIZE / 2,
            arcade.color.BLACK,
            font_size=28,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

    def _sr(self, r: float) -> float:
        """Game-pixel radius → display pixels."""
        return r / GAME_WIDTH * DISPLAY_SIZE

    def on_key_press(self, key, _modifiers):
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.RIGHT:
            self._skip_game()

    def _skip_game(self):
        if self.frame_ptr >= len(self.raw_states):
            return
        cur = int(self.game_ids[self.frame_ptr])
        while (
            self.frame_ptr < len(self.raw_states)
            and int(self.game_ids[self.frame_ptr]) == cur
        ):
            self.frame_ptr += 1
        self.indicator = None
        self.indicator_ticks = 0

    def on_update(self, dt):
        if self.paused:
            return

        if self.transitioning:
            self.transition_ticks -= 1
            if self.transition_ticks <= 0:
                self.transitioning = False
            return

        if self.frame_ptr >= len(self.raw_states):
            self.frame_ptr = 0
            self.current_game_id = None
            self.game_number = 0

        new_id = int(self.game_ids[self.frame_ptr])

        if self.current_game_id is not None and new_id != self.current_game_id:
            self.game_number += 1
            self.current_game_id = new_id
            self.indicator = None
            self.indicator_ticks = 0
            self.transitioning = True
            self.transition_ticks = TRANSITION_TICKS
            return

        self.current_game_id = new_id
        self.display_frame = self.frame_ptr

        action = Action(int(self.actions[self.frame_ptr]))
        if action in (Action.ACCELERATE, Action.DECELERATE, Action.SHOOT):
            self.indicator = action
            self.indicator_ticks = INDICATOR_TICKS
        elif self.indicator_ticks > 0:
            self.indicator_ticks -= 1
            if self.indicator_ticks == 0:
                self.indicator = None

        self.frame_ptr += 1

    def on_draw(self):
        self.clear()

        # ── white flash between games ──────────────────────────────────────
        if self.transitioning:
            arcade.draw_polygon_filled(
                [
                    (0, 0),
                    (DISPLAY_SIZE, 0),
                    (DISPLAY_SIZE, DISPLAY_SIZE),
                    (0, DISPLAY_SIZE),
                ],
                arcade.color.WHITE,
            )
            self.transition_text.text = (
                f"Game {self.game_number + 1} / {self.total_games}"
            )
            self.transition_text.draw()
            return

        # Convert raw state to polar
        raw = self.raw_states[self.display_frame]
        polar = convert_raw_geometry_state(raw, GAME_WIDTH, GAME_HEIGHT)
        action = Action(int(self.actions[self.display_frame]))

        # ── decode polar state ──────────────────────────────────────────────
        player_speed = polar[0]
        shot_cooldown = polar[1]
        asteroid_count = polar[2]

        cx = DISPLAY_SIZE / 2
        cy = DISPLAY_SIZE / 2
        pr = self._sr(PLAYER_RADIUS)

        # ── draw range rings ────────────────────────────────────────────────
        for frac in (0.25, 0.5, 0.75):
            ring_r = frac * DISPLAY_SIZE / 2
            arcade.draw_circle_outline(cx, cy, ring_r, (40, 40, 40), border_width=1)

        # ── draw asteroids ──────────────────────────────────────────────────
        for i in range(MAX_ASTEROID_SLOTS):
            base = 3 + i * 4
            dist_norm = polar[base]
            rel_angle_norm = polar[base + 1]
            closing_norm = polar[base + 2]
            size_val = polar[base + 3]

            if size_val < 0.01:
                continue  # empty slot

            # Convert back to display coordinates
            # Distance is normalized by screen diagonal, map to display
            display_dist = dist_norm * DISPLAY_SIZE / 2
            # Angle is normalized by pi, player faces right (angle=0)
            angle = rel_angle_norm * math.pi

            ax = cx + display_dist * math.cos(angle)
            ay = cy + display_dist * math.sin(angle)
            ar = self._sr(_size_to_radius(size_val))

            # Color based on closing speed: red=approaching, blue=receding
            if closing_norm > 0.01:
                intensity = min(255, int(100 + closing_norm * 500))
                color = (intensity, 80, 80)
            elif closing_norm < -0.01:
                intensity = min(255, int(100 + abs(closing_norm) * 500))
                color = (80, 80, intensity)
            else:
                color = (160, 160, 160)

            arcade.draw_circle_filled(ax, ay, ar, color)

            # Draw closing speed as a line toward/away from center
            if abs(closing_norm) > 0.005:
                line_len = closing_norm * DISPLAY_SIZE * 0.3
                lx = ax - line_len * math.cos(angle)
                ly = ay - line_len * math.sin(angle)
                line_color = (
                    arcade.color.RED if closing_norm > 0 else arcade.color.LIGHT_BLUE
                )
                arcade.draw_line(ax, ay, lx, ly, line_color, line_width=2)

        # ── player (always at center, facing right) ─────────────────────────
        # Decelerate indicator
        if self.indicator == Action.DECELERATE:
            arcade.draw_circle_filled(cx - pr * 2.5, cy, pr * 0.9, arcade.color.RED)

        arcade.draw_circle_outline(cx, cy, pr, arcade.color.WHITE, border_width=1)
        # Triangle facing right (angle=0)
        x1 = cx + pr
        y1 = cy
        x2 = cx + pr * math.cos(0.8 * math.pi)
        y2 = cy + pr * math.sin(0.8 * math.pi)
        x3 = cx + pr * math.cos(1.2 * math.pi)
        y3 = cy + pr * math.sin(1.2 * math.pi)
        arcade.draw_triangle_filled(x1, y1, x2, y2, x3, y3, arcade.color.WHITE)

        # Accelerate indicator
        if self.indicator == Action.ACCELERATE:
            arcade.draw_circle_filled(cx + pr * 2.5, cy, pr * 0.9, arcade.color.RED)

        # Shoot indicator
        elif self.indicator == Action.SHOOT:
            corners = [
                (cx + pr * 2.5, cy + pr * 0.35),
                (cx + pr * 2.5, cy - pr * 0.35),
                (cx + pr * 5.0, cy - pr * 0.35),
                (cx + pr * 5.0, cy + pr * 0.35),
            ]
            arcade.draw_polygon_filled(corners, arcade.color.GREEN)

        # ── speed indicator bar ─────────────────────────────────────────────
        bar_max = 100
        bar_h = 8
        bar_x = 10
        bar_y = DISPLAY_SIZE - 20
        arcade.draw_line(bar_x, bar_y, bar_x + bar_max, bar_y, (60, 60, 60), bar_h)
        speed_len = player_speed * bar_max * 5  # scale for visibility
        arcade.draw_line(
            bar_x,
            bar_y,
            bar_x + min(speed_len, bar_max),
            bar_y,
            arcade.color.YELLOW,
            bar_h,
        )

        # ── cooldown bar ────────────────────────────────────────────────────
        cd_y = DISPLAY_SIZE - 35
        arcade.draw_line(bar_x, cd_y, bar_x + bar_max, cd_y, (60, 60, 60), bar_h)
        cd_len = shot_cooldown * bar_max
        arcade.draw_line(bar_x, cd_y, bar_x + cd_len, cd_y, arcade.color.ORANGE, bar_h)

        # ── HUD ────────────────────────────────────────────────────────────
        active = int(round(asteroid_count * MAX_ASTEROID_SLOTS))
        self.hud_text.text = (
            f"{os.path.basename(self.filename)}  │  "
            f"game {self.game_number + 1}/{self.total_games}  │  "
            f"{action.name}  │  "
            f"asteroids: {active}" + ("  │  [PAUSED]" if self.paused else "")
        )
        self.hud_text.draw()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PolarNN state data converted from raw geometry recordings"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .npz data file (default: first training_data20k_converted in data/)",
    )
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        candidates = sorted(
            glob.glob("data/training_data20k_converted_*.npz"),
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )
        if not candidates:
            print("No .npz files found in data/")
            sys.exit(1)
        path = candidates[0]

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path} ...")
    raw = np.load(path)
    states = raw["states"]
    actions = raw["actions"]
    game_ids = raw["game_ids"]
    tick_nums = raw["tick_nums"]
    n_games = len(np.unique(game_ids))
    print(f"  {len(states):,} frames across {n_games:,} games")
    print(f"  Converting to polar format...")

    window = arcade.Window(
        DISPLAY_SIZE,
        DISPLAY_SIZE,
        f"Polar State Visualizer — {os.path.basename(path)}",
    )
    window.set_update_rate(1 / 60)
    view = PolarVisualizerView(states, actions, game_ids, tick_nums, path)
    window.show_view(view)
    arcade.run()


if __name__ == "__main__":
    main()
