#!/usr/bin/env python3
"""
Visualize NN training state data from a .npz recording file.
Shows what the model sees: player, asteroids, velocities, and action indicators.

Controls:
  SPACE      – pause / unpause
  RIGHT      – skip to next game

Usage:
    python tools/visualize_state.py                    # first .npz in data/
    python tools/visualize_state.py data/recording.npz
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
from asteroids.core.game import (
    Action,
    BIG_ASTEROID_RADIUS,
    MEDIUM_ASTEROID_RADIUS,
    PLAYER_RADIUS,
    SMALL_ASTEROID_RADIUS,
)

# ── tuneable constants ─────────────────────────────────────────────────────────
DISPLAY_SIZE     = 720   # square window side in pixels
GAME_WIDTH       = 1280  # original game width used for normalising positions/velocities
VEL_SCALE        = 1.0   # 1.0 → arrow length = 1-second travel distance in display coords
INDICATOR_TICKS  = 6     # ticks to hold SHOOT / ACCEL / DECEL indicator on screen
TRANSITION_TICKS = 8     # ticks to show white flash between games
# ──────────────────────────────────────────────────────────────────────────────


def _asteroid_radius(slot: int) -> float:
    """Return game-pixel radius for an asteroid in the given state-vector slot."""
    if slot < 3:
        return BIG_ASTEROID_RADIUS
    if slot < 9:
        return MEDIUM_ASTEROID_RADIUS
    return SMALL_ASTEROID_RADIUS


class VisualizerView(arcade.View):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        game_ids: np.ndarray,
        tick_nums: np.ndarray,
        filename: str,
    ):
        super().__init__()

        # Sort by (game_id, tick_num) so games play in order
        order = np.lexsort((tick_nums, game_ids))
        self.states   = states[order]
        self.actions  = actions[order]
        self.game_ids = game_ids[order]

        self.filename    = filename
        self.total_games = int(len(np.unique(game_ids)))

        # Playback state
        self.frame_ptr       = 0
        self.display_frame   = 0
        self.current_game_id = None
        self.game_number     = 0
        self.paused          = False

        # Transition flash
        self.transitioning    = False
        self.transition_ticks = 0

        # Action indicator
        self.indicator       = None   # Action or None
        self.indicator_ticks = 0

        # Pre-built Text objects (avoids per-frame PerformanceWarning)
        self.hud_text = arcade.Text(
            "", 6, 6, arcade.color.WHITE, font_size=11,
        )
        self.transition_text = arcade.Text(
            "", DISPLAY_SIZE / 2, DISPLAY_SIZE / 2,
            arcade.color.BLACK, font_size=28,
            anchor_x="center", anchor_y="center", bold=True,
        )

    # ── coordinate helpers ─────────────────────────────────────────────────────

    def _s(self, n: float) -> float:
        """Normalised [0, 1] → display pixels."""
        return n * DISPLAY_SIZE

    def _sr(self, r: float) -> float:
        """Game-pixel radius → display pixels."""
        return r / GAME_WIDTH * DISPLAY_SIZE

    # ── arcade callbacks ───────────────────────────────────────────────────────

    def on_key_press(self, key, _modifiers):
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.RIGHT:
            self._skip_game()

    def _skip_game(self):
        if self.frame_ptr >= len(self.states):
            return
        cur = int(self.game_ids[self.frame_ptr])
        while self.frame_ptr < len(self.states) and int(self.game_ids[self.frame_ptr]) == cur:
            self.frame_ptr += 1
        self.indicator       = None
        self.indicator_ticks = 0

    def on_update(self, dt):
        if self.paused:
            return

        # Count down transition flash
        if self.transitioning:
            self.transition_ticks -= 1
            if self.transition_ticks <= 0:
                self.transitioning = False
            return

        # Loop when all data played
        if self.frame_ptr >= len(self.states):
            self.frame_ptr       = 0
            self.current_game_id = None
            self.game_number     = 0

        new_id = int(self.game_ids[self.frame_ptr])

        # Game boundary → trigger white flash
        if self.current_game_id is not None and new_id != self.current_game_id:
            self.game_number    += 1
            self.current_game_id = new_id
            self.indicator       = None
            self.indicator_ticks = 0
            self.transitioning   = True
            self.transition_ticks = TRANSITION_TICKS
            return  # hold frame_ptr until flash finishes

        self.current_game_id = new_id
        self.display_frame   = self.frame_ptr

        # Update indicator
        action = Action(int(self.actions[self.frame_ptr]))
        if action in (Action.ACCELERATE, Action.DECELERATE, Action.SHOOT):
            self.indicator       = action
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
                [(0, 0), (DISPLAY_SIZE, 0), (DISPLAY_SIZE, DISPLAY_SIZE), (0, DISPLAY_SIZE)],
                arcade.color.WHITE,
            )
            self.transition_text.text = f"Game {self.game_number + 1} / {self.total_games}"
            self.transition_text.draw()
            return

        s      = self.states[self.display_frame]
        action = Action(int(self.actions[self.display_frame]))

        # ── decode player ──────────────────────────────────────────────────
        px      = self._s(s[0])
        py      = self._s(s[1])
        pvx     = s[2]
        pvy     = s[3]
        bx      = s[4]   # cos(angle)
        by      = s[5]   # sin(angle)
        pr      = self._sr(PLAYER_RADIUS)
        p_angle = math.atan2(by, bx)

        # ── asteroids ──────────────────────────────────────────────────────
        for i in range(27):
            base   = 7 + i * 5
            ax     = self._s(s[base])
            ay     = self._s(s[base + 1])
            avx    = s[base + 2]
            avy    = s[base + 3]
            active = s[base + 4] > 0.5
            r      = self._sr(_asteroid_radius(i))

            if active:
                arcade.draw_circle_filled(ax, ay, r, (160, 160, 160))
                self._draw_arrow(ax, ay, avx, avy, arcade.color.LIGHT_BLUE)
            else:
                arcade.draw_circle_outline(ax, ay, r, (60, 60, 60), border_width=1)

        # ── decelerate indicator: red circle behind ship ───────────────────
        if self.indicator == Action.DECELERATE:
            ix = px - bx * pr * 2.5
            iy = py - by * pr * 2.5
            arcade.draw_circle_filled(ix, iy, pr * 0.9, arcade.color.RED)

        # ── player ─────────────────────────────────────────────────────────
        arcade.draw_circle_outline(px, py, pr, arcade.color.WHITE, border_width=1)
        x1 = px + pr * math.cos(p_angle)
        y1 = py + pr * math.sin(p_angle)
        x2 = px + pr * math.cos(0.8 * math.pi + p_angle)
        y2 = py + pr * math.sin(0.8 * math.pi + p_angle)
        x3 = px + pr * math.cos(1.2 * math.pi + p_angle)
        y3 = py + pr * math.sin(1.2 * math.pi + p_angle)
        arcade.draw_triangle_filled(x1, y1, x2, y2, x3, y3, arcade.color.WHITE)

        # ── accelerate indicator: red circle in front of ship ──────────────
        if self.indicator == Action.ACCELERATE:
            ix = px + bx * pr * 2.5
            iy = py + by * pr * 2.5
            arcade.draw_circle_filled(ix, iy, pr * 0.9, arcade.color.RED)

        # ── shoot indicator: green rectangle along bearing ──────────────────
        elif self.indicator == Action.SHOOT:
            offset = pr * 3.5
            cx = px + bx * offset
            cy = py + by * offset
            self._draw_rotated_rect(cx, cy, pr * 3.0, pr * 0.7, p_angle, arcade.color.GREEN)

        # ── player velocity arrow ───────────────────────────────────────────
        self._draw_arrow(px, py, pvx, pvy, arcade.color.YELLOW)

        # ── HUD ────────────────────────────────────────────────────────────
        self.hud_text.text = (
            f"{os.path.basename(self.filename)}  │  "
            f"game {self.game_number + 1}/{self.total_games}  │  "
            f"{action.name}"
            + ("  │  [PAUSED]" if self.paused else "")
        )
        self.hud_text.draw()

    def _draw_rotated_rect(self, cx, cy, length, width, angle, color):
        """Draw a filled rectangle centred at (cx,cy), long axis along angle (radians)."""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw, hl = width / 2, length / 2
        corners = [
            (cx + hl * cos_a - hw * sin_a, cy + hl * sin_a + hw * cos_a),
            (cx - hl * cos_a - hw * sin_a, cy - hl * sin_a + hw * cos_a),
            (cx - hl * cos_a + hw * sin_a, cy - hl * sin_a - hw * cos_a),
            (cx + hl * cos_a + hw * sin_a, cy + hl * sin_a - hw * cos_a),
        ]
        arcade.draw_polygon_filled(corners, color)

    def _draw_arrow(self, ox: float, oy: float, nvx: float, nvy: float, color):
        """
        Draw a velocity arrow from (ox, oy).
        nvx / nvy are normalised velocities (divided by game width/height).
        """
        speed = math.sqrt(nvx * nvx + nvy * nvy)
        if speed < 1e-4:
            return

        tip_x = ox + nvx * VEL_SCALE * DISPLAY_SIZE
        tip_y = oy + nvy * VEL_SCALE * DISPLAY_SIZE
        arcade.draw_line(ox, oy, tip_x, tip_y, color, line_width=2)

        # Arrowhead
        angle    = math.atan2(nvy, nvx)
        head_len = min(12.0, VEL_SCALE * DISPLAY_SIZE * speed * 0.35)
        spread   = 0.4
        hx1 = tip_x - head_len * math.cos(angle - spread)
        hy1 = tip_y - head_len * math.sin(angle - spread)
        hx2 = tip_x - head_len * math.cos(angle + spread)
        hy2 = tip_y - head_len * math.sin(angle + spread)
        arcade.draw_triangle_filled(tip_x, tip_y, hx1, hy1, hx2, hy2, color)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NN training state data from a .npz file"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .npz data file (default: first in data/ sorted numerically)",
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
    raw       = np.load(path)
    states    = raw["states"]
    actions   = raw["actions"]
    game_ids  = raw["game_ids"]
    tick_nums = raw["tick_nums"]
    n_games   = len(np.unique(game_ids))
    print(f"  {len(states):,} frames across {n_games:,} games")

    window = arcade.Window(
        DISPLAY_SIZE, DISPLAY_SIZE,
        f"State Visualizer — {os.path.basename(path)}",
    )
    window.set_update_rate(1 / 60)
    view = VisualizerView(states, actions, game_ids, tick_nums, path)
    window.show_view(view)
    arcade.run()


if __name__ == "__main__":
    main()
