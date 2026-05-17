from __future__ import annotations

import arcade
import random
from dataclasses import dataclass, field

SCREEN_TITLE = "Maze Viewer"
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
RAW_TILE_SIZE = 12
MAZE_WIDTH = 640 * 2
MAZE_HEIGHT = 340 * 2
DISPLAY_SCALE = min(
    SCREEN_WIDTH / (MAZE_WIDTH * RAW_TILE_SIZE),
    SCREEN_HEIGHT / (MAZE_HEIGHT * RAW_TILE_SIZE),
)
TILE_SIZE = RAW_TILE_SIZE * DISPLAY_SCALE

SHEET_PATH = "assets/urizen_onebit_tileset__v2d0.png"

SPRITES_COORDS: dict[str, tuple[int, int, int, int]] = {
    "stone1": (1, 27, 12, 12),
    "stone2": (14, 27, 12, 12),
    "stonecorner1": (66, 27, 12, 12),
    "stonecorner2": (79, 27, 12, 12),
    "stonecorner3": (92, 27, 12, 12),
    "floor1": (1, 92, 12, 12),
    "floor2": (14, 92, 12, 12),
    "floor3": (27, 92, 12, 12),
}


@dataclass
class Rect:
    i: int
    j: int
    w: int
    h: int
    l: Rect | None = None
    r: Rect | None = None


def recursive_generate_rect(parent):
    MIN_DIM = 10
    TARGET_AREA = 1000

    eligible_divide = []
    if parent.w >= MIN_DIM * 2 + 1:
        eligible_divide.append("w")
    if parent.h >= MIN_DIM * 2 + 1:
        eligible_divide.append("h")
    if len(eligible_divide) == 0:
        return

    dim = random.choice(eligible_divide)
    if dim == "w":
        print("cut w")
        cut = random.randint(MIN_DIM, parent.w - MIN_DIM - 1)
        print("cut", cut)
        parent.l = Rect(parent.i, parent.j, cut, parent.h)
        parent.r = Rect(parent.i + cut + 1, parent.j, parent.w - cut - 1, parent.h)
    else:
        print("cut h")
        cut = random.randint(MIN_DIM, parent.h - MIN_DIM - 1)
        parent.l = Rect(parent.i, parent.j, parent.w, cut)
        parent.r = Rect(parent.i, parent.j + cut + 1, parent.w, parent.h - cut - 1)

    if (parent.l.w * parent.l.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.l)
    if (parent.r.w * parent.r.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.r)


def set_rect_bnd_1(rect, level):
    print("drawing rect", rect)
    for i in range(rect.i, rect.i + rect.w + 1):
        level[i][rect.j] = 1
        level[i][rect.h + rect.j] = 1
    for j in range(rect.j, rect.j + rect.h + 1):
        level[rect.i][j] = 1
        level[rect.w + rect.i][j] = 1
    # top rirght
    level[rect.i + rect.w][rect.j + rect.h] = 1


def recursive_set_rect_bdn_1(rect, level):
    set_rect_bnd_1(rect, level)
    if rect.l:
        recursive_set_rect_bdn_1(rect.l, level)
    if rect.r:
        recursive_set_rect_bdn_1(rect.r, level)


# BSP room generation
def generate_level(width, height, seed=42):
    random.seed(seed)
    root = Rect(0, 0, width - 1, height - 1)
    recursive_generate_rect(root)
    # all 1 for wall at first. We will carve out the rooms
    # level = [[1 for _ in range(height)] for _ in range(width)]
    # all 0 for now, show nothing at first
    level = [[0 for _ in range(height)] for _ in range(width)]

    # draw each rectangle (which is wrong, but I want to see it)
    recursive_set_rect_bdn_1(root, level)

    return level


def tex_to_sprite(tex):
    return arcade.Sprite(tex, DISPLAY_SCALE)


class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BLACK)
        self.sheet = arcade.load_spritesheet(SHEET_PATH)
        self.sprites = arcade.SpriteList()
        self.setup_level()

    def setup_level(self):
        level_int = generate_level(MAZE_WIDTH, MAZE_HEIGHT)
        wall_tex = self.get_texture("stone1")
        for i in range(MAZE_WIDTH):
            for j in range(MAZE_HEIGHT):
                if level_int[i][j] == 1:
                    wall = tex_to_sprite(wall_tex)
                    wall.center_x = i * TILE_SIZE + TILE_SIZE / 2
                    wall.center_y = j * TILE_SIZE + TILE_SIZE / 2
                    self.sprites.append(wall)

    def get_texture(self, name: str) -> arcade.Sprite:
        x, y, w, h = SPRITES_COORDS[name]
        return self.sheet.get_texture(arcade.LBWH(x, y, w, h))

    def on_update(self, delta_time):
        pass

    def on_draw(self):
        self.clear()
        self.sprites.draw()

    def on_key_press(self, key, modifiers):
        pass

    def on_key_release(self, key, modifiers):
        pass


def main():
    window = Game()
    arcade.run()


if __name__ == "__main__":
    main()
