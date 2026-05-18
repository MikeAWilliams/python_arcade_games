from __future__ import annotations

import arcade
import random
from dataclasses import dataclass, field

SCREEN_TITLE = "Maze Viewer"
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
RAW_TILE_SIZE = 12
MAZE_WIDTH = 640
MAZE_WIDTH = 320
MAZE_HEIGHT = 340
MAZE_HEIGHT = 120
DISPLAY_SCALE = min(
    SCREEN_WIDTH / (MAZE_WIDTH * RAW_TILE_SIZE),
    SCREEN_HEIGHT / (MAZE_HEIGHT * RAW_TILE_SIZE),
)
TILE_SIZE = RAW_TILE_SIZE * DISPLAY_SCALE
MIN_DIM = 10
PAD = 2
MIN_LEAF = MIN_DIM + 2 * PAD
MIN_CORIDOR_WIDTH = 3

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
    room: Rect | None = None
    corridors: list[Rect] = field(default_factory=list)
    l: Rect | None = None
    r: Rect | None = None


def recursive_generate_rect(parent):
    TARGET_AREA = 1000

    eligible_divide = []
    if parent.w >= MIN_LEAF * 2 + 1:
        eligible_divide.append("w")
    if parent.h >= MIN_LEAF * 2 + 1:
        eligible_divide.append("h")
    if len(eligible_divide) == 0:
        return

    dim = random.choice(eligible_divide)
    if dim == "w":
        cut = random.randint(MIN_LEAF, parent.w - MIN_LEAF - 1)
        parent.l = Rect(parent.i, parent.j, cut, parent.h)
        parent.r = Rect(parent.i + cut + 1, parent.j, parent.w - cut - 1, parent.h)
    else:
        cut = random.randint(MIN_LEAF, parent.h - MIN_LEAF - 1)
        parent.l = Rect(parent.i, parent.j, parent.w, cut)
        parent.r = Rect(parent.i, parent.j + cut + 1, parent.w, parent.h - cut - 1)

    if (parent.l.w * parent.l.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.l)
    if (parent.r.w * parent.r.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.r)


# for debug, set level 0 before you start and use this to draw the rectangles
def recursive_set_rect_bnd_1(rect, level):
    def set_rect_bnd_1(rect, level):
        for i in range(rect.i, rect.i + rect.w + 1):
            level[i][rect.j] = 1
            level[i][rect.h + rect.j] = 1
        for j in range(rect.j, rect.j + rect.h + 1):
            level[rect.i][j] = 1
            level[rect.w + rect.i][j] = 1
        # top rirght
        level[rect.i + rect.w][rect.j + rect.h] = 1

    set_rect_bnd_1(rect, level)
    if rect.l:
        recursive_set_rect_bnd_1(rect.l, level)
    if rect.r:
        recursive_set_rect_bnd_1(rect.r, level)


def recursive_set_coridor_0(root, level):
    if root.l:
        recursive_set_coridor_0(root.l, level)
    if root.r:
        recursive_set_coridor_0(root.r, level)
    if len(root.corridors) > 0:
        for r in root.corridors:
            fill_rect_0(r, level)


def fill_rect_0(rect, level):
    for i in range(rect.i, rect.i + rect.w):
        for j in range(rect.j, rect.j + rect.h):
            level[i][j] = 0


def recursive_set_room_0(root, level):
    if root.l:
        recursive_set_room_0(root.l, level)
    if root.r:
        recursive_set_room_0(root.r, level)

    if root.room:
        fill_rect_0(root.room, level)


def generate_coridors_two_rooms(room1, room2):
    # Vertical split: rooms side by side — horizontal corridor.
    if room1.i + room1.w <= room2.i or room2.i + room2.w <= room1.i:
        if room2.i < room1.i:
            room1, room2 = room2, room1
        overlap_start = max(room1.j, room2.j)
        overlap_end = min(room1.j + room1.h, room2.j + room2.h)
        overlap = overlap_end - overlap_start
        if overlap >= MIN_CORIDOR_WIDTH:
            width = random.randint(MIN_CORIDOR_WIDTH, overlap)
            j = random.randint(overlap_start, overlap_end - width)
            start_i = room1.i + room1.w
            length = room2.i - start_i
            return [Rect(start_i, j, length, width)]
        raise NotImplementedError("L-bend")

    # Horizontal split: rooms stacked — vertical corridor.
    if room2.j < room1.j:
        room1, room2 = room2, room1
    overlap_start = max(room1.i, room2.i)
    overlap_end = min(room1.i + room1.w, room2.i + room2.w)
    overlap = overlap_end - overlap_start
    if overlap >= MIN_CORIDOR_WIDTH:
        width = random.randint(MIN_CORIDOR_WIDTH, overlap)
        i = random.randint(overlap_start, overlap_end - width)
        start_j = room1.j + room1.h
        length = room2.j - start_j
        return [Rect(i, start_j, width, length)]
    raise NotImplementedError("L-bend")


def pick_room_in_subtree(node):
    if node.room:
        return node.room
    if node.l:
        room = pick_room_in_subtree(node.l)
        if room:
            return room
    if node.r:
        return pick_room_in_subtree(node.r)
    return None


def generate_coridors(root):
    if root.l and not root.l.room:
        generate_coridors(root.l)
    if root.r and not root.r.room:
        generate_coridors(root.r)

    if root.l and root.r:
        lroom = pick_room_in_subtree(root.l)
        rroom = pick_room_in_subtree(root.r)
        root.corridors = generate_coridors_two_rooms(lroom, rroom)


def generate_random_room_in_leaves(root):
    if root.l:
        generate_random_room_in_leaves(root.l)
    if root.r:
        generate_random_room_in_leaves(root.r)

    if not root.l and not root.r:
        # this is a leaf — fill most of it, with small random shrink + offset.
        # BSP guarantees root.w >= MIN_LEAF, so max_w >= MIN_DIM always.
        MAX_SHRINK = 4

        max_w = root.w - 2 * PAD
        max_h = root.h - 2 * PAD

        shrink_w = random.randint(0, min(MAX_SHRINK, max_w - MIN_DIM))
        room_w = max_w - shrink_w
        room_i = root.i + PAD + random.randint(0, shrink_w)

        shrink_h = random.randint(0, min(MAX_SHRINK, max_h - MIN_DIM))
        room_h = max_h - shrink_h
        room_j = root.j + PAD + random.randint(0, shrink_h)

        root.room = Rect(room_i, room_j, room_w, room_h)


# BSP room generation
def generate_level(width, height, seed=42):
    random.seed(seed)
    root = Rect(0, 0, width - 1, height - 1)
    recursive_generate_rect(root)
    generate_random_room_in_leaves(root)
    generate_coridors(root)

    level = [[2 for _ in range(height)] for _ in range(width)]
    # draw each rectangle (which is wrong, but I want to see it)
    recursive_set_rect_bnd_1(root, level)
    recursive_set_room_0(root, level)
    recursive_set_coridor_0(root, level)

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
        floor_tex = self.get_texture("floor1")
        for i in range(MAZE_WIDTH):
            for j in range(MAZE_HEIGHT):
                if level_int[i][j] != 0:
                    if level_int[i][j] == 1:
                        sprite = tex_to_sprite(wall_tex)
                    else:
                        sprite = tex_to_sprite(floor_tex)
                    sprite.center_x = i * TILE_SIZE + TILE_SIZE / 2
                    sprite.center_y = j * TILE_SIZE + TILE_SIZE / 2
                    self.sprites.append(sprite)

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
