from __future__ import annotations

import arcade

from level_gen import generate_level

SCREEN_TITLE = "Maze Viewer"
SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160
RAW_TILE_SIZE = 12
MAZE_WIDTH = 640
MAZE_HEIGHT = 340
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
