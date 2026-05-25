from __future__ import annotations

import arcade

from level_gen import generate_level
import astar

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


def screen_to_map(screen_x, screen_y):
    return (screen_x // TILE_SIZE, screen_y // RAW_TILE_SIZE)


class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BLACK)
        self.sheet = arcade.load_spritesheet(SHEET_PATH)
        self.astar_flood = None
        self.wall_sprites = arcade.SpriteList()
        self.setup_level()

    def setup_level(self):
        self.level_int = generate_level(MAZE_WIDTH, MAZE_HEIGHT)
        wall_tex = self.get_texture("stone1")
        floor_tex = self.get_texture("floor1")
        for i in range(MAZE_WIDTH):
            for j in range(MAZE_HEIGHT):
                if self.level_int[i][j] != 0:
                    if self.level_int[i][j] == 1:
                        sprite = tex_to_sprite(wall_tex)
                    else:
                        sprite = tex_to_sprite(floor_tex)
                    sprite.center_x = i * TILE_SIZE + TILE_SIZE / 2
                    sprite.center_y = j * TILE_SIZE + TILE_SIZE / 2
                    self.wall_sprites.append(sprite)

    def get_texture(self, name: str) -> arcade.Sprite:
        x, y, w, h = SPRITES_COORDS[name]
        return self.sheet.get_texture(arcade.LBWH(x, y, w, h))

    def on_update(self, delta_time):
        pass

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            map_x, map_y = screen_to_map(x, y)
            self.astar_flood = astar.astar_flood(
                self.level_int, astar.Coord(map_x, map_y), True, 9
            )

    def draw_astar_flood(self):
        if self.astar_flood:
            for x in range(MAZE_WIDTH):
                for y in range(MAZE_HEIGHT):
                    if self.level_int[y][x] == 0:
                        cx = x * TILE_SIZE + TILE_SIZE / 2
                        cy = y * TILE_SIZE + TILE_SIZE / 2
                        tile_text = str(self.astar_flood[y][x])
                        print("tile value", self.astar_flood[y][x], tile_text)
                        arcade.draw_text(
                            tile_text,
                            cx,
                            cy,
                            arcade.color.WHITE,
                            10,
                            anchor_x="center",
                            anchor_y="center",
                        )

    def on_draw(self):
        self.clear()
        self.wall_sprites.draw()
        self.draw_astar_flood()

    def on_key_press(self, key, modifiers):
        pass

    def on_key_release(self, key, modifiers):
        pass


def main():
    window = Game()
    arcade.run()


if __name__ == "__main__":
    main()
