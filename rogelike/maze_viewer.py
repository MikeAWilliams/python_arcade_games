import arcade
import random

SCREEN_TITLE = "Maze Viewer"
DISPLAY_SCALE = 1
RAW_TILE_SIZE = 12
TILE_SIZE = RAW_TILE_SIZE * DISPLAY_SCALE
VIEW_WIDTH = 320
VIEW_HEIGHT = 180
SCREEN_WIDTH = VIEW_WIDTH * TILE_SIZE
SCREEN_HEIGHT = VIEW_HEIGHT * TILE_SIZE

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


def generate_level(width, height, seed=42):
    random.seed(seed)
    level = [[1 for _ in range(width)] for _ in range(height)]
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
        level_int = generate_level(VIEW_WIDTH, VIEW_HEIGHT)
        wall_tex = self.get_texture("stone1")
        for j in range(VIEW_WIDTH):
            for i in range(VIEW_HEIGHT):
                if level_int[i][j] == 1:
                    wall = tex_to_sprite(wall_tex)
                    wall.center_x = j * TILE_SIZE + TILE_SIZE / 2
                    wall.center_y = i * TILE_SIZE + TILE_SIZE / 2
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
