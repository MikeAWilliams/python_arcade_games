import arcade

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Game"
DISPLAY_SCALE = 4

SHEET_PATH = "assets/urizen_onebit_tileset__v2d0.png"

SPRITES: dict[str, tuple[int, int, int, int]] = {
    "stone1": (1, 27, 12, 12),
    "stone2": (14, 27, 12, 12),
    "stonecorner1": (66, 27, 12, 12),
    "stonecorner2": (79, 27, 12, 12),
    "stonecorner3": (92, 27, 12, 12),
    "floor1": (1, 92, 12, 12),
    "floor2": (14, 92, 12, 12),
    "floor3": (27, 92, 12, 12),
}


class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BLACK)
        self.sheet = arcade.load_spritesheet(SHEET_PATH)
        self.sprites = arcade.SpriteList()
        self.sprites.append(self.get_sprite("stone1"))

    def setup(self):
        pass

    def get_sprite(self, name: str) -> arcade.Texture:
        x, y, w, h = SPRITES[name]
        tex = self.sheet.get_texture(arcade.LBWH(x, y, w, h))
        return arcade.Sprite(tex, scale=DISPLAY_SCALE)

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
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
