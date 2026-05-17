"""Show a list of named sprites extracted from the tileset.

Add entries to SPRITES as `name: (x, y, w, h)` in source-pixel coords
(top-left origin in the sheet). For a single tile at sheet column C, row R:
    x = 1 + C * 13,  y = 1 + R * 13,  w = h = 12
For an N-tile-wide sprite include the inter-tile gaps: w = N * 13 - 1.
"""

import arcade

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

DISPLAY_SCALE = 16
GRID_COLS = 4
PAD = 12
LABEL_H = 18
BG_COLOR = arcade.color.WHITE


class SpriteViewer(arcade.Window):
    def __init__(self):
        max_w = max(w for _, _, w, _ in SPRITES.values())
        max_h = max(h for _, _, _, h in SPRITES.values())
        self.cell_w = max_w * DISPLAY_SCALE + PAD * 2
        self.cell_h = max_h * DISPLAY_SCALE + LABEL_H + PAD * 2
        self.cols = min(len(SPRITES), GRID_COLS)
        rows = (len(SPRITES) + self.cols - 1) // self.cols
        self.win_w = self.cols * self.cell_w
        self.win_h = rows * self.cell_h
        super().__init__(self.win_w, self.win_h, "Sprite Viewer")
        arcade.set_background_color(BG_COLOR)
        self.sprites = arcade.SpriteList()
        self.labels: list[tuple[str, float, float]] = []

    def setup(self):
        sheet = arcade.load_spritesheet(SHEET_PATH)
        for idx, (name, (x, y, w, h)) in enumerate(SPRITES.items()):
            tex = sheet.get_texture(arcade.LBWH(x, y, w, h))
            sprite = arcade.Sprite(tex, scale=DISPLAY_SCALE)
            col = idx % self.cols
            row = idx // self.cols
            sprite.center_x = col * self.cell_w + self.cell_w / 2
            sprite.center_y = (
                self.win_h - row * self.cell_h - PAD - (h * DISPLAY_SCALE) / 2
            )
            self.sprites.append(sprite)
            self.labels.append(
                (
                    f"{name}  ({x},{y},{w},{h})",
                    col * self.cell_w + PAD,
                    self.win_h - (row + 1) * self.cell_h + PAD,
                )
            )

    def on_draw(self):
        self.clear()
        self.sprites.draw()
        for text, lx, ly in self.labels:
            arcade.draw_text(text, lx, ly, arcade.color.WHITE, 10)


def main():
    win = SpriteViewer()
    win.setup()
    arcade.run()


if __name__ == "__main__":
    main()
