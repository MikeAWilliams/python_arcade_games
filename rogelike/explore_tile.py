import arcade
from pathlib import Path

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Tile Explorer"

# Urizen tiles are 12x12 — scale them up so they're visible
TILE_SIZE = 12
DISPLAY_SCALE = 4
CELL = TILE_SIZE * DISPLAY_SCALE   # displayed size of each tile in pixels
MARGIN = 3
STATUS_BAR_H = 28

COLS = (WINDOW_WIDTH + MARGIN) // (CELL + MARGIN)
ROWS = (WINDOW_HEIGHT - STATUS_BAR_H + MARGIN) // (CELL + MARGIN)
TILES_PER_PAGE = COLS * ROWS


def find_sheet() -> str | None:
    """Return the first PNG found in assets/, or None."""
    assets = Path("assets")
    if not assets.is_dir():
        return None
    pngs = sorted(assets.glob("*.png"))
    return str(pngs[0]) if pngs else None


class TileExplorer(arcade.Window):
    def __init__(self, sheet_path: str):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)
        self.sheet_path = sheet_path
        self.sheet: arcade.SpriteSheet | None = None
        self.page = 0
        self.total_tiles = 0
        self.sheet_cols = 0
        self.sprites: arcade.SpriteList = arcade.SpriteList()
        self.hovered_index = -1

    def setup(self):
        self.sheet = arcade.load_spritesheet(self.sheet_path)
        w, h = self.sheet.image.size
        self.sheet_cols = w // TILE_SIZE
        sheet_rows = h // TILE_SIZE
        self.total_tiles = self.sheet_cols * sheet_rows
        print(f"Loaded: {self.sheet_path}  ({self.sheet_cols}x{sheet_rows} tiles, {self.total_tiles} total)")
        self.load_page()

    def load_page(self):
        assert self.sheet is not None
        self.sprites = arcade.SpriteList()
        start = self.page * TILES_PER_PAGE
        for i in range(TILES_PER_PAGE):
            tile_idx = start + i
            if tile_idx >= self.total_tiles:
                break
            sheet_row = tile_idx // self.sheet_cols
            sheet_col = tile_idx % self.sheet_cols
            tex = self.sheet.get_texture(
                arcade.LBWH(sheet_col * TILE_SIZE, sheet_row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            )
            grid_col = i % COLS
            grid_row = i // COLS
            sprite = arcade.Sprite(tex, scale=DISPLAY_SCALE)
            sprite.left = MARGIN + grid_col * (CELL + MARGIN)
            sprite.top = WINDOW_HEIGHT - STATUS_BAR_H - MARGIN - grid_row * (CELL + MARGIN)
            self.sprites.append(sprite)

    def on_draw(self):
        self.clear()
        self.sprites.draw()

        if 0 <= self.hovered_index < len(self.sprites):
            s = self.sprites[self.hovered_index]
            arcade.draw_rect_outline(
                arcade.XYWH(s.center_x, s.center_y, CELL + 2, CELL + 2),
                arcade.color.YELLOW,
                2,
            )

        total_pages = max(1, (self.total_tiles + TILES_PER_PAGE - 1) // TILES_PER_PAGE)
        status = f"Page {self.page + 1}/{total_pages}   Left/Right arrows to navigate"
        if 0 <= self.hovered_index < len(self.sprites):
            tile_idx = self.page * TILES_PER_PAGE + self.hovered_index
            sc = tile_idx % self.sheet_cols
            sr = tile_idx // self.sheet_cols
            status += f"   |   tile {tile_idx}  (col={sc}, row={sr})"
        arcade.draw_text(status, 6, 6, arcade.color.WHITE, 12)

    def on_key_press(self, key, modifiers):
        total_pages = max(1, (self.total_tiles + TILES_PER_PAGE - 1) // TILES_PER_PAGE)
        if key == arcade.key.RIGHT and self.page < total_pages - 1:
            self.page += 1
            self.load_page()
        elif key == arcade.key.LEFT and self.page > 0:
            self.page -= 1
            self.load_page()

    def on_mouse_motion(self, x, y, dx, dy):
        grid_col = x // (CELL + MARGIN)
        grid_row = (WINDOW_HEIGHT - STATUS_BAR_H - y) // (CELL + MARGIN)
        if grid_row < 0:
            self.hovered_index = -1
            return
        idx = grid_row * COLS + grid_col
        self.hovered_index = idx if 0 <= idx < len(self.sprites) else -1


def main():
    sheet = find_sheet()
    if sheet is None:
        print("No PNG found in assets/ — download the Urizen tileset and place it there.")
        return
    window = TileExplorer(sheet)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
