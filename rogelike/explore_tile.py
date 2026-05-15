import shutil
import subprocess

import arcade
from pathlib import Path


def copy_to_clipboard(text: str) -> bool:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True)
                return True
            except subprocess.SubprocessError:
                continue
    return False

WINDOW_TITLE = "Tile Explorer"

# Urizen tiles are 12x12 — scale them up so they're visible
TILE_SIZE = 12
SHEET_GAP = 1  # padding pixels between tiles in the source sheet
SHEET_STRIDE = TILE_SIZE + SHEET_GAP
DISPLAY_SCALE = 4
CELL = TILE_SIZE * DISPLAY_SCALE  # displayed size of each tile in pixels
STATUS_BAR_H = 28

# Sheet is organized into column groups this wide, with a 1-column separator
# between groups. One content group fills the window; separators are skipped.
COL_GROUP = 25
ROW_GROUP = 16
GROUP_STRIDE = COL_GROUP + 1  # 27 content cols + 1 separator
WINDOW_WIDTH = COL_GROUP * CELL
WINDOW_HEIGHT = ROW_GROUP * CELL

COLS = WINDOW_WIDTH // CELL
ROWS = (WINDOW_HEIGHT - STATUS_BAR_H) // CELL


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
        arcade.set_background_color(arcade.color.BLACK)
        self.sheet_path = sheet_path
        self.sheet: arcade.SpriteSheet | None = None
        self.col_offset = 0
        self.row_offset = 0
        self.sheet_cols = 0
        self.sheet_rows = 0
        self.sprites: arcade.SpriteList = arcade.SpriteList()
        self.hover_col = -1
        self.hover_row = -1
        self.last_copied = ""

    def setup(self):
        self.sheet = arcade.load_spritesheet(self.sheet_path)
        w, h = self.sheet.image.size
        self.sheet_cols = (w + SHEET_GAP) // SHEET_STRIDE
        self.sheet_rows = (h + SHEET_GAP) // SHEET_STRIDE
        print(f"Loaded: {self.sheet_path}  ({self.sheet_cols}x{self.sheet_rows} tiles)")
        self.load_view()

    def load_view(self):
        assert self.sheet is not None
        self.sprites = arcade.SpriteList()
        for grid_row in range(ROWS):
            sheet_row = self.row_offset + grid_row
            if sheet_row >= self.sheet_rows:
                break
            for grid_col in range(COLS):
                sheet_col = self.col_offset + grid_col
                if sheet_col >= self.sheet_cols:
                    break
                tex = self.sheet.get_texture(
                    arcade.LBWH(
                        sheet_col * SHEET_STRIDE + SHEET_GAP,
                        sheet_row * SHEET_STRIDE + SHEET_GAP,
                        TILE_SIZE,
                        TILE_SIZE,
                    )
                )
                sprite = arcade.Sprite(tex, scale=DISPLAY_SCALE)
                sprite.left = grid_col * CELL
                sprite.top = WINDOW_HEIGHT - grid_row * CELL
                self.sprites.append(sprite)

    def on_draw(self):
        self.clear()
        self.sprites.draw()

        if 0 <= self.hover_col < COLS and 0 <= self.hover_row < ROWS:
            cx = self.hover_col * CELL + CELL / 2
            cy = WINDOW_HEIGHT - self.hover_row * CELL - CELL / 2
            arcade.draw_rect_outline(
                arcade.XYWH(cx, cy, CELL, CELL),
                arcade.color.YELLOW,
                2,
            )

        status = (
            f"cols {self.col_offset}-{self.col_offset + COLS - 1}   "
            f"row {self.row_offset}+   "
            f"Left/Right: ±{COL_GROUP} cols   Up/Down: ±{ROWS} rows"
        )
        if 0 <= self.hover_col < COLS and 0 <= self.hover_row < ROWS:
            sc = self.col_offset + self.hover_col
            sr = self.row_offset + self.hover_row
            if sc < self.sheet_cols and sr < self.sheet_rows:
                tile_idx = sr * self.sheet_cols + sc
                status += f"   |   tile {tile_idx}  (col={sc}, row={sr})"
        if self.last_copied:
            status += f"   |   copied: {self.last_copied}"
        arcade.draw_text(status, 6, 6, arcade.color.WHITE, 12)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.RIGHT:
            if self.col_offset + GROUP_STRIDE < self.sheet_cols:
                self.col_offset += GROUP_STRIDE
                self.load_view()
        elif key == arcade.key.LEFT:
            if self.col_offset - GROUP_STRIDE >= 0:
                self.col_offset -= GROUP_STRIDE
                self.load_view()
        elif key == arcade.key.DOWN:
            if self.row_offset + ROWS < self.sheet_rows:
                self.row_offset += ROWS
                self.load_view()
        elif key == arcade.key.UP:
            if self.row_offset - ROWS >= 0:
                self.row_offset -= ROWS
                self.load_view()

    def on_mouse_motion(self, x, y, dx, dy):
        self.hover_col = int(x // CELL)
        self.hover_row = int((WINDOW_HEIGHT - y) // CELL)

    def on_mouse_press(self, x, y, button, modifiers):
        if button != arcade.MOUSE_BUTTON_LEFT:
            return
        if not (0 <= self.hover_col < COLS and 0 <= self.hover_row < ROWS):
            return
        sc = self.col_offset + self.hover_col
        sr = self.row_offset + self.hover_row
        if sc >= self.sheet_cols or sr >= self.sheet_rows:
            return
        px = SHEET_GAP + sc * SHEET_STRIDE
        py = SHEET_GAP + sr * SHEET_STRIDE
        coords = f"({px}, {py}, {TILE_SIZE}, {TILE_SIZE})"
        if copy_to_clipboard(coords):
            self.last_copied = coords
            print(f"copied: {coords}")
        else:
            self.last_copied = f"{coords}  (no clipboard tool)"
            print(f"no clipboard tool found (install wl-copy/xclip/xsel): {coords}")


def main():
    sheet = find_sheet()
    if sheet is None:
        print(
            "No PNG found in assets/ — download the Urizen tileset and place it there."
        )
        return
    window = TileExplorer(sheet)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
