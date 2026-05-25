"""Animate a sprite from a horizontal sprite sheet.

Right arrow cycles between animation sheets (Death, Idle, Run).
"""

import arcade

BASE = (
    "assets/Pixel Crawler - Free Pack/Entities/Mobs/"
    "Orc Crew/Orc"
)

SHEETS = [
    {"name": "Death", "path": f"{BASE}/Death/Death-Sheet.png", "cell_w": 96, "frame_w": 32, "frame_h": 64, "x_offset": 32},
    {"name": "Idle",  "path": f"{BASE}/Idle/Idle-Sheet.png",   "cell_w": 32, "frame_w": 32, "frame_h": 32, "x_offset": 0},
    {"name": "Run",   "path": f"{BASE}/Run/Run-Sheet.png",     "cell_w": 96, "frame_w": 32, "frame_h": 64, "x_offset": 32},
]

SCALE = 4
FPS = 5
WIN_W = 400
WIN_H = 400


class AnimationViewer(arcade.Window):
    def __init__(self):
        super().__init__(WIN_W, WIN_H, "Animation Viewer")
        arcade.set_background_color(arcade.color.DARK_SLATE_GRAY)
        self.all_textures: list[list[arcade.Texture]] = []
        self.sheet_idx = 0
        self.sprite: arcade.Sprite | None = None
        self.sprite_list = arcade.SpriteList()
        self.frame = 0
        self.elapsed = 0.0

    def setup(self):
        for info in SHEETS:
            sheet = arcade.load_spritesheet(info["path"])
            img_w = arcade.load_texture(info["path"]).width
            count = img_w // info["cell_w"]
            textures = []
            for i in range(count):
                tex = sheet.get_texture(arcade.LBWH(
                    i * info["cell_w"] + info["x_offset"], 0,
                    info["frame_w"], info["frame_h"],
                ))
                textures.append(tex)
            self.all_textures.append(textures)

        self.sprite = arcade.Sprite(self.all_textures[0][0], scale=SCALE)
        self.sprite.center_x = WIN_W / 2
        self.sprite.center_y = WIN_H / 2 + 20
        self.sprite_list.append(self.sprite)

    def _switch_sheet(self, idx: int):
        self.sheet_idx = idx
        self.frame = 0
        self.elapsed = 0.0
        self.sprite.texture = self.all_textures[idx][0]

    def on_key_press(self, key: int, modifiers: int):
        if key == arcade.key.RIGHT:
            self._switch_sheet((self.sheet_idx + 1) % len(SHEETS))

    def on_update(self, delta_time: float):
        self.elapsed += delta_time
        textures = self.all_textures[self.sheet_idx]
        if self.elapsed >= 1.0 / FPS:
            self.elapsed -= 1.0 / FPS
            self.frame = (self.frame + 1) % len(textures)
            self.sprite.texture = textures[self.frame]

    def on_draw(self):
        self.clear()
        self.sprite_list.draw(pixelated=True)
        name = SHEETS[self.sheet_idx]["name"]
        textures = self.all_textures[self.sheet_idx]
        arcade.draw_text(
            f"{name}  Frame: {self.frame}/{len(textures)}",
            WIN_W / 2, 20,
            arcade.color.WHITE, 16,
            anchor_x="center",
        )


def main():
    win = AnimationViewer()
    win.setup()
    arcade.run()


if __name__ == "__main__":
    main()
