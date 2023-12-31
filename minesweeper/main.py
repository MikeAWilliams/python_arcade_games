import arcade
from enum import Enum

SCREEN_TITLE = "Mine Sweeper"
GRID_COLUMNS = 14
GRID_ROWS = 14
SPRITE_SHEET_PATH = "./sprites.png"
SPRITE_NATIVE_SIZE = 16
SPRITE_SCALE = 4
SPRITE_FINAL_SIZE = SPRITE_SCALE * SPRITE_NATIVE_SIZE
SPRITE_HALF_SIZE = SPRITE_FINAL_SIZE / 2
SCREEN_WIDTH = GRID_COLUMNS * SPRITE_FINAL_SIZE
SCREEN_HEIGHT = GRID_ROWS * SPRITE_FINAL_SIZE

class Square:
    def __init__(self, x, y):
        # maybe this is an empty square
        self.sprite = arcade.Sprite(SPRITE_SHEET_PATH, SPRITE_SCALE, 0, 50, 16, 16)
        self.sprite.center_x = x
        self.sprite.center_y = y

    def Draw(self):
        self.sprite.draw()
    

class MinesweeperGame(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.WHITE)
        self.squares = []
        for i in range(GRID_ROWS):
            x = i * SPRITE_FINAL_SIZE + SPRITE_HALF_SIZE
            for j in range(GRID_COLUMNS):
                y = j * SPRITE_FINAL_SIZE + SPRITE_HALF_SIZE
                self.squares.append(Square(x,y))


    def on_draw(self):
        arcade.start_render()
        for s in self.squares:
            s.Draw()


    def on_key_press(self, symbol, modifiers):
        pass

    def on_key_release(self, symbol, modifiers):
        pass

    def on_update(self, delta_time: float):
        pass

    def on_update(self, delta_time: float):
        pass

def main():
    gameObject = MinesweeperGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    arcade.run()


if __name__ == "__main__":
    main()