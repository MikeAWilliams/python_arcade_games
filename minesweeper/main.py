import arcade
from enum import Enum
import math

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

class SquareImage(Enum):
    BLANK_UP   = 1
    BLANK_DOWN = 2
    FLAG = 3
    QUESTION_UP = 4
    QUESTION_DOWN = 5
    MINE_GREY = 6
    MINE_RED = 7
    MINE_X = 8
    ONE = 9
    TWO = 10
    THREE = 11
    FOUR = 12
    FIVE = 13
    SIX = 14
    SEVEN = 15
    EIGHT = 16

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

SHEET_OFFSETS = {
    SquareImage.BLANK_UP: Coordinate(0,50),
    SquareImage.BLANK_DOWN: Coordinate(16,50),
    SquareImage.FLAG: Coordinate(33,50),
    SquareImage.QUESTION_UP: Coordinate(50,50),
    SquareImage.QUESTION_DOWN: Coordinate(67,50),
    SquareImage.MINE_GREY: Coordinate(84,50),
    SquareImage.MINE_RED: Coordinate(101,50),
    SquareImage.MINE_X: Coordinate(118,50),
    SquareImage.ONE: Coordinate(0,68),
    SquareImage.TWO: Coordinate(16,68),
    SquareImage.THREE: Coordinate(33,68),
    SquareImage.FOUR: Coordinate(50,68),
    SquareImage.FIVE: Coordinate(67,68),
    SquareImage.SIX: Coordinate(84,68),
    SquareImage.SEVEN: Coordinate(101,68),
    SquareImage.EIGHT: Coordinate(118,68),
}

class Square:
    def __init__(self, x, y):
        offset = SHEET_OFFSETS[SquareImage.BLANK_UP]
        self.sprite = arcade.Sprite(SPRITE_SHEET_PATH, SPRITE_SCALE, offset.x, offset.y, SPRITE_NATIVE_SIZE, SPRITE_NATIVE_SIZE)
        self.x = x
        self.y = y
        self.sprite.center_x = x
        self.sprite.center_y = y
    
    def DrawAs(self, image):
        offset = SHEET_OFFSETS[image]
        self.sprite = arcade.Sprite(SPRITE_SHEET_PATH, SPRITE_SCALE, offset.x, offset.y, SPRITE_NATIVE_SIZE, SPRITE_NATIVE_SIZE)
        self.sprite.center_x = self.x
        self.sprite.center_y = self.y


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
        
        # this is just for using the mouse to change the imatge, delete this later
        self.last_image = 1

    def GetMineIndex(self, x, y):
        i = math.floor(x / SPRITE_FINAL_SIZE)
        j = math.floor(y / SPRITE_FINAL_SIZE)
        index = i * GRID_COLUMNS + j
        #print("i {0}, j {1}, index {2}".format(i, j, index))
        return index


    def on_mouse_press(self, x, y, button, key_modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            index = self.GetMineIndex(x,y)
            # just for PoC of different drawing remove this later
            # cylcle the image to draw
            self.last_image = self.last_image + 1
            if self.last_image > 16:
                self.last_image = 1
            self.squares[index].DrawAs(SquareImage(self.last_image))

            

    def on_draw(self):
        arcade.start_render()
        for s in self.squares:
            s.Draw()


def main():
    gameObject = MinesweeperGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    arcade.run()


if __name__ == "__main__":
    main()