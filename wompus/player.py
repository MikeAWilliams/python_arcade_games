import arcade

SPRITE_PATH = "./assets/player.png"

class Player():
    def __init__(self, width, height):
        self.sprite = arcade.Sprite(SPRITE_PATH)
        self.sprite.width = width
        self.sprite.height = height
        self.sprite.center_x = width / 2
        self.sprite.center_y = height / 2

    def SetIJ(self, i, j):
        self.sprite.center_x = i * self.sprite.width + self.sprite.width / 2
        self.sprite.center_y = j * self.sprite.height + self.sprite.height / 2

    def Draw(self):
        self.sprite.draw()