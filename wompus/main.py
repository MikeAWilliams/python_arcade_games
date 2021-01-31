import game
import arcade
import random

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

def main():
    gameObject = game.WompusGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    gameObject.setup()
    arcade.run()


if __name__ == "__main__":
    main()