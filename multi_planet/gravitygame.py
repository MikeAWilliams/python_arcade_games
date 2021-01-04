import game
import arcade

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

def main():
    gameObject = game.GravityGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    gameObject.setup()
    arcade.run()


if __name__ == "__main__":
    main()