import arcade

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Roguelike"


class GameWindow(arcade.Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        self.clear()


def main():
    window = GameWindow()
    arcade.run()


if __name__ == "__main__":
    main()
