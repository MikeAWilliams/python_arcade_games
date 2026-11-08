import arcade

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class GameView(arcade.View):
    def __init__(self) -> None:
        super().__init__()

    def draw_sample_strip(self):
        point_list = (
            (510, 450),
            (570, 450),
            (510, 480),
            (570, 480),
            (510, 510),
            (570, 510),
        )
        arcade.draw_line_strip(point_list, arcade.color.TROPICAL_RAIN_FOREST, 3)

    def on_draw(self) -> bool | None:
        self.draw_sample_strip()

    def on_update(self, delta_time: float) -> bool | None:
        pass


def main():
    window = arcade.Window(
        WINDOW_WIDTH, WINDOW_HEIGHT, "Show Me A Sound", resizable=True
    )
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
