import arcade
import numpy as np
from PIL import Image
import random

SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 1200


class MyWindow(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.BLACK)
        self.pixel_buffer = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )  # RGB format
        self.sprite_list = arcade.SpriteList()  # Create a SpriteList to manage sprites
        self.generate_data()
        size = (SCREEN_HEIGHT, SCREEN_WIDTH)  # Image dimensions: width, height

        data = self.pixel_buffer.tobytes()

        # Create an image from the buffer
        image = Image.frombuffer("RGB", size, data).convert("RGBA")
        sprite = arcade.Sprite(
            path_or_texture=arcade.Texture(name="pixel_buffer", image=image),
            center_x=SCREEN_WIDTH // 2,
            center_y=SCREEN_HEIGHT // 2,
        )
        sprite.width = SCREEN_WIDTH
        sprite.height = SCREEN_HEIGHT
        self.sprite_list.append(sprite)

    def update_texture(self):
        size = (SCREEN_HEIGHT, SCREEN_WIDTH)  # Image dimensions: width, height

        data = self.pixel_buffer.tobytes()

        # Create an image from the buffer
        image = Image.frombuffer("RGB", size, data).convert("RGBA")
        self.sprite_list[0].texture = arcade.Texture(name="pixel_buffer", image=image)

    def on_draw(self):
        self.clear()
        self.generate_data()
        self.update_texture()
        # Draw all sprites in the SpriteList
        print("frame")
        self.sprite_list.draw()

    # simulate a render loop to update the buffer
    def generate_data(self):
        # this is faster but only static
        # Create a random binary mask for the entire buffer
        # mask = np.random.rand(SCREEN_HEIGHT, SCREEN_WIDTH) < 0.5
        ## Set black and white colors
        # black = np.array([0, 0, 0], dtype=np.uint8)
        # white = np.array([255, 255, 255], dtype=np.uint8)
        ## Use the mask to assign black or white to the entire buffer at once
        # self.pixel_buffer[mask] = white
        # self.pixel_buffer[~mask] = black

        # this is really slow
        # black = (0, 0, 0)
        # white = (255, 255, 255)
        # for x in range(SCREEN_WIDTH):
        # for y in range(SCREEN_HEIGHT):
        # color = black
        # if random.random() < 0.5:
        # color = white
        # self.pixel_buffer[x, y, 0] = color[0]
        # self.pixel_buffer[x, y, 1] = color[1]
        # self.pixel_buffer[x, y, 2] = color[2]

        # simulate a render loop
        self.pixel_buffer[:, :] = [0, 0, 0]
        c1 = [255, 0, 255]
        c2 = [0, 255, 255]
        c3 = [255, 255, 0]
        color = c1
        y0 = 200
        y1 = 500
        y2 = 700
        y_start = 0
        y_end = 0
        for x in range(SCREEN_WIDTH):
            if x > 200 and x < 500:
                color = c1
                y_start = 0
                y_end = y0
            elif x > 700 and x < 1000:
                color = c2
                y_start = y0
                y_end = y1
            elif x > 1100:
                color = c3
                y_start = y1
                y_end = y2
            self.pixel_buffer[y_start:y_end, x] = color


if __name__ == "__main__":
    window = MyWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Pixel Buffer Example")
    arcade.run()
