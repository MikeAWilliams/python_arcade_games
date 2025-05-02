# A guide I used https://lodev.org/cgtutor/raycasting.html
# using Digital Differential Analysis (DDA) algorithm
import arcade
import numpy as np
from PIL import Image
import arcade

print("arcade version", arcade.__version__)

# this is sooooooo slow that I had to drop the resolution to 300x300
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
MAP_WIDTH = 24
MAP_HEIGHT = 24
TEXTURE_WIDTH = 64
TEXTURE_HEIGHT = 64

# Player position and direction
initial_pos = arcade.Vec2(22, 12)
initial_facing_dir = arcade.Vec2(-1, 0)
initial_view_plane = arcade.Vec2(0, 0.66)

WORLD_MAP = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 4, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 0, 7, 7, 7, 7, 7],
    [4, 0, 5, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 7, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 6, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 8],
    [4, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 8, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 8],
    [4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 6, 0, 6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 5, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2],
    [4, 0, 6, 0, 6, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2],
    [4, 0, 0, 5, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2],
    [4, 0, 6, 0, 6, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 5, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
]


class Raycaster(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Raycaster")
        self.position = initial_pos
        self.facing_dir = initial_facing_dir
        self.view_plane = initial_view_plane
        self.keys = set()
        self.init_single_sprite_as_buffer()
        self.load_textures()

    def init_single_sprite_as_buffer(self):
        self.pixel_buffer = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )  # RGB format
        self.sprite_list = arcade.SpriteList()  # Create a SpriteList to manage sprites

        # Create an image from the buffer
        image = Image.frombuffer(
            "RGB", (SCREEN_HEIGHT, SCREEN_WIDTH), self.pixel_buffer.tobytes()
        ).convert("RGBA")
        sprite = arcade.Sprite(
            path_or_texture=arcade.Texture(name="pixel_buffer", image=image),
            center_x=SCREEN_WIDTH // 2,
            center_y=SCREEN_HEIGHT // 2,
        )
        sprite.width = SCREEN_WIDTH
        sprite.height = SCREEN_HEIGHT
        self.sprite_list.append(sprite)

    def load_textures(self):
        self.images = []
        # you can downlaod the textures from https://lodev.org/cgtutor/raycasting.html
        self.images.append(Image.open("pics/eagle.png"))
        self.images.append(Image.open("./pics/redbrick.png"))
        self.images.append(Image.open("./pics/purplestone.png"))
        self.images.append(Image.open("./pics/greystone.png"))
        self.images.append(Image.open("./pics/bluestone.png"))
        self.images.append(Image.open("./pics/mossy.png"))
        self.images.append(Image.open("./pics/wood.png"))
        self.images.append(Image.open("./pics/colorstone.png"))

    def on_key_press(self, key, modifiers):
        self.keys.add(key)

    def on_key_release(self, key, modifiers):
        self.keys.discard(key)

    def on_draw(self):
        self.clear()
        self.render()
        self.update_texture()
        self.sprite_list.draw()

    def update_texture(self):
        # Create an image from the buffer
        image = Image.frombuffer(
            "RGB", (SCREEN_HEIGHT, SCREEN_WIDTH), self.pixel_buffer.tobytes()
        ).convert("RGBA")
        self.sprite_list[0].texture = arcade.Texture(name="pixel_buffer", image=image)

    def render(self):
        # paint it black
        self.pixel_buffer[:, :] = [0, 0, 0]
        # draw one vertical line for each column of pixels
        for x in range(SCREEN_WIDTH):
            # Calculate ray position and direction
            camera_x = 2 * x / SCREEN_WIDTH - 1
            ray_dir = self.facing_dir + self.view_plane * camera_x

            # Integer map position for DDA
            map_x = int(self.position.x)
            map_y = int(self.position.y)

            # Length of ray from current position to next x or y side
            side_dist_x = 0
            side_dist_y = 0

            # Length of ray from one x or y side to next x or y side
            delta_dist_x = abs(1 / ray_dir.x) if ray_dir.x != 0 else float("inf")
            delta_dist_y = abs(1 / ray_dir.y) if ray_dir.y != 0 else float("inf")
            perp_wall_dist = 0

            # Step and initial sideDist
            step_x, step_y = 0, 0
            if ray_dir.x < 0:
                step_x = -1
                side_dist_x = (self.position.x - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - self.position.x) * delta_dist_x
            if ray_dir.y < 0:
                step_y = -1
                side_dist_y = (self.position.y - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - self.position.y) * delta_dist_y

            # Perform DDA
            # find the first intersection with a wall
            last_step_was_x_side = False
            while True:
                # Jump to next map square
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    last_step_was_x_side = True
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    last_step_was_x_side = False
                # Check if ray has hit a wall
                # note this will infinite loop if the ray escapes the world map without hitting a wall
                if WORLD_MAP[map_x][map_y] > 0:
                    break

            # Calculate distance to the wall
            if last_step_was_x_side:
                perp_wall_dist = (
                    map_x - self.position.x + (1 - step_x) / 2
                ) / ray_dir.x
            else:
                perp_wall_dist = (
                    map_y - self.position.y + (1 - step_y) / 2
                ) / ray_dir.y

            if perp_wall_dist <= 0:
                perp_wall_dist = 0.1

            # Calculate height of line to draw
            line_height = int(SCREEN_HEIGHT / perp_wall_dist)

            # Calculate lowest and highest pixel to fill in current stripe
            draw_start = -line_height // 2 + SCREEN_HEIGHT // 2
            draw_end = line_height // 2 + SCREEN_HEIGHT // 2

            # Clamp values
            draw_start = max(0, draw_start)
            draw_end = min(SCREEN_HEIGHT - 1, draw_end)

            wall_x = 0
            if last_step_was_x_side:
                wall_x = self.position.y + perp_wall_dist * ray_dir.y
            else:
                wall_x = self.position.x + perp_wall_dist * ray_dir.x
            wall_x -= int(wall_x)

            texture_x = int(wall_x * TEXTURE_WIDTH)
            if last_step_was_x_side and ray_dir.x > 0:
                texture_x = TEXTURE_WIDTH - texture_x - 1
            if not last_step_was_x_side and ray_dir.y < 0:
                texture_x = TEXTURE_WIDTH - texture_x - 1

            step = TEXTURE_HEIGHT / line_height

            texture_num = WORLD_MAP[map_x][map_y] - 1  # 0-indexed

            texPos = (draw_start - SCREEN_HEIGHT / 2 + line_height / 2) * step
            for y in range(draw_start, draw_end):
                texY = int(texPos) & (TEXTURE_HEIGHT - 1)
                texPos += step
                color = self.images[texture_num].getpixel((texture_x, texY))

                if not last_step_was_x_side:
                    color = tuple(c // 2 for c in color)
                self.pixel_buffer[y, x] = color

    def on_update(self, delta_time):
        move_speed = delta_time * 5.0
        rot_speed = delta_time * 3.0

        # forward and backward movement
        # check x and y seperately to allow for player to slide allong the wall
        if arcade.key.UP in self.keys:
            new_position = self.position + self.facing_dir * move_speed
            if WORLD_MAP[int(new_position.x)][int(self.position.y)] == 0:
                self.position = arcade.Vec2(new_position.x, self.position.y)
            if WORLD_MAP[int(self.position.x)][int(new_position.y)] == 0:
                self.position = arcade.Vec2(self.position.x, new_position.y)
        if arcade.key.DOWN in self.keys:
            new_position = self.position - self.facing_dir * move_speed
            if WORLD_MAP[int(new_position.x)][int(self.position.y)] == 0:
                self.position = arcade.Vec2(new_position.x, self.position.y)
            if WORLD_MAP[int(self.position.x)][int(new_position.y)] == 0:
                self.position = arcade.Vec2(self.position.x, new_position.y)

        # rotation
        if arcade.key.RIGHT in self.keys:
            self.facing_dir = self.facing_dir.rotate(-rot_speed)
            self.view_plane = self.view_plane.rotate(-rot_speed)
        if arcade.key.LEFT in self.keys:
            self.facing_dir = self.facing_dir.rotate(rot_speed)
            self.view_plane = self.view_plane.rotate(rot_speed)


if __name__ == "__main__":
    window = Raycaster()
    arcade.run()
