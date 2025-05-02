# A guide I used https://lodev.org/cgtutor/raycasting.html
# using Digital Differential Analysis (DDA) algorithm
import arcade
import numpy as np
from PIL import Image
import arcade

print("arcade version", arcade.__version__)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1200
MAP_WIDTH = 24
MAP_HEIGHT = 24
TEXTURE_WIDTH = 64
TEXTURE_HEIGHT = 64

# Player position and direction
initial_pos = arcade.Vec2(20, 12)
initial_facing_dir = arcade.Vec2(-1, 0)
initial_view_plane = arcade.Vec2(0, 0.66)

# Convert WORLD_MAP to a NumPy array
WORLD_MAP = np.array(
    [
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
)


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
        self.images.append(np.array(Image.open("pics/eagle.png")))
        self.images.append(np.array(Image.open("./pics/redbrick.png")))
        self.images.append(np.array(Image.open("./pics/purplestone.png")))
        self.images.append(np.array(Image.open("./pics/greystone.png")))
        self.images.append(np.array(Image.open("./pics/bluestone.png")))
        self.images.append(np.array(Image.open("./pics/mossy.png")))
        self.images.append(np.array(Image.open("./pics/wood.png")))
        self.images.append(np.array(Image.open("./pics/colorstone.png")))

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
        # Paint the screen black
        self.pixel_buffer[:, :] = [0, 0, 0]

        # Calculate ray directions for all columns
        camera_x = np.linspace(-1, 1, SCREEN_WIDTH, dtype=np.float32)
        ray_dirs = np.array(
            [
                self.facing_dir.x + self.view_plane.x * camera_x,
                self.facing_dir.y + self.view_plane.y * camera_x,
            ]
        )

        # Integer map positions for DDA (one value per ray)
        map_x = np.floor(self.position.x + np.zeros(SCREEN_WIDTH)).astype(np.int32)
        map_y = np.floor(self.position.y + np.zeros(SCREEN_WIDTH)).astype(np.int32)

        # DDA step calculations
        delta_dist_x = np.abs(1 / ray_dirs[0])
        delta_dist_y = np.abs(1 / ray_dirs[1])

        step_x = np.where(ray_dirs[0] < 0, -1, 1)
        step_y = np.where(ray_dirs[1] < 0, -1, 1)

        side_dist_x = np.where(
            ray_dirs[0] < 0,
            (self.position.x - map_x) * delta_dist_x,
            (map_x + 1.0 - self.position.x) * delta_dist_x,
        )

        side_dist_y = np.where(
            ray_dirs[1] < 0,
            (self.position.y - map_y) * delta_dist_y,
            (map_y + 1.0 - self.position.y) * delta_dist_y,
        )

        # Perform DDA for all rays
        hit = np.zeros(SCREEN_WIDTH, dtype=bool)
        side = np.zeros(SCREEN_WIDTH, dtype=bool)

        while not np.all(hit):
            mask = (side_dist_x < side_dist_y) & ~hit
            side_dist_x[mask] += delta_dist_x[mask]
            map_x[mask] += step_x[mask]
            side[mask] = True

            mask = ~mask & ~hit
            side_dist_y[mask] += delta_dist_y[mask]
            map_y[mask] += step_y[mask]
            side[mask] = False

            hit = WORLD_MAP[map_x, map_y] > 0

        # Calculate perpendicular wall distances
        perp_wall_dist = np.where(
            side,
            (map_x - self.position.x + (1 - step_x) / 2) / ray_dirs[0],
            (map_y - self.position.y + (1 - step_y) / 2) / ray_dirs[1],
        )

        # Calculate line heights and draw ranges
        line_heights = (SCREEN_HEIGHT / perp_wall_dist).astype(np.int32)
        draw_starts = np.clip(-line_heights // 2 + SCREEN_HEIGHT // 2, 0, SCREEN_HEIGHT)
        draw_ends = np.clip(line_heights // 2 + SCREEN_HEIGHT // 2, 0, SCREEN_HEIGHT)

        # Texture calculations
        wall_x = np.where(
            side,
            self.position.y + perp_wall_dist * ray_dirs[1],
            self.position.x + perp_wall_dist * ray_dirs[0],
        )
        wall_x -= np.floor(wall_x)

        texture_x = (wall_x * TEXTURE_WIDTH).astype(np.int32)
        texture_x = np.where(
            (side & (ray_dirs[0] > 0)) | (~side & (ray_dirs[1] < 0)),
            TEXTURE_WIDTH - texture_x - 1,
            texture_x,
        )

        # Draw vertical stripes for all columns
        for x in range(SCREEN_WIDTH):
            texture_num = WORLD_MAP[map_x[x], map_y[x]] - 1
            step = TEXTURE_HEIGHT / line_heights[x]
            texY = (
                (
                    np.arange(draw_starts[x], draw_ends[x])
                    - SCREEN_HEIGHT / 2
                    + line_heights[x] / 2
                )
                * step
            ).astype(np.int32) & (TEXTURE_HEIGHT - 1)
            stripe = self.images[texture_num][texY, texture_x[x]]

            if not side[x]:
                stripe = stripe // 2  # Darken the color for y-side walls

            self.pixel_buffer[draw_starts[x] : draw_ends[x], x] = stripe

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
