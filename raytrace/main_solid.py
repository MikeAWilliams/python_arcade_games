# A guide I used https://lodev.org/cgtutor/raycasting.html
# using Digital Differential Analysis (DDA) algorithm
import math
import arcade

print("arcade version", arcade.__version__)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1200
MAP_WIDTH = 24
MAP_HEIGHT = 24

# Player position and direction
initial_pos = arcade.Vec2(22, 12)
initial_facing_dir = arcade.Vec2(-1, 0)
initial_view_plane = arcade.Vec2(0, 0.66)

# Colors for walls
COLORS = {
    1: arcade.color.RED,
    2: arcade.color.GREEN,
    3: arcade.color.BLUE,
    4: arcade.color.WHITE,
    5: arcade.color.YELLOW,
}

WORLD_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


class Raycaster(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Raycaster")
        self.position = initial_pos
        self.facing_dir = initial_facing_dir
        self.view_plane = initial_view_plane
        self.keys = set()

    def on_key_press(self, key, modifiers):
        self.keys.add(key)

    def on_key_release(self, key, modifiers):
        self.keys.discard(key)

    def on_draw(self):
        self.clear()

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

            # Choose wall color
            color = COLORS.get(WORLD_MAP[map_x][map_y], arcade.color.YELLOW)
            if not last_step_was_x_side:
                # Darken color for x-side
                # recall that color is a tuple of RGB values
                color = tuple(c // 2 for c in color)

            # Draw the vertical line
            arcade.draw_line(x, draw_start, x, draw_end, color)

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
