import arcade

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Gravity Game"

OCEAN_WIDTH = 427
LAND_WIDTH = 426
EARTH_HEIGHT = 100

arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
arcade.set_background_color(arcade.color.WHITE)
arcade.start_render()

#arcade.draw_circle_filled( SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 150, arcade.color.BLUE)
#arcade.draw_text("draw_filled_rect", 363, 3, arcade.color.BLACK, 10)
# draw the ground as two blue rectangles with a green in the  middle
earth_x_cursor = 0
arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
earth_x_cursor += OCEAN_WIDTH
arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, LAND_WIDTH, EARTH_HEIGHT, arcade.color.GREEN)
earth_x_cursor += LAND_WIDTH
arcade.draw_xywh_rectangle_filled(earth_x_cursor, 0, OCEAN_WIDTH, EARTH_HEIGHT, arcade.color.BLUE)
arcade.finish_render()

arcade.run()