import arcade

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = "Platformer"

CHARACTER_SCALING = 2.0 
TILE_SCALING = 2.0
COIN_SCALING = 2.0

PLAYER_MOVEMENT_SPEED = 5
PLAYER_JUMP_SPEED = 15
GRAVITY_CONSTANT = 0.5

LEFT_VIEWPORT_MARGIN = 250
RIGHT_VIEWPORT_MARGIN = 250
BOTTOM_VIEWPORT_MARGIN = 50
TOP_VIEWPORT_MARGIN = 100

MAXIMUM_MAP_NUMBER = 3

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.world_list = None
        self.player_list = None
        self.coin_list = None
        self.hazard_list = None

        self.player_sprite = None

        self.physics_engine = None

        self.view_bottom = 0
        self.view_left = 0

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

    def setup(self, map_number = 1):
        self.map_number = map_number
        self.player_list = arcade.SpriteList()
        self.world_list = arcade.SpriteList(use_spatial_hash=True)
        self.hazard_list = arcade.SpriteList(use_spatial_hash=True)
        self.coin_list = arcade.SpriteList()

        image_source = "./player.png"
        self.player_sprite = arcade.Sprite(image_source, CHARACTER_SCALING)
        self.player_sprite.center_x = 64
        self.player_sprite.center_y = 128
        self.player_list.append(self.player_sprite)
        self.player_sprite_right_texture = arcade.load_texture(image_source)
        self.player_sprite_left_texture = arcade.load_texture(image_source, flipped_horizontally=True)

        my_map = arcade.tilemap.read_tmx("./map"+str(map_number)+".tmx")
        self.world_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='ground',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        self.coin_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='coins',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        self.hazard_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='hazards',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        player_location = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='player',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        self.player_sprite.left = player_location[0].left
        self.player_sprite.bottom = player_location[0].bottom

        self.game_over = False
        self.player_won = False
        self.view_bottom = 0
        self.view_left = 0

        self.physics_engine = arcade.PhysicsEnginePlatformer(self.player_sprite, self.world_list, GRAVITY_CONSTANT)
    	

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.close_window()
        if key == arcade.key.N:
            self.setup()

        if self.game_over:
            return

        if key == arcade.key.W and self.physics_engine.can_jump():
            self.player_sprite.change_y = PLAYER_JUMP_SPEED
        elif key == arcade.key.A:
            self.player_sprite.change_x = -PLAYER_MOVEMENT_SPEED
            self.player_sprite.texture = self.player_sprite_left_texture
        elif key == arcade.key.D:
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED
            self.player_sprite.texture = self.player_sprite_right_texture
    
    def on_key_release(self, key, modifiers):
        if key == arcade.key.A:
            self.player_sprite.change_x = 0
        elif key == arcade.key.D:
            self.player_sprite.change_x = 0

    def on_update(self, delta_time):
        if self.game_over:
            return
        self.physics_engine.update()

        coin_hit_list = arcade.check_for_collision_with_list(self.player_sprite, self.coin_list)
        for coin in coin_hit_list:
            coin.remove_from_sprite_lists()
        if 0 == len(self.coin_list):
            if self.map_number < MAXIMUM_MAP_NUMBER:
                self.setup(self.map_number + 1)
            else:
                self.game_over = True
                self.player_won = True

        harard_hits = arcade.check_for_collision_with_list(self.player_sprite, self.hazard_list)
        if len(harard_hits) > 0:
            self.setup(self.map_number)

        self.update_viewport()
    
    def update_viewport(self):
        changed = False

        left_boundary = self.view_left + LEFT_VIEWPORT_MARGIN
        if self.player_sprite.left < left_boundary:
            self.view_left -= left_boundary - self.player_sprite.left
            changed = True

        right_boundary = self.view_left + SCREEN_WIDTH - RIGHT_VIEWPORT_MARGIN
        if self.player_sprite.right > right_boundary:
            self.view_left += self.player_sprite.right - right_boundary
            changed = True

        top_boundary = self.view_bottom + SCREEN_HEIGHT - TOP_VIEWPORT_MARGIN
        if self.player_sprite.top > top_boundary:
            self.view_bottom += self.player_sprite.top - top_boundary
            changed = True

        bottom_boundary = self.view_bottom + BOTTOM_VIEWPORT_MARGIN
        if self.player_sprite.bottom < bottom_boundary:
            self.view_bottom -= bottom_boundary - self.player_sprite.bottom
            changed = True

        if changed:
            self.view_bottom = int(self.view_bottom)
            self.view_left = int(self.view_left)

            arcade.set_viewport(self.view_left,
                                SCREEN_WIDTH + self.view_left,
                                self.view_bottom,
                                SCREEN_HEIGHT + self.view_bottom)    



    def on_draw(self):
        arcade.start_render()
        self.world_list.draw()
        self.hazard_list.draw()
        self.player_list.draw()
        self.coin_list.draw()

        score_text = f"Coins Remaining: {len(self.coin_list)}"
        arcade.draw_text(score_text, self.view_left + 10, self.view_bottom + 10, arcade.csscolor.WHITE, 18)

        if self.game_over:
            if self.player_won:
                arcade.draw_text("YOU WON!", self.view_left + 10, self.view_bottom + 30, arcade.csscolor.WHITE, 18)
            else:
                arcade.draw_text("GAME OVER", self.view_left + 10, self.view_bottom + 30, arcade.csscolor.WHITE, 18)



def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
