import arcade

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = "Platformer"

CHARACTER_SCALING = 2.0 
TILE_SCALING = 2.0
COIN_SCALING = 2.0

PLAYER_MOVEMENT_SPEED = 5
PLAYER_JUMP_SPEED = 10
GRAVITY_CONSTANT = 0.5

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.world_list = None
        self.player_list = None
        #self.coin_list = None
        #self.hazard_list = None

        self.player_sprite = None

        self.physics_engine = None

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

    def setup(self):
        self.player_list = arcade.SpriteList()
        self.world_list = arcade.SpriteList(use_spatial_hash=True)
        #self.hazard_list = arcade.SpriteList(use_spatial_hash=True)
        self.coin_list = arcade.SpriteList()

        image_source = "./player.png"
        self.player_sprite = arcade.Sprite(image_source, CHARACTER_SCALING)
        self.player_sprite.center_x = 64
        self.player_sprite.center_y = 128
        self.player_list.append(self.player_sprite)

        my_map = arcade.tilemap.read_tmx("./map1.tmx")
        self.world_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='ground',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        self.coin_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='coins',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        player_location = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name='player',
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)
        self.player_sprite.left = player_location[0].left
        self.player_sprite.bottom = player_location[0].bottom

        self.score = 0
        self.game_over = False

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
        elif key == arcade.key.D:
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED
    
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
            self.score += 1

        #harard_hits = arcade.check_for_collision_with_list(self.player_sprite, self.hazard_list)
        #if len(harard_hits) > 0:
            #self.game_over = True


    def on_draw(self):
        arcade.start_render()
        self.world_list.draw()
        #self.hazard_list.draw()
        self.player_list.draw()
        self.coin_list.draw()

        score_text = f"Score: {self.score}"
        arcade.draw_text(score_text, 10, 10, arcade.csscolor.WHITE, 18)

        if self.game_over:
            arcade.draw_text("GAME OVER", 10, 30, arcade.csscolor.WHITE, 18)



def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
