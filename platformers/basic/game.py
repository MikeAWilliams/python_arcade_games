import arcade

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = "Platformer"

CHARACTER_SCALING = 1
TILE_SCALING = 0.5
COIN_SCALING = 0.5

PLAYER_MOVEMENT_SPEED = 5
PLAYER_JUMP_SPEED = 10
GRAVITY_CONSTANT = 0.5

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.world_list = None
        self.player_list = None
        self.coin_list = None

        self.player_sprite = None

        self.physics_engine = None

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

    def setup(self):
        self.player_list = arcade.SpriteList()
        self.world_list = arcade.SpriteList(use_spatial_hash=True)
        self.coin_list = arcade.SpriteList()

        image_source = ":resources:images/animated_characters/female_adventurer/femaleAdventurer_idle.png"
        self.player_sprite = arcade.Sprite(image_source, CHARACTER_SCALING)
        self.player_sprite.center_x = 64
        self.player_sprite.center_y = 128
        self.player_list.append(self.player_sprite)

        # Create the ground
        # This shows using a loop to place multiple sprites horizontally
        for x in range(0, 1250, 64):
            wall = arcade.Sprite(":resources:images/tiles/grassMid.png", TILE_SCALING)
            wall.center_x = x
            wall.center_y = 32
            self.world_list.append(wall)

        # Put some crates on the ground
        # This shows using a coordinate list to place sprites
        crate_coordinate_list = [[256, 96],
                                 [300, 150],
                                 [512, 96],
                                 [768, 96]]

        for crate_coordinate in crate_coordinate_list:
            # Add a crate on the ground
            crate = arcade.Sprite(":resources:images/tiles/boxCrate_double.png", TILE_SCALING)
            crate.position = crate_coordinate
            self.world_list.append(crate)
        
        coint_coordinates = [[300, 200],
                            [512, 160],
                            [768, 160]]

        for coordinate in coint_coordinates:
            coin = arcade.Sprite(":resources:images/items/coinGold.png", COIN_SCALING)
            coin.position = coordinate
            self.coin_list.append(coin)

        self.score = 0

        self.physics_engine = arcade.PhysicsEnginePlatformer(self.player_sprite, self.world_list, GRAVITY_CONSTANT)
    	

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.close_window()
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
        self.physics_engine.update()

        coin_hit_list = arcade.check_for_collision_with_list(self.player_sprite, self.coin_list)
        for coin in coin_hit_list:
            coin.remove_from_sprite_lists()
            self.score += 1


    def on_draw(self):
        arcade.start_render()
        self.world_list.draw()
        self.player_list.draw()
        self.coin_list.draw()

        score_text = f"Score: {self.score}"
        arcade.draw_text(score_text, 10, 10, arcade.csscolor.WHITE, 18)


def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
