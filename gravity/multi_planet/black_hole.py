import arcade
import random

import vector

FILE_PATH = "../assets/black_hole.png"

def GenerateRandomSprite(width, height):
    sprite = arcade.Sprite(FILE_PATH, 0.5)
    minX = int(sprite.width / 2)
    maxX = int(width - sprite.width / 2)
    minY = int(sprite.height / 2)
    maxY = int(height - sprite.height / 2)
    
    sprite.center_x = random.randint(minX, maxX)
    sprite.center_y = random.randint(minY, maxY)
    return sprite

class BlackHole():
    def __init__(self, width, height, sprite_list):
        collisions = [2]
        sprite = None
        while len(collisions) > 0:
            sprite = GenerateRandomSprite(width, height)
            collisions = arcade.check_for_collision_with_list(sprite, sprite_list)
        self.sprite = sprite

    def draw(self):
        self.sprite.draw()

    def get_collision_sprite(self):
        return self.sprite