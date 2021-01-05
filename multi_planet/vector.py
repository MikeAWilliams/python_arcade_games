import math

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Vector2D(self.x, self.y)
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def make_unit(self):
        len = self.length()
        self.x /= len
        self.y /= len
    
    def print(self, name):
        print("{0} ({1}, {2})".format(name, self.x, self.y))


def Add(first, second):
    return Vector2D(first.x + second.x, first.y + second.y)

def Subtract(first, second):
    return Vector2D(first.x - second.x, first.y - second.y)

def Multipy(vec, const):
    return Vector2D(vec.x * const, vec.y * const)