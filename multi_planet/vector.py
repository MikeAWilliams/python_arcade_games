class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def Add(first, second):
    return Vector2D(first.x + second.x, first.y + second.y)

def Subtract(first, second):
    return Vector2D(first.x - second.x, first.y - second.y)