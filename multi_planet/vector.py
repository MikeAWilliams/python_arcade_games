class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Vector2D(self.x, self.y)


def Add(first, second):
    return Vector2D(first.x + second.x, first.y + second.y)

def Subtract(first, second):
    return Vector2D(first.x - second.x, first.y - second.y)

def Multipy(vec, const):
    return Vector2D(vec.x * const, vec.y * const)