from dataclasses import dataclass, field
import math


@dataclass
class Coord:
    x: int
    y: int


def init_astar_flood_solution(width, height):
    result = []
    for y in range(height):
        new_row = []
        for x in range(width):
            new_row.append(math.inf)
        result.append(new_row)
    return result


def astar_flood(level, destination, allow_diagonal, max_dist=None):
    result = init_astar_flood_solution(len(level[0]), len(level))
    return result
