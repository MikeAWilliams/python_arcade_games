from dataclasses import dataclass, field
import math
from collections import deque


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


def get_destinations_from_tile(level, tile, allow_diagonal):
    possible = []
    possible.append(Coord(tile.x + 1, tile.y))
    possible.append(Coord(tile.x - 1, tile.y))
    possible.append(Coord(tile.x, tile.y + 1))
    possible.append(Coord(tile.x, tile.y - 1))
    if allow_diagonal:
        possible.append(Coord(tile.x + 1, tile.y + 1))
        possible.append(Coord(tile.x - 1, tile.y + 1))
        possible.append(Coord(tile.x + 1, tile.y - 1))
        possible.append(Coord(tile.x - 1, tile.y - 1))
    to_remove = []
    for p in possible:
        if level[p.y][p.x] != 0:
            to_remove.append(p)
    for r in to_remove:
        possible.remove(r)
    return possible


def remove_visitited_destinations(result, destinations):
    to_remove = []
    for d in destinations:
        if result[d.y][d.x] != math.inf:
            to_remove.append(d)
    for r in to_remove:
        destinations.remove(r)
    return destinations


def astar_flood(level, destination, allow_diagonal, max_dist=None):
    if level[destination.y][destination.x] != 0:
        raise Exception("Invalid astar desination, must target a zero tile")
    result = init_astar_flood_solution(len(level[0]), len(level))
    current_tile = destination
    result[current_tile.y][current_tile.x] = 0
    queue = deque([current_tile])
    while len(queue) > 0:
        current_tile = queue.popleft()
        current_value = result[current_tile.y][current_tile.x] + 1
        if max_dist and current_value > max_dist:
            continue
        destinations = get_destinations_from_tile(level, current_tile, allow_diagonal)
        destinations = remove_visitited_destinations(result, destinations)
        for d in destinations:
            result[d.y][d.x] = current_value
            queue.append(d)
    return result
