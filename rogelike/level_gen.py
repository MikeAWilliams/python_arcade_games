from __future__ import annotations

import random
from dataclasses import dataclass, field

MIN_DIM = 10
PAD = 2
MIN_LEAF = MIN_DIM + 2 * PAD
MIN_CORRIDOR_WIDTH = 3


@dataclass
class Rect:
    i: int
    j: int
    w: int
    h: int
    room: Rect | None = None
    corridors: list[Rect] = field(default_factory=list)
    l: Rect | None = None
    r: Rect | None = None


def recursive_generate_rect(parent):
    TARGET_AREA = 1000

    eligible_divide = []
    if parent.w >= MIN_LEAF * 2 + 1:
        eligible_divide.append("w")
    if parent.h >= MIN_LEAF * 2 + 1:
        eligible_divide.append("h")
    if len(eligible_divide) == 0:
        return

    dim = random.choice(eligible_divide)
    if dim == "w":
        cut = random.randint(MIN_LEAF, parent.w - MIN_LEAF - 1)
        parent.l = Rect(parent.i, parent.j, cut, parent.h)
        parent.r = Rect(parent.i + cut + 1, parent.j, parent.w - cut - 1, parent.h)
    else:
        cut = random.randint(MIN_LEAF, parent.h - MIN_LEAF - 1)
        parent.l = Rect(parent.i, parent.j, parent.w, cut)
        parent.r = Rect(parent.i, parent.j + cut + 1, parent.w, parent.h - cut - 1)

    if (parent.l.w * parent.l.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.l)
    if (parent.r.w * parent.r.h - TARGET_AREA) > 0:
        recursive_generate_rect(parent.r)


# for debug, set level 0 before you start and use this to draw the rectangles
def recursive_set_rect_bnd_1(rect, level):
    def set_rect_bnd_1(rect, level):
        for i in range(rect.i, rect.i + rect.w + 1):
            level[i][rect.j] = 1
            level[i][rect.h + rect.j] = 1
        for j in range(rect.j, rect.j + rect.h + 1):
            level[rect.i][j] = 1
            level[rect.w + rect.i][j] = 1
        # top right
        level[rect.i + rect.w][rect.j + rect.h] = 1

    set_rect_bnd_1(rect, level)
    if rect.l:
        recursive_set_rect_bnd_1(rect.l, level)
    if rect.r:
        recursive_set_rect_bnd_1(rect.r, level)


def fill_rect_0(rect, level):
    for i in range(rect.i, rect.i + rect.w):
        for j in range(rect.j, rect.j + rect.h):
            level[i][j] = 0


def recursive_set_corridor_0(node, level):
    if node.l:
        recursive_set_corridor_0(node.l, level)
    if node.r:
        recursive_set_corridor_0(node.r, level)
    if node.corridors:
        for r in node.corridors:
            fill_rect_0(r, level)


def recursive_set_room_0(node, level):
    if node.l:
        recursive_set_room_0(node.l, level)
    if node.r:
        recursive_set_room_0(node.r, level)

    if node.room:
        fill_rect_0(node.room, level)


def generate_corridors_two_rooms(room1, room2):
    # Vertical split: rooms side by side — horizontal corridor.
    if room1.i + room1.w <= room2.i or room2.i + room2.w <= room1.i:
        if room2.i < room1.i:
            room1, room2 = room2, room1
        overlap_start = max(room1.j, room2.j)
        overlap_end = min(room1.j + room1.h, room2.j + room2.h)
        overlap = overlap_end - overlap_start
        if overlap >= MIN_CORRIDOR_WIDTH + 2:
            width = random.randint(MIN_CORRIDOR_WIDTH, overlap - 2)
            j = random.randint(overlap_start + 1, overlap_end - width - 1)
            start_i = room1.i + room1.w
            length = room2.i - start_i
            return [Rect(start_i, j, length, width)]
        raise NotImplementedError("L-bend")

    # Horizontal split: rooms stacked — vertical corridor.
    if room2.j < room1.j:
        room1, room2 = room2, room1
    overlap_start = max(room1.i, room2.i)
    overlap_end = min(room1.i + room1.w, room2.i + room2.w)
    overlap = overlap_end - overlap_start
    if overlap >= MIN_CORRIDOR_WIDTH + 2:
        width = random.randint(MIN_CORRIDOR_WIDTH, overlap - 2)
        i = random.randint(overlap_start + 1, overlap_end - width - 1)
        start_j = room1.j + room1.h
        length = room2.j - start_j
        return [Rect(i, start_j, width, length)]
    raise NotImplementedError("L-bend")


def get_all_rooms_in_subtree(node):
    rooms = []
    if node.room:
        rooms.append(node.room)
    if node.l:
        rooms.extend(get_all_rooms_in_subtree(node.l))
    if node.r:
        rooms.extend(get_all_rooms_in_subtree(node.r))
    return rooms


def closest_pair_of_rooms(rooms_a, rooms_b):
    best = None
    best_dist = float("inf")
    for a in rooms_a:
        ax = a.i + a.w / 2
        ay = a.j + a.h / 2
        for b in rooms_b:
            bx = b.i + b.w / 2
            by = b.j + b.h / 2
            dist = (ax - bx) ** 2 + (ay - by) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (a, b)
    return best


def generate_corridors(node):
    if node.l:
        generate_corridors(node.l)
    if node.r:
        generate_corridors(node.r)

    if node.l and node.r:
        rooms_l = get_all_rooms_in_subtree(node.l)
        rooms_r = get_all_rooms_in_subtree(node.r)
        if rooms_l and rooms_r:
            room_l, room_r = closest_pair_of_rooms(rooms_l, rooms_r)
            node.corridors = generate_corridors_two_rooms(room_l, room_r)


def generate_random_room_in_leaves(node):
    if node.l:
        generate_random_room_in_leaves(node.l)
    if node.r:
        generate_random_room_in_leaves(node.r)

    if not node.l and not node.r:
        # this is a leaf — fill most of it, with small random shrink + offset.
        # BSP guarantees node.w >= MIN_LEAF, so max_w >= MIN_DIM always.
        MAX_SHRINK = 4

        max_w = node.w - 2 * PAD
        max_h = node.h - 2 * PAD

        shrink_w = random.randint(0, min(MAX_SHRINK, max_w - MIN_DIM))
        room_w = max_w - shrink_w
        room_i = node.i + PAD + random.randint(0, shrink_w)

        shrink_h = random.randint(0, min(MAX_SHRINK, max_h - MIN_DIM))
        room_h = max_h - shrink_h
        room_j = node.j + PAD + random.randint(0, shrink_h)

        node.room = Rect(room_i, room_j, room_w, room_h)


# BSP room generation
def generate_level(width, height, seed=42):
    random.seed(seed)
    root = Rect(0, 0, width - 1, height - 1)
    recursive_generate_rect(root)
    generate_random_room_in_leaves(root)
    generate_corridors(root)

    level = [[2 for _ in range(height)] for _ in range(width)]
    # draw each rectangle (which is wrong, but I want to see it)
    recursive_set_rect_bnd_1(root, level)
    recursive_set_room_0(root, level)
    recursive_set_corridor_0(root, level)

    return level
