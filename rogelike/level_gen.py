from __future__ import annotations

import random
from dataclasses import dataclass, field

PAD = 2


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


def recursive_generate_rect(parent, min_leaf, target_area):
    eligible_divide = []
    if parent.w >= min_leaf * 2 + 1:
        eligible_divide.append("w")
    if parent.h >= min_leaf * 2 + 1:
        eligible_divide.append("h")
    if len(eligible_divide) == 0:
        return

    dim = random.choice(eligible_divide)
    if dim == "w":
        cut = random.randint(min_leaf, parent.w - min_leaf - 1)
        parent.l = Rect(parent.i, parent.j, cut, parent.h)
        parent.r = Rect(parent.i + cut + 1, parent.j, parent.w - cut - 1, parent.h)
    else:
        cut = random.randint(min_leaf, parent.h - min_leaf - 1)
        parent.l = Rect(parent.i, parent.j, parent.w, cut)
        parent.r = Rect(parent.i, parent.j + cut + 1, parent.w, parent.h - cut - 1)

    if (parent.l.w * parent.l.h - target_area) > 0:
        recursive_generate_rect(parent.l, min_leaf, target_area)
    if (parent.r.w * parent.r.h - target_area) > 0:
        recursive_generate_rect(parent.r, min_leaf, target_area)


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


def generate_corridors_two_rooms(room1, room2, min_corridor_width):
    # Vertical split: rooms side by side — horizontal corridor.
    if room1.i + room1.w <= room2.i or room2.i + room2.w <= room1.i:
        if room2.i < room1.i:
            room1, room2 = room2, room1
        overlap_start = max(room1.j, room2.j)
        overlap_end = min(room1.j + room1.h, room2.j + room2.h)
        overlap = overlap_end - overlap_start
        if overlap >= min_corridor_width + 2:
            width = random.randint(min_corridor_width, overlap - 2)
            j = random.randint(overlap_start + 1, overlap_end - width - 1)
            start_i = room1.i + room1.w
            length = room2.i - start_i
            return [Rect(start_i, j, length, width)]
        return l_bend_horizontal_first(room1, room2, min_corridor_width)

    # Horizontal split: rooms stacked — vertical corridor.
    if room2.j < room1.j:
        room1, room2 = room2, room1
    overlap_start = max(room1.i, room2.i)
    overlap_end = min(room1.i + room1.w, room2.i + room2.w)
    overlap = overlap_end - overlap_start
    if overlap >= min_corridor_width + 2:
        width = random.randint(min_corridor_width, overlap - 2)
        i = random.randint(overlap_start + 1, overlap_end - width - 1)
        start_j = room1.j + room1.h
        length = room2.j - start_j
        return [Rect(i, start_j, width, length)]
    return l_bend_vertical_first(room1, room2, min_corridor_width)


def l_bend_horizontal_first(room1, room2, min_corridor_width):
    # room1 is to the left of room2. Insufficient j-overlap.
    # Exit room1's right edge horizontally, bend, enter room2's top or bottom.
    W = min_corridor_width

    r1_lo = room1.j + 1
    r1_hi = room1.j + room1.h - W - 1

    candidates = []
    below_hi = min(r1_hi, room2.j - W)
    if r1_lo <= below_hi:
        candidates.append((r1_lo, below_hi, "below"))
    above_lo = max(r1_lo, room2.j + room2.h)
    if above_lo <= r1_hi:
        candidates.append((above_lo, r1_hi, "above"))

    if not candidates:
        raise ValueError("L-bend: no row in room1 lies outside room2's j-range")

    lo, hi, direction = random.choice(candidates)
    Y = random.randint(lo, hi)
    X = random.randint(room2.i + 1, room2.i + room2.w - W - 1)

    h_start = room1.i + room1.w
    h_length = X + W - h_start
    horizontal = Rect(h_start, Y, h_length, W)

    if direction == "below":
        v_start = Y
        v_length = room2.j - Y
    else:
        v_start = room2.j + room2.h
        v_length = Y + W - v_start
    vertical = Rect(X, v_start, W, v_length)

    return [horizontal, vertical]


def l_bend_vertical_first(room1, room2, min_corridor_width):
    # room1 is below room2 (smaller j). Insufficient i-overlap.
    # Exit room1's top edge vertically, bend, enter room2's left or right.
    W = min_corridor_width

    r1_lo = room1.i + 1
    r1_hi = room1.i + room1.w - W - 1

    candidates = []
    left_hi = min(r1_hi, room2.i - W)
    if r1_lo <= left_hi:
        candidates.append((r1_lo, left_hi, "left"))
    right_lo = max(r1_lo, room2.i + room2.w)
    if right_lo <= r1_hi:
        candidates.append((right_lo, r1_hi, "right"))

    if not candidates:
        raise ValueError("L-bend: no column in room1 lies outside room2's i-range")

    lo, hi, direction = random.choice(candidates)
    X = random.randint(lo, hi)
    Y = random.randint(room2.j + 1, room2.j + room2.h - W - 1)

    v_start = room1.j + room1.h
    v_length = Y + W - v_start
    vertical = Rect(X, v_start, W, v_length)

    if direction == "left":
        h_start = X
        h_length = room2.i - X
    else:
        h_start = room2.i + room2.w
        h_length = X + W - h_start
    horizontal = Rect(h_start, Y, h_length, W)

    return [vertical, horizontal]


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


def generate_corridors(node, min_corridor_width):
    if node.l:
        generate_corridors(node.l, min_corridor_width)
    if node.r:
        generate_corridors(node.r, min_corridor_width)

    if node.l and node.r:
        rooms_l = get_all_rooms_in_subtree(node.l)
        rooms_r = get_all_rooms_in_subtree(node.r)
        if rooms_l and rooms_r:
            room_l, room_r = closest_pair_of_rooms(rooms_l, rooms_r)
            node.corridors = generate_corridors_two_rooms(
                room_l, room_r, min_corridor_width
            )


def generate_random_room_in_leaves(node, min_dim):
    if node.l:
        generate_random_room_in_leaves(node.l, min_dim)
    if node.r:
        generate_random_room_in_leaves(node.r, min_dim)

    if not node.l and not node.r:
        # this is a leaf — fill most of it, with small random shrink + offset.
        # BSP guarantees node.w >= min_dim + 2*PAD, so max_w >= min_dim always.
        MAX_SHRINK = 4

        max_w = node.w - 2 * PAD
        max_h = node.h - 2 * PAD

        shrink_w = random.randint(0, min(MAX_SHRINK, max_w - min_dim))
        room_w = max_w - shrink_w
        room_i = node.i + PAD + random.randint(0, shrink_w)

        shrink_h = random.randint(0, min(MAX_SHRINK, max_h - min_dim))
        room_h = max_h - shrink_h
        room_j = node.j + PAD + random.randint(0, shrink_h)

        node.room = Rect(room_i, room_j, room_w, room_h)


# BSP room generation
def generate_level(
    width,
    height,
    seed=None,
    min_dim=10,
    min_corridor_width=3,
    target_area=1000,
):
    if min_corridor_width < 3:
        raise ValueError(
            f"min_corridor_width must be >= 3, got {min_corridor_width}"
        )
    if min_dim < min_corridor_width + 2:
        raise ValueError(
            f"min_dim ({min_dim}) must be >= min_corridor_width + 2 "
            f"({min_corridor_width + 2}) so rooms can hold a corridor with margin"
        )
    min_leaf = min_dim + 2 * PAD
    if target_area < min_leaf * min_leaf:
        raise ValueError(
            f"target_area ({target_area}) must be >= min_leaf**2 "
            f"({min_leaf * min_leaf}); smaller values have no effect since "
            f"the BSP can't split below min_leaf anyway"
        )

    if seed:
        random.seed(seed)

    root = Rect(0, 0, width - 1, height - 1)
    recursive_generate_rect(root, min_leaf, target_area)
    generate_random_room_in_leaves(root, min_dim)
    generate_corridors(root, min_corridor_width)

    level = [[2 for _ in range(height)] for _ in range(width)]
    # draw each rectangle (which is wrong, but I want to see it)
    recursive_set_rect_bnd_1(root, level)
    recursive_set_room_0(root, level)
    recursive_set_corridor_0(root, level)

    return level
