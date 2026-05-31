from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SIDE = 4
DIGITS = (1, 2, 3, 4)
BOX_H = 2
BOX_W = 2
CELL_COUNT = SIDE * SIDE
ALL_VALUES_MASK = sum(1 << d for d in DIGITS)

ROWS: list[tuple[int, ...]] = [tuple(r * SIDE + c for c in range(SIDE)) for r in range(SIDE)]
COLS: list[tuple[int, ...]] = [tuple(r * SIDE + c for r in range(SIDE)) for c in range(SIDE)]
BOXES: list[tuple[int, ...]] = []
for br in range(0, SIDE, BOX_H):
    for bc in range(0, SIDE, BOX_W):
        box = []
        for r in range(br, br + BOX_H):
            for c in range(bc, bc + BOX_W):
                box.append(r * SIDE + c)
        BOXES.append(tuple(box))

UNITS = ROWS + COLS + BOXES
CELL_TO_UNITS = {i: [u for u in UNITS if i in u] for i in range(CELL_COUNT)}
PEERS = {i: tuple(sorted({c for u in CELL_TO_UNITS[i] for c in u if c != i})) for i in range(CELL_COUNT)}


def parse_grid(flat: str) -> list[int]:
    flat = flat.strip()
    if len(flat) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} chars for 4x4 grid, got {len(flat)}")
    if not flat.isdigit():
        raise ValueError("Grid must contain only digits")
    values = [int(ch) for ch in flat]
    if any(v < 0 or v > SIDE for v in values):
        raise ValueError("Grid contains out-of-range values")
    return values


def grid_to_string(grid: Iterable[int]) -> str:
    return "".join(str(int(x)) for x in grid)


def count_givens(grid: Iterable[int]) -> int:
    return sum(1 for x in grid if int(x) != 0)


def orthogonal_neighbors(cell: int) -> tuple[int, ...]:
    r, c = divmod(cell, SIDE)
    out: list[int] = []
    if r > 0:
        out.append((r - 1) * SIDE + c)
    if r < SIDE - 1:
        out.append((r + 1) * SIDE + c)
    if c > 0:
        out.append(r * SIDE + (c - 1))
    if c < SIDE - 1:
        out.append(r * SIDE + (c + 1))
    return tuple(out)


def king_neighbors(cell: int) -> tuple[int, ...]:
    r, c = divmod(cell, SIDE)
    out: list[int] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < SIDE and 0 <= cc < SIDE:
                out.append(rr * SIDE + cc)
    return tuple(sorted(set(out)))


def knight_neighbors() -> dict[int, tuple[int, ...]]:
    offsets = (
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2), (1, 2),
        (2, -1), (2, 1),
    )
    out: dict[int, tuple[int, ...]] = {}
    for cell in range(CELL_COUNT):
        r, c = divmod(cell, SIDE)
        nbs: list[int] = []
        for dr, dc in offsets:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < SIDE and 0 <= cc < SIDE:
                nbs.append(rr * SIDE + cc)
        out[cell] = tuple(sorted(set(nbs)))
    return out


KNIGHT_NEIGHBORS = knight_neighbors()


def is_connected(cells: tuple[int, ...]) -> bool:
    if not cells:
        return False
    remaining = set(cells)
    stack = [next(iter(remaining))]
    seen = set(stack)
    while stack:
        cur = stack.pop()
        for nb in orthogonal_neighbors(cur):
            if nb in remaining and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == remaining


def canonical_path(path: tuple[int, ...]) -> tuple[int, ...]:
    rev = tuple(reversed(path))
    return min(path, rev)
