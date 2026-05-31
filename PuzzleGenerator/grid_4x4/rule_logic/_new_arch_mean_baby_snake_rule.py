from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "mean_baby_snake"
MAX_CANDIDATES_PER_BOARD: int | None = None


@dataclass(frozen=True)
class MeanBabySnake:
    active: bool = True

    def to_json_dict(self) -> dict:
        return {"active": self.active}


def _friendly_box_number(cell: int) -> int:
    r, c = divmod(cell, SIDE)
    return (r // 2) * 2 + (c // 2) + 1


def _is_friendly_value(cell: int, value: int) -> bool:
    r, c = divmod(cell, SIDE)
    return value == (r + 1) or value == (c + 1) or value == _friendly_box_number(cell)


def _mask_to_values(mask: int, *, ctx: Any | None = None) -> tuple[int, ...]:
    digits = ctx.digits if ctx is not None else DIGITS
    return tuple(d for d in digits if mask & (1 << d))


def _cell_can_be_snake(domains: list[int], cell: int, *, ctx: Any | None = None) -> bool:
    return any(not _is_friendly_value(cell, v) for v in _mask_to_values(domains[cell], ctx=ctx))


def _cell_can_be_friendly(domains: list[int], cell: int, *, ctx: Any | None = None) -> bool:
    return any(_is_friendly_value(cell, v) for v in _mask_to_values(domains[cell], ctx=ctx))


def solution_satisfies_mean_baby_snake(solution_grid: list[int], candidate: MeanBabySnake | None = None) -> bool:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        return False

    snake_cells = {
        cell for cell, value in enumerate(solution_grid)
        if not _is_friendly_value(cell, int(value))
    }

    top_left = 0
    if top_left not in snake_cells:
        return False

    # Must visit every 2x2 box at least once.
    for box in range(4):
        cells = {
            r * SIDE + c
            for r in range((box // 2) * 2, (box // 2) * 2 + 2)
            for c in range((box % 2) * 2, (box % 2) * 2 + 2)
        }
        if not (snake_cells & cells):
            return False

    # Snake must form a single simple orthogonally connected path.
    degrees: dict[int, int] = {}
    for cell in snake_cells:
        r, c = divmod(cell, SIDE)
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < SIDE and 0 <= cc < SIDE and (rr * SIDE + cc) in snake_cells:
                deg += 1
        if deg > 2:
            return False
        degrees[cell] = deg

    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(snake_cells) == 1:
        return False
    if len(endpoints) != 2:
        return False
    if degrees[top_left] != 1:
        return False

    seen = {top_left}
    stack = [top_left]
    while stack:
        cur = stack.pop()
        r, c = divmod(cur, SIDE)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            nb = rr * SIDE + cc
            if 0 <= rr < SIDE and 0 <= cc < SIDE and nb in snake_cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == snake_cells


def get_candidate_mean_baby_snake(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[MeanBabySnake]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    candidates = [MeanBabySnake()] if solution_satisfies_mean_baby_snake(solution_grid) else []
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def propagate_mean_baby_snake(
    domains: list[int],
    candidate: MeanBabySnake,
    *,
    ctx: Any | None = None,
) -> bool:
    if not candidate.active:
        return True

    # Top-left must belong to the snake, so it must be able to take a non-friendly value.
    if not _cell_can_be_snake(domains, 0, ctx=ctx):
        return False

    # Every 2x2 box must still be able to contain at least one snake cell.
    for box in range(4):
        cells = [
            r * SIDE + c
            for r in range((box // 2) * 2, (box // 2) * 2 + 2)
            for c in range((box % 2) * 2, (box % 2) * 2 + 2)
        ]
        if not any(_cell_can_be_snake(domains, cell, ctx=ctx) for cell in cells):
            return False

    # A cell that is already forced to be on the snake cannot be isolated forever.
    for cell in range(CELL_COUNT):
        if _cell_can_be_snake(domains, cell, ctx=ctx) and not _cell_can_be_friendly(domains, cell, ctx=ctx):
            r, c = divmod(cell, SIDE)
            possible_neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIDE and 0 <= cc < SIDE:
                    nb = rr * SIDE + cc
                    if _cell_can_be_snake(domains, nb, ctx=ctx):
                        possible_neighbors += 1
            if possible_neighbors < 1:
                return False

    return True


def mean_baby_snake_candidate_key(candidate: MeanBabySnake) -> tuple[int]:
    return (1 if candidate.active else 0,)


def mean_baby_snake_describe(candidate: MeanBabySnake) -> str:
    return "mean_baby_snake=on" if candidate.active else "mean_baby_snake=off"


def mean_baby_snake_to_jsonable(candidate: MeanBabySnake) -> dict:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_mean_baby_snake,
    solution_satisfies_fn=lambda solution_grid, candidate: solution_satisfies_mean_baby_snake(solution_grid, candidate),
    propagate_fn=lambda domains, candidate, ctx=None: propagate_mean_baby_snake(domains, candidate, ctx=ctx),
    candidate_key_fn=mean_baby_snake_candidate_key,
    describe_fn=mean_baby_snake_describe,
    to_jsonable_fn=mean_baby_snake_to_jsonable,
    metadata={"kind": "global_shape_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "valid_mean_baby_snake_solution",
        "solution_grid": [
            2, 1, 4, 3,
            4, 3, 2, 1,
            3, 4, 1, 2,
            1, 2, 3, 4,
        ],
        "expect_candidates": True,
    },
    {
        "name": "invalid_mean_baby_snake_solution",
        "solution_grid": [
            1, 2, 3, 4,
            3, 4, 1, 2,
            2, 1, 4, 3,
            4, 3, 2, 1,
        ],
        "expect_candidates": False,
    },
]
