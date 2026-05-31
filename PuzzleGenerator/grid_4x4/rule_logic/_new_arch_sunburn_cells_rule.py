from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
        king_neighbors,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
        king_neighbors,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "sunburn_cells"
MIN_TOTAL_SUNBURN_CELLS = 1
MAX_TOTAL_SUNBURN_CELLS = 3
MAX_CANDIDATES_PER_BOARD: int | None = None


@dataclass(frozen=True)
class SunburnCells:
    cells: tuple[int, ...]

    def to_json_dict(self) -> list[dict]:
        return [
            {
                "cells": [list(divmod(c, SIDE))],
                "anchor": list(divmod(c, SIDE)),
            }
            for c in self.cells
        ]


def _count_smaller_neighbors(solution_grid: list[int], cell: int) -> int:
    center = solution_grid[cell]
    return sum(1 for nb in king_neighbors(cell) if solution_grid[nb] < center)


def solution_satisfies_sunburn_cells(
    solution_grid: list[int],
    candidate: SunburnCells,
) -> bool:
    for cell in candidate.cells:
        center = solution_grid[cell]
        if center == 0:
            return False
        if _count_smaller_neighbors(solution_grid, cell) != center:
            return False
    return True


def enumerate_candidate_sunburn_cells(solution_grid: list[int]) -> list[SunburnCells]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    valid_positions = [
        cell for cell in range(CELL_COUNT)
        if _count_smaller_neighbors(solution_grid, cell) == solution_grid[cell]
    ]

    out: list[SunburnCells] = []
    max_total = min(MAX_TOTAL_SUNBURN_CELLS, len(valid_positions))
    for size in range(MIN_TOTAL_SUNBURN_CELLS, max_total + 1):
        for cells in combinations(valid_positions, size):
            out.append(SunburnCells(cells=tuple(cells)))
    return sorted(out, key=lambda item: (len(item.cells), item.cells))


def get_candidate_sunburn_cells(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[SunburnCells]:
    candidates = enumerate_candidate_sunburn_cells(solution_grid)
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def _mask_to_values(mask: int, *, ctx: Any | None = None) -> tuple[int, ...]:
    digits = ctx.digits if ctx is not None else DIGITS
    return tuple(d for d in digits if mask & (1 << d))


def _neighbor_allows_smaller(domains: list[int], nb: int, center_value: int, *, ctx: Any | None = None) -> bool:
    return any(v < center_value for v in _mask_to_values(domains[nb], ctx=ctx))


def _neighbor_allows_not_smaller(domains: list[int], nb: int, center_value: int, *, ctx: Any | None = None) -> bool:
    return any(v >= center_value for v in _mask_to_values(domains[nb], ctx=ctx))


def _center_value_feasible(domains: list[int], cell: int, center_value: int, *, ctx: Any | None = None) -> bool:
    neighbors = king_neighbors(cell)
    must_small = 0
    can_small = 0
    for nb in neighbors:
        allows_small = _neighbor_allows_smaller(domains, nb, center_value, ctx=ctx)
        allows_not_small = _neighbor_allows_not_smaller(domains, nb, center_value, ctx=ctx)
        if not allows_small and not allows_not_small:
            return False
        if allows_small:
            can_small += 1
        if allows_small and not allows_not_small:
            must_small += 1
    return must_small <= center_value <= can_small


def _neighbor_value_feasible(
    domains: list[int],
    cell: int,
    center_value: int,
    nb: int,
    nb_value: int,
    *,
    ctx: Any | None = None,
) -> bool:
    target_remaining = center_value - (1 if nb_value < center_value else 0)
    others = [x for x in king_neighbors(cell) if x != nb]
    must_small = 0
    can_small = 0
    for other in others:
        allows_small = _neighbor_allows_smaller(domains, other, center_value, ctx=ctx)
        allows_not_small = _neighbor_allows_not_smaller(domains, other, center_value, ctx=ctx)
        if not allows_small and not allows_not_small:
            return False
        if allows_small:
            can_small += 1
        if allows_small and not allows_not_small:
            must_small += 1
    return must_small <= target_remaining <= can_small


def propagate_sunburn_cells(
    domains: list[int],
    candidate: SunburnCells,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False

        for cell in candidate.cells:
            center_mask = domains[cell]
            feasible_center_values = [
                cv for cv in _mask_to_values(center_mask, ctx=ctx)
                if _center_value_feasible(domains, cell, cv, ctx=ctx)
            ]
            if not feasible_center_values:
                return False

            allowed_center_mask = 0
            for cv in feasible_center_values:
                allowed_center_mask |= 1 << cv

            new_center_mask = center_mask & allowed_center_mask
            if new_center_mask == 0:
                return False
            if new_center_mask != center_mask:
                domains[cell] = new_center_mask
                changed = True

            # prune neighbor domains relative to the remaining feasible center values
            for nb in king_neighbors(cell):
                nb_mask = domains[nb]
                allowed_nb_mask = 0
                for cv in _mask_to_values(domains[cell], ctx=ctx):
                    for nv in _mask_to_values(nb_mask, ctx=ctx):
                        if _neighbor_value_feasible(domains, cell, cv, nb, nv, ctx=ctx):
                            allowed_nb_mask |= 1 << nv
                new_nb_mask = nb_mask & allowed_nb_mask
                if new_nb_mask == 0:
                    return False
                if new_nb_mask != nb_mask:
                    domains[nb] = new_nb_mask
                    changed = True

    return True


def sunburn_cells_candidate_key(candidate: SunburnCells) -> tuple:
    return (len(candidate.cells), candidate.cells)


def sunburn_cells_describe(candidate: SunburnCells) -> str:
    return "sunburn_cells=" + str(tuple(divmod(c, SIDE) for c in candidate.cells))


def sunburn_cells_to_jsonable(candidate: SunburnCells) -> list[dict]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_sunburn_cells,
    solution_satisfies_fn=solution_satisfies_sunburn_cells,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_sunburn_cells(domains, candidate, ctx=ctx),
    candidate_key_fn=sunburn_cells_candidate_key,
    describe_fn=sunburn_cells_describe,
    to_jsonable_fn=sunburn_cells_to_jsonable,
    metadata={"kind": "neighborhood_count_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "puzzle_0083_style_solution",
        "solution_grid": [
            3, 4, 2, 1,
            1, 2, 4, 3,
            2, 3, 1, 4,
            4, 1, 3, 2,
        ],
        "expect_candidates": True,
    }
]
