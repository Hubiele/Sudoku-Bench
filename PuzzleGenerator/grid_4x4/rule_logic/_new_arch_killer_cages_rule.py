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
        is_connected,
        orthogonal_neighbors,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
        is_connected,
        orthogonal_neighbors,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "killer_cages"
MAX_CAGE_SIZE = 4
MAX_CANDIDATES_PER_BOARD: int | None = 48
MAX_SAMPLE_ATTEMPTS_PER_BOARD = 600


@dataclass(frozen=True)
class KillerCage:
    cells: tuple[int, ...]
    clue_sum: int

    def to_json_dict(self) -> dict:
        anchor = min(self.cells)
        return {
            "cells": [list(divmod(c, SIDE)) for c in self.cells],
            "clue_sum": int(self.clue_sum),
            "anchor": list(divmod(anchor, SIDE)),
        }


def enumerate_candidate_cages(solution_grid: list[int], max_cage_size: int | None = None) -> list[KillerCage]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError("Expected 16-cell solution grid")
    if max_cage_size is None:
        max_cage_size = MAX_CAGE_SIZE

    cages: set[KillerCage] = set()
    for size in range(2, max_cage_size + 1):
        for cells in combinations(range(CELL_COUNT), size):
            cells = tuple(sorted(cells))
            if not is_connected(cells):
                continue
            values = [solution_grid[c] for c in cells]
            if 0 in values:
                continue
            if len(set(values)) != len(values):
                continue
            cages.add(KillerCage(cells=cells, clue_sum=sum(values)))

    return sorted(cages, key=lambda cage: (len(cage.cells), cage.clue_sum, cage.cells))


def _sample_one_connected_cage(
    solution_grid: list[int],
    *,
    rng: random.Random,
    max_cage_size: int,
) -> KillerCage | None:
    size = rng.randint(2, max_cage_size)
    start = rng.randrange(CELL_COUNT)
    cells = {start}
    values = {solution_grid[start]}
    if 0 in values:
        return None

    while len(cells) < size:
        frontier = set()
        for cell in cells:
            for nb in orthogonal_neighbors(cell):
                if nb in cells:
                    continue
                if solution_grid[nb] == 0:
                    continue
                if solution_grid[nb] in values:
                    continue
                frontier.add(nb)
        if not frontier:
            return None
        chosen = rng.choice(tuple(frontier))
        cells.add(chosen)
        values.add(solution_grid[chosen])

    ordered = tuple(sorted(cells))
    if not is_connected(ordered):
        return None
    return KillerCage(cells=ordered, clue_sum=sum(solution_grid[c] for c in ordered))


def _sample_candidate_cages(
    solution_grid: list[int],
    *,
    rng: random.Random,
    max_cage_size: int,
    max_candidates: int,
) -> list[KillerCage]:
    candidates: set[KillerCage] = set()

    for a in range(CELL_COUNT):
        for b in orthogonal_neighbors(a):
            if b <= a:
                continue
            va = solution_grid[a]
            vb = solution_grid[b]
            if va == 0 or vb == 0 or va == vb:
                continue
            cells = (a, b)
            candidates.add(KillerCage(cells=cells, clue_sum=va + vb))
            if len(candidates) >= max_candidates:
                return sorted(candidates, key=lambda cage: (len(cage.cells), cage.clue_sum, cage.cells))

    attempts = 0
    while len(candidates) < max_candidates and attempts < MAX_SAMPLE_ATTEMPTS_PER_BOARD:
        attempts += 1
        cage = _sample_one_connected_cage(
            solution_grid,
            rng=rng,
            max_cage_size=max_cage_size,
        )
        if cage is not None:
            candidates.add(cage)

    return sorted(candidates, key=lambda cage: (len(cage.cells), cage.clue_sum, cage.cells))


def get_candidate_cages(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_cage_size: int | None = None,
    max_candidates: int | None = None,
) -> list[KillerCage]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError("Expected 16-cell solution grid")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    if rng is None:
        rng = random.Random()
    if max_cage_size is None:
        max_cage_size = MAX_CAGE_SIZE
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD

    if max_candidates is None:
        return enumerate_candidate_cages(solution_grid, max_cage_size=max_cage_size)

    cages = _sample_candidate_cages(
        solution_grid,
        rng=rng,
        max_cage_size=max_cage_size,
        max_candidates=max_candidates,
    )
    cages = cages[:]
    rng.shuffle(cages)
    return cages


def solution_satisfies_killer_cages(
    solution_grid: list[int],
    candidate: KillerCage,
) -> bool:
    values = [solution_grid[c] for c in candidate.cells]
    if any(v == 0 for v in values):
        return False
    if len(set(values)) != len(values):
        return False
    return sum(values) == candidate.clue_sum


def _mask_to_values(mask: int, *, ctx: Any | None = None) -> tuple[int, ...]:
    digits = ctx.digits if ctx is not None else DIGITS
    return tuple(d for d in digits if mask & (1 << d))


def _product_lists(lists: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    if not lists:
        return [()]
    head, *tail = lists
    out: list[tuple[int, ...]] = []
    for v in head:
        for rest in _product_lists(tail):
            out.append((v,) + rest)
    return out


def propagate_killer_cages(
    domains: list[int],
    candidate: KillerCage,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False
        cell_values = [_mask_to_values(domains[cell], ctx=ctx) for cell in candidate.cells]
        feasible: list[tuple[int, ...]] = []
        for combo in _product_lists(cell_values):
            if len(set(combo)) != len(combo):
                continue
            if sum(combo) != candidate.clue_sum:
                continue
            feasible.append(combo)

        if not feasible:
            return False

        for idx, cell in enumerate(candidate.cells):
            allowed_mask = 0
            for combo in feasible:
                allowed_mask |= 1 << combo[idx]
            new_mask = domains[cell] & allowed_mask
            if new_mask == 0:
                return False
            if new_mask != domains[cell]:
                domains[cell] = new_mask
                changed = True
    return True


def killer_cages_candidate_key(candidate: KillerCage) -> tuple:
    return (len(candidate.cells), candidate.clue_sum, candidate.cells)


def killer_cages_describe(candidate: KillerCage) -> str:
    return (
        "killer_cells=" + str(tuple(divmod(c, SIDE) for c in candidate.cells))
        + " | killer_sum=" + str(candidate.clue_sum)
    )


def killer_cages_to_jsonable(candidate: KillerCage) -> dict:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_cages,
    solution_satisfies_fn=solution_satisfies_killer_cages,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_killer_cages(domains, candidate, ctx=ctx),
    candidate_key_fn=killer_cages_candidate_key,
    describe_fn=killer_cages_describe,
    to_jsonable_fn=killer_cages_to_jsonable,
    metadata={"kind": "cage_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "basic_valid_solution",
        "solution_grid": [
            1, 2, 3, 4,
            3, 4, 1, 2,
            2, 1, 4, 3,
            4, 3, 2, 1,
        ],
        "expect_candidates": True,
    }
]
