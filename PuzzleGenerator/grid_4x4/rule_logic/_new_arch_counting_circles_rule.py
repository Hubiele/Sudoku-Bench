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
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "counting_circles"
MIN_TOTAL_CIRCLES = 1
MAX_TOTAL_CIRCLES = 9
MAX_CANDIDATES_PER_BOARD: int | None = 40
MAX_SAMPLE_ATTEMPTS_PER_BOARD = 500


@dataclass(frozen=True)
class CountingCircles:
    cells: tuple[int, ...]

    def to_json_dict(self) -> dict:
        return {"cells": [list(divmod(c, SIDE)) for c in self.cells]}


def _positions_by_digit(solution_grid: list[int]) -> dict[int, list[int]]:
    pos = {int(d): [] for d in DIGITS}
    for cell, value in enumerate(solution_grid):
        pos[int(value)].append(cell)
    return pos


def _valid_digit_subsets() -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    digits = list(DIGITS)
    for k in range(1, len(digits) + 1):
        for combo in combinations(digits, k):
            total = sum(combo)
            if MIN_TOTAL_CIRCLES <= total <= MAX_TOTAL_CIRCLES:
                subsets.append(tuple(combo))
    return subsets


VALID_COUNTING_SUBSETS = _valid_digit_subsets()


def enumerate_candidate_counting_circles(solution_grid: list[int]) -> list[CountingCircles]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    positions = _positions_by_digit(solution_grid)
    candidates: set[CountingCircles] = set()

    def rec_build(digits_left: tuple[int, ...], acc_cells: tuple[int, ...]) -> None:
        if not digits_left:
            candidates.add(CountingCircles(cells=tuple(sorted(acc_cells))))
            return
        d = digits_left[0]
        for picks in combinations(positions[d], d):
            rec_build(digits_left[1:], acc_cells + tuple(picks))

    for digit_subset in VALID_COUNTING_SUBSETS:
        rec_build(digit_subset, ())

    return sorted(candidates, key=lambda item: (len(item.cells), item.cells))


def _sample_one_counting_circles(
    solution_grid: list[int],
    *,
    rng: random.Random,
) -> CountingCircles | None:
    positions = _positions_by_digit(solution_grid)
    digit_subset = rng.choice(VALID_COUNTING_SUBSETS)

    acc_cells: list[int] = []
    for d in digit_subset:
        pos = positions[d]
        if len(pos) < d:
            return None
        picks = rng.sample(pos, d)
        acc_cells.extend(picks)

    return CountingCircles(cells=tuple(sorted(acc_cells)))


def _sample_candidate_counting_circles(
    solution_grid: list[int],
    *,
    rng: random.Random,
    max_candidates: int,
) -> list[CountingCircles]:
    candidates: set[CountingCircles] = set()

    small_subsets = [subset for subset in VALID_COUNTING_SUBSETS if sum(subset) <= 4]
    seed_subsets = small_subsets if small_subsets else VALID_COUNTING_SUBSETS
    positions = _positions_by_digit(solution_grid)

    for digit_subset in seed_subsets:
        acc_cells: list[int] = []
        ok = True
        for d in digit_subset:
            pos = positions[d]
            if len(pos) < d:
                ok = False
                break
            picks = tuple(sorted(pos[:d]))
            acc_cells.extend(picks)
        if ok:
            candidates.add(CountingCircles(cells=tuple(sorted(acc_cells))))
            if len(candidates) >= max_candidates:
                return sorted(candidates, key=lambda item: (len(item.cells), item.cells))

    attempts = 0
    while len(candidates) < max_candidates and attempts < MAX_SAMPLE_ATTEMPTS_PER_BOARD:
        attempts += 1
        candidate = _sample_one_counting_circles(solution_grid, rng=rng)
        if candidate is not None:
            candidates.add(candidate)

    return sorted(candidates, key=lambda item: (len(item.cells), item.cells))


def get_candidate_counting_circles(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[CountingCircles]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    if rng is None:
        rng = random.Random()
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD

    if max_candidates is None:
        return enumerate_candidate_counting_circles(solution_grid)

    candidates = _sample_candidate_counting_circles(
        solution_grid,
        rng=rng,
        max_candidates=max_candidates,
    )
    candidates = candidates[:]
    rng.shuffle(candidates)
    return candidates


def solution_satisfies_counting_circles(
    solution_grid: list[int],
    counting_circles: CountingCircles,
) -> bool:
    counts = {d: 0 for d in DIGITS}
    for cell in counting_circles.cells:
        counts[solution_grid[cell]] += 1
    return all(counts[d] in (0, d) for d in DIGITS)


def _mask_to_values(mask: int) -> tuple[int, ...]:
    return tuple(d for d in DIGITS if mask & (1 << d))


def _counting_bounds_ok(counts: dict[int, int], remaining: int) -> bool:
    for d in DIGITS:
        a = counts[d]
        if a > d:
            return False
        if 0 < a < d and a + remaining < d:
            return False
    return True


def _enumerate_feasible_counting_assignments(
    domains: list[int],
    counting_circles: CountingCircles,
) -> list[tuple[int, ...]]:
    cells = counting_circles.cells
    feasible: list[tuple[int, ...]] = []
    counts = {d: 0 for d in DIGITS}
    assignment = [0] * len(cells)

    def rec(i: int) -> None:
        remaining = len(cells) - i
        if not _counting_bounds_ok(counts, remaining):
            return
        if i == len(cells):
            if all(counts[d] in (0, d) for d in DIGITS):
                feasible.append(tuple(assignment))
            return
        cell = cells[i]
        for v in _mask_to_values(domains[cell]):
            counts[v] += 1
            assignment[i] = v
            rec(i + 1)
            counts[v] -= 1

    rec(0)
    return feasible


def propagate_counting_circles(
    domains: list[int],
    candidate: CountingCircles,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False
        feasible = _enumerate_feasible_counting_assignments(domains, candidate)
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


def counting_circles_candidate_key(candidate: CountingCircles) -> tuple:
    return (len(candidate.cells), candidate.cells)


def counting_circles_describe(candidate: CountingCircles) -> str:
    return "counting_cells=" + str(tuple(divmod(c, SIDE) for c in candidate.cells))


def counting_circles_to_jsonable(candidate: CountingCircles) -> dict:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_counting_circles,
    solution_satisfies_fn=solution_satisfies_counting_circles,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_counting_circles(domains, candidate, ctx=ctx),
    candidate_key_fn=counting_circles_candidate_key,
    describe_fn=counting_circles_describe,
    to_jsonable_fn=counting_circles_to_jsonable,
    metadata={"kind": "set_rule"},
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
