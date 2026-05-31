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


RULE_NAME = "odd_cells"
MIN_TOTAL_ODD_CELLS = 1
MAX_TOTAL_ODD_CELLS = 3
MAX_CANDIDATES_PER_BOARD: int | None = None
ODD_DIGITS = tuple(d for d in DIGITS if d % 2 == 1)
ODD_MASK = sum(1 << d for d in ODD_DIGITS)


@dataclass(frozen=True)
class OddCells:
    cells: tuple[int, ...]

    def to_json_dict(self) -> list[list[int]]:
        return [list(divmod(c, SIDE)) for c in self.cells]


def solution_satisfies_odd_cells(
    solution_grid: list[int],
    candidate: OddCells,
) -> bool:
    return all(solution_grid[cell] in ODD_DIGITS for cell in candidate.cells)


def enumerate_candidate_odd_cells(solution_grid: list[int]) -> list[OddCells]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    odd_positions = [idx for idx, value in enumerate(solution_grid) if value % 2 == 1]
    out: list[OddCells] = []
    max_total = min(MAX_TOTAL_ODD_CELLS, len(odd_positions))
    for size in range(MIN_TOTAL_ODD_CELLS, max_total + 1):
        for cells in combinations(odd_positions, size):
            out.append(OddCells(cells=tuple(cells)))
    return sorted(out, key=lambda item: (len(item.cells), item.cells))


def get_candidate_odd_cells(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[OddCells]:
    candidates = enumerate_candidate_odd_cells(solution_grid)
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def propagate_odd_cells(
    domains: list[int],
    candidate: OddCells,
    *,
    ctx: Any | None = None,
) -> bool:
    for cell in candidate.cells:
        new_mask = domains[cell] & ODD_MASK
        if new_mask == 0:
            return False
        domains[cell] = new_mask
    return True


def odd_cells_candidate_key(candidate: OddCells) -> tuple:
    return (len(candidate.cells), candidate.cells)


def odd_cells_describe(candidate: OddCells) -> str:
    return "odd_cells=" + str(tuple(divmod(c, SIDE) for c in candidate.cells))


def odd_cells_to_jsonable(candidate: OddCells) -> list[list[int]]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_odd_cells,
    solution_satisfies_fn=solution_satisfies_odd_cells,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_odd_cells(domains, candidate, ctx=ctx),
    candidate_key_fn=odd_cells_candidate_key,
    describe_fn=odd_cells_describe,
    to_jsonable_fn=odd_cells_to_jsonable,
    metadata={"kind": "set_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "puzzle_0074_style_solution",
        "solution_grid": [
            3, 2, 4, 1,
            4, 1, 3, 2,
            1, 4, 2, 3,
            2, 3, 1, 4,
        ],
        "expect_candidates": True,
    }
]
