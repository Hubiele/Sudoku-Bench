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


RULE_NAME = "even_cells"
MIN_TOTAL_EVEN_CELLS = 1
MAX_TOTAL_EVEN_CELLS = 3
MAX_CANDIDATES_PER_BOARD: int | None = None
EVEN_DIGITS = tuple(d for d in DIGITS if d % 2 == 0)
EVEN_MASK = sum(1 << d for d in EVEN_DIGITS)


@dataclass(frozen=True)
class EvenCells:
    cells: tuple[int, ...]

    def to_json_dict(self) -> list[list[int]]:
        return [list(divmod(c, SIDE)) for c in self.cells]


def solution_satisfies_even_cells(
    solution_grid: list[int],
    candidate: EvenCells,
) -> bool:
    return all(solution_grid[cell] in EVEN_DIGITS for cell in candidate.cells)


def enumerate_candidate_even_cells(solution_grid: list[int]) -> list[EvenCells]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    even_positions = [idx for idx, value in enumerate(solution_grid) if value % 2 == 0]
    out: list[EvenCells] = []
    max_total = min(MAX_TOTAL_EVEN_CELLS, len(even_positions))
    for size in range(MIN_TOTAL_EVEN_CELLS, max_total + 1):
        for cells in combinations(even_positions, size):
            out.append(EvenCells(cells=tuple(cells)))
    return sorted(out, key=lambda item: (len(item.cells), item.cells))


def get_candidate_even_cells(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[EvenCells]:
    candidates = enumerate_candidate_even_cells(solution_grid)
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def propagate_even_cells(
    domains: list[int],
    candidate: EvenCells,
    *,
    ctx: Any | None = None,
) -> bool:
    for cell in candidate.cells:
        new_mask = domains[cell] & EVEN_MASK
        if new_mask == 0:
            return False
        domains[cell] = new_mask
    return True


def even_cells_candidate_key(candidate: EvenCells) -> tuple:
    return (len(candidate.cells), candidate.cells)


def even_cells_describe(candidate: EvenCells) -> str:
    return "even_cells=" + str(tuple(divmod(c, SIDE) for c in candidate.cells))


def even_cells_to_jsonable(candidate: EvenCells) -> list[list[int]]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_even_cells,
    solution_satisfies_fn=solution_satisfies_even_cells,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_even_cells(domains, candidate, ctx=ctx),
    candidate_key_fn=even_cells_candidate_key,
    describe_fn=even_cells_describe,
    to_jsonable_fn=even_cells_to_jsonable,
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
