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


RULE_NAME = "little_killer"
MAX_CLUES_PER_RULE = 2
MAX_RULE_CANDIDATES_PER_BOARD: int | None = None

_DIAG_DIRS = {
    "upper left": (-1, -1),
    "upper right": (-1, 1),
    "lower left": (1, -1),
    "lower right": (1, 1),
}


@dataclass(frozen=True)
class LittleKillerClue:
    cells: tuple[int, ...]
    clue_sum: int
    outside_coord: tuple[int, int]
    direction: str

    def to_json_dict(self) -> dict:
        return {
            "outside_coord": [int(self.outside_coord[0]), int(self.outside_coord[1])],
            "direction": self.direction,
            "cells": [list(divmod(c, SIDE)) for c in self.cells],
            "clue_sum": int(self.clue_sum),
            "anchor": [int(self.outside_coord[0]), int(self.outside_coord[1])],
        }


@dataclass(frozen=True)
class LittleKiller:
    clues: tuple[LittleKillerClue, ...]

    def to_json_dict(self) -> list[dict]:
        return [clue.to_json_dict() for clue in self.clues]


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < SIDE and 0 <= c < SIDE


def _canonical_lk_path(path: tuple[int, ...]) -> tuple[int, ...]:
    rev = tuple(reversed(path))
    return min(path, rev)


def _enumerate_edge_diagonals() -> list[tuple[tuple[int, ...], tuple[int, int], str]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[tuple[int, ...], tuple[int, int], str]] = []

    for start_r in range(SIDE):
        for start_c in range(SIDE):
            for direction, (dr, dc) in _DIAG_DIRS.items():
                prev_r = start_r - dr
                prev_c = start_c - dc
                if _in_bounds(prev_r, prev_c):
                    continue

                cells: list[int] = []
                r, c = start_r, start_c
                while _in_bounds(r, c):
                    cells.append(r * SIDE + c)
                    r += dr
                    c += dc

                if len(cells) < 2:
                    continue

                seq = tuple(cells)
                canon = _canonical_lk_path(seq)
                if canon in seen:
                    continue
                seen.add(canon)

                outside = (start_r - dr, start_c - dc)
                out_dir = direction

                if canon != seq:
                    end_r, end_c = divmod(seq[-1], SIDE)
                    outside = (end_r + dr, end_c + dc)
                    rev_map = {
                        "upper left": "lower right",
                        "upper right": "lower left",
                        "lower left": "upper right",
                        "lower right": "upper left",
                    }
                    out_dir = rev_map[direction]

                out.append((canon, outside, out_dir))

    return sorted(out, key=lambda item: (len(item[0]), item[0], item[1], item[2]))


EDGE_DIAGONALS = _enumerate_edge_diagonals()


def solution_satisfies_little_killer(
    solution_grid: list[int],
    candidate: LittleKiller,
) -> bool:
    return all(sum(solution_grid[cell] for cell in clue.cells) == clue.clue_sum for clue in candidate.clues)


def enumerate_candidate_little_killer(solution_grid: list[int]) -> list[LittleKiller]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    single_clues: list[LittleKillerClue] = []
    for cells, outside, direction in EDGE_DIAGONALS:
        single_clues.append(
            LittleKillerClue(
                cells=cells,
                clue_sum=sum(solution_grid[cell] for cell in cells),
                outside_coord=outside,
                direction=direction,
            )
        )

    candidates: set[LittleKiller] = set()
    for clue in single_clues:
        candidates.add(LittleKiller(clues=(clue,)))

    if MAX_CLUES_PER_RULE >= 2:
        for clue_a, clue_b in combinations(single_clues, 2):
            if set(clue_a.cells) & set(clue_b.cells):
                continue
            clues = tuple(
                sorted(
                    (clue_a, clue_b),
                    key=lambda x: (len(x.cells), x.cells, x.outside_coord, x.direction),
                )
            )
            candidates.add(LittleKiller(clues=clues))

    return sorted(
        candidates,
        key=lambda item: (
            sum(len(clue.cells) for clue in item.clues),
            len(item.clues),
            tuple((clue.cells, clue.clue_sum, clue.outside_coord, clue.direction) for clue in item.clues),
        ),
    )


def get_candidate_little_killer(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[LittleKiller]:
    candidates = enumerate_candidate_little_killer(solution_grid)
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_RULE_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


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


def _enumerate_feasible_lk_assignments(
    domains: list[int],
    clue: LittleKillerClue,
    *,
    ctx: Any | None = None,
) -> list[tuple[int, ...]]:
    value_options = [_mask_to_values(domains[cell], ctx=ctx) for cell in clue.cells]
    feasible: list[tuple[int, ...]] = []
    for combo in _product_lists(value_options):
        if sum(combo) == clue.clue_sum:
            feasible.append(combo)
    return feasible


def propagate_little_killer(
    domains: list[int],
    candidate: LittleKiller,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False
        for clue in candidate.clues:
            feasible = _enumerate_feasible_lk_assignments(domains, clue, ctx=ctx)
            if not feasible:
                return False

            for idx, cell in enumerate(clue.cells):
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


def little_killer_candidate_key(candidate: LittleKiller) -> tuple:
    return (
        len(candidate.clues),
        sum(len(clue.cells) for clue in candidate.clues),
        tuple((clue.cells, clue.clue_sum, clue.outside_coord, clue.direction) for clue in candidate.clues),
    )


def little_killer_describe(candidate: LittleKiller) -> str:
    return "little_killer=" + str(
        tuple(
            {
                "cells": tuple(divmod(c, SIDE) for c in clue.cells),
                "sum": clue.clue_sum,
                "outside": clue.outside_coord,
                "direction": clue.direction,
            }
            for clue in candidate.clues
        )
    )


def little_killer_to_jsonable(candidate: LittleKiller) -> list[dict]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_little_killer,
    solution_satisfies_fn=solution_satisfies_little_killer,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_little_killer(domains, candidate, ctx=ctx),
    candidate_key_fn=little_killer_candidate_key,
    describe_fn=little_killer_describe,
    to_jsonable_fn=little_killer_to_jsonable,
    metadata={"kind": "diagonal_sum_rule"},
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
