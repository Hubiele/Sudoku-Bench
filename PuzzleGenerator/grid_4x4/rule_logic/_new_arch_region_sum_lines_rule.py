from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        CELL_COUNT,
        SIDE,
        BOX_H,
        BOX_W,
        canonical_path,
        king_neighbors,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        SIDE,
        BOX_H,
        BOX_W,
        canonical_path,
        king_neighbors,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "region_sum_lines"
MIN_LINE_LENGTH = 3
MAX_LINE_LENGTH = 5
MAX_LINES_PER_RULE = 3
MAX_SINGLE_LINE_CANDIDATES: int | None = None
MAX_RULE_CANDIDATES_PER_BOARD: int | None = None


@dataclass(frozen=True)
class RegionSumLines:
    lines: tuple[tuple[int, ...], ...]

    def to_json_dict(self) -> list[dict]:
        out: list[dict] = []
        for line in self.lines:
            out.append(
                {
                    "cells": [list(divmod(c, SIDE)) for c in line],
                    "anchor": list(divmod(line[0], SIDE)),
                }
            )
        return out


def _box_index(cell: int) -> int:
    r, c = divmod(cell, SIDE)
    return (r // BOX_H) * (SIDE // BOX_W) + (c // BOX_W)


def _region_sum_property_holds(values: tuple[int, ...], line: tuple[int, ...]) -> bool:
    if len(values) != len(line) or len(values) < 2:
        return False

    sums_by_box: dict[int, int] = defaultdict(int)
    counts_by_box: dict[int, int] = defaultdict(int)

    for cell, value in zip(line, values):
        box = _box_index(cell)
        sums_by_box[box] += value
        counts_by_box[box] += 1

    if len(sums_by_box) < 2:
        return False

    target_sums = set(sums_by_box.values())
    return len(target_sums) == 1


def solution_satisfies_region_sum_lines(
    solution_grid: list[int],
    candidate: RegionSumLines,
) -> bool:
    for line in candidate.lines:
        values = tuple(solution_grid[cell] for cell in line)
        if any(v == 0 for v in values):
            return False
        if not _region_sum_property_holds(values, line):
            return False
    return True


def _enumerate_simple_paths(*, min_length: int, max_length: int) -> list[tuple[int, ...]]:
    paths: set[tuple[int, ...]] = set()

    def dfs(path: tuple[int, ...]) -> None:
        if len(path) >= min_length:
            canon = canonical_path(path)
            # Require that the path actually passes through at least two boxes.
            if len({_box_index(cell) for cell in canon}) >= 2:
                paths.add(canon)
        if len(path) == max_length:
            return
        for nb in king_neighbors(path[-1]):
            if nb in path:
                continue
            dfs(path + (nb,))

    for start in range(CELL_COUNT):
        dfs((start,))

    return sorted(paths, key=lambda line: (len(line), line))


def _single_line_candidates(solution_grid: list[int]) -> list[tuple[int, ...]]:
    paths = _enumerate_simple_paths(min_length=MIN_LINE_LENGTH, max_length=MAX_LINE_LENGTH)
    out: list[tuple[int, ...]] = []
    for line in paths:
        values = tuple(solution_grid[cell] for cell in line)
        if any(v == 0 for v in values):
            continue
        if _region_sum_property_holds(values, line):
            out.append(line)
    return out


def enumerate_candidate_region_sum_lines(solution_grid: list[int]) -> list[RegionSumLines]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    single_lines = _single_line_candidates(solution_grid)
    if MAX_SINGLE_LINE_CANDIDATES is not None:
        single_lines = single_lines[:MAX_SINGLE_LINE_CANDIDATES]

    candidates: set[RegionSumLines] = set()
    for line in single_lines:
        candidates.add(RegionSumLines(lines=(line,)))

    if MAX_LINES_PER_RULE >= 2:
        for num_lines in range(2, min(MAX_LINES_PER_RULE, len(single_lines)) + 1):
            for choice in combinations(single_lines, num_lines):
                used = set()
                ok = True
                for line in choice:
                    if used & set(line):
                        ok = False
                        break
                    used.update(line)
                if not ok:
                    continue
                lines = tuple(sorted(choice, key=lambda x: (len(x), x)))
                candidates.add(RegionSumLines(lines=lines))

    return sorted(
        candidates,
        key=lambda item: (
            sum(len(line) for line in item.lines),
            len(item.lines),
            item.lines,
        ),
    )


def get_candidate_region_sum_lines(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[RegionSumLines]:
    candidates = enumerate_candidate_region_sum_lines(solution_grid)
    if rng is not None:
        candidates = candidates[:]
        rng.shuffle(candidates)
    if max_candidates is None:
        max_candidates = MAX_RULE_CANDIDATES_PER_BOARD
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def _mask_to_values(mask: int, *, ctx: Any | None = None) -> tuple[int, ...]:
    digits = ctx.digits if ctx is not None else (1, 2, 3, 4)
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


def _enumerate_feasible_line_assignments(
    domains: list[int],
    line: tuple[int, ...],
    *,
    ctx: Any | None = None,
) -> list[tuple[int, ...]]:
    value_options = [_mask_to_values(domains[cell], ctx=ctx) for cell in line]
    feasible: list[tuple[int, ...]] = []
    for combo in _product_lists(value_options):
        if _region_sum_property_holds(combo, line):
            feasible.append(combo)
    return feasible


def propagate_region_sum_lines(
    domains: list[int],
    candidate: RegionSumLines,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False
        for line in candidate.lines:
            feasible = _enumerate_feasible_line_assignments(domains, line, ctx=ctx)
            if not feasible:
                return False
            for idx, cell in enumerate(line):
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


def region_sum_lines_candidate_key(candidate: RegionSumLines) -> tuple:
    return (
        len(candidate.lines),
        sum(len(line) for line in candidate.lines),
        candidate.lines,
    )


def region_sum_lines_describe(candidate: RegionSumLines) -> str:
    return "region_sum_lines=" + str(
        tuple(tuple(divmod(c, SIDE) for c in line) for line in candidate.lines)
    )


def region_sum_lines_to_jsonable(candidate: RegionSumLines) -> list[dict]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_region_sum_lines,
    solution_satisfies_fn=solution_satisfies_region_sum_lines,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_region_sum_lines(domains, candidate, ctx=ctx),
    candidate_key_fn=region_sum_lines_candidate_key,
    describe_fn=region_sum_lines_describe,
    to_jsonable_fn=region_sum_lines_to_jsonable,
    metadata={"kind": "region_sum_path_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "puzzle_0077_style_solution",
        "solution_grid": [
            2, 3, 1, 4,
            4, 1, 3, 2,
            3, 4, 2, 1,
            1, 2, 4, 3,
        ],
        "expect_candidates": True,
    }
]
