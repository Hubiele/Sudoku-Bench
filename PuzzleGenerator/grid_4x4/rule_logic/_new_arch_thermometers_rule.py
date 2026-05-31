from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        CELL_COUNT,
        SIDE,
        king_neighbors,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        SIDE,
        king_neighbors,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "thermometers"
MIN_LINE_LENGTH = 2
MAX_LINE_LENGTH = 3
MAX_THERMOMETERS_PER_RULE = 6
MAX_SINGLE_THERMO_CANDIDATES: int | None = None
MAX_RULE_CANDIDATES_PER_BOARD: int | None = None


@dataclass(frozen=True)
class Thermometers:
    lines: tuple[tuple[int, ...], ...]

    def to_json_dict(self) -> list[dict]:
        out: list[dict] = []
        for line in self.lines:
            anchor = line[0]
            out.append(
                {
                    "cells": [list(divmod(c, SIDE)) for c in line],
                    "bulb": list(divmod(anchor, SIDE)),
                    "anchor": list(divmod(anchor, SIDE)),
                }
            )
        return out


def _strictly_increasing(values: tuple[int, ...]) -> bool:
    if len(values) < 2:
        return False
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def solution_satisfies_thermometers(
    solution_grid: list[int],
    candidate: Thermometers,
) -> bool:
    for line in candidate.lines:
        values = tuple(solution_grid[cell] for cell in line)
        if any(v == 0 for v in values):
            return False
        if not _strictly_increasing(values):
            return False
    return True


def _enumerate_simple_paths(*, min_length: int, max_length: int) -> list[tuple[int, ...]]:
    paths: set[tuple[int, ...]] = set()

    def dfs(path: tuple[int, ...]) -> None:
        if len(path) >= min_length:
            paths.add(path)
        if len(path) == max_length:
            return
        for nb in king_neighbors(path[-1]):
            if nb in path:
                continue
            dfs(path + (nb,))

    for start in range(CELL_COUNT):
        dfs((start,))
    return sorted(paths, key=lambda line: (len(line), line))


def _single_thermometer_candidates(solution_grid: list[int]) -> list[tuple[int, ...]]:
    paths = _enumerate_simple_paths(min_length=MIN_LINE_LENGTH, max_length=MAX_LINE_LENGTH)
    out: list[tuple[int, ...]] = []
    for line in paths:
        values = tuple(solution_grid[cell] for cell in line)
        if any(v == 0 for v in values):
            continue
        if _strictly_increasing(values):
            out.append(line)
    return out


def enumerate_candidate_thermometers(solution_grid: list[int]) -> list[Thermometers]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    single_lines = _single_thermometer_candidates(solution_grid)
    if MAX_SINGLE_THERMO_CANDIDATES is not None:
        single_lines = single_lines[:MAX_SINGLE_THERMO_CANDIDATES]

    candidates: set[Thermometers] = set()
    for line in single_lines:
        candidates.add(Thermometers(lines=(line,)))

    # Preserve existing behavior from the old rule module: single thermometers,
    # plus disjoint pairs when MAX_THERMOMETERS_PER_RULE >= 2.
    if MAX_THERMOMETERS_PER_RULE >= 2:
        for line_a, line_b in combinations(single_lines, 2):
            if set(line_a) & set(line_b):
                continue
            lines = tuple(sorted((line_a, line_b), key=lambda x: (len(x), x)))
            candidates.add(Thermometers(lines=lines))

    return sorted(
        candidates,
        key=lambda item: (
            sum(len(line) for line in item.lines),
            len(item.lines),
            item.lines,
        ),
    )


def get_candidate_thermometers(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[Thermometers]:
    candidates = enumerate_candidate_thermometers(solution_grid)
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


def _enumerate_feasible_thermometer_assignments(
    domains: list[int],
    line: tuple[int, ...],
    *,
    ctx: Any | None = None,
) -> list[tuple[int, ...]]:
    value_options = [_mask_to_values(domains[cell], ctx=ctx) for cell in line]
    feasible: list[tuple[int, ...]] = []
    for combo in _product_lists(value_options):
        if _strictly_increasing(combo):
            feasible.append(combo)
    return feasible


def propagate_thermometers(
    domains: list[int],
    candidate: Thermometers,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False
        for line in candidate.lines:
            feasible = _enumerate_feasible_thermometer_assignments(domains, line, ctx=ctx)
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


def thermometers_candidate_key(candidate: Thermometers) -> tuple:
    return (
        len(candidate.lines),
        sum(len(line) for line in candidate.lines),
        candidate.lines,
    )


def thermometers_describe(candidate: Thermometers) -> str:
    return "thermometers=" + str(
        tuple(tuple(divmod(c, SIDE) for c in line) for line in candidate.lines)
    )


def thermometers_to_jsonable(candidate: Thermometers) -> list[dict]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_thermometers,
    solution_satisfies_fn=solution_satisfies_thermometers,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_thermometers(domains, candidate, ctx=ctx),
    candidate_key_fn=thermometers_candidate_key,
    describe_fn=thermometers_describe,
    to_jsonable_fn=thermometers_to_jsonable,
    metadata={"kind": "ordered_path_rule"},
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
