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
        orthogonal_neighbors,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        SIDE,
        orthogonal_neighbors,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "partial_kropki"
MAX_WHITE_DOTS_PER_RULE = 1
MAX_BLACK_DOTS_PER_RULE = 1
MAX_RULE_CANDIDATES_PER_BOARD: int | None = None


@dataclass(frozen=True)
class PartialKropki:
    white_dots: tuple[tuple[int, int], ...]
    black_dots: tuple[tuple[int, int], ...]

    def to_json_dict(self) -> dict:
        return {
            "white_dots": [
                {
                    "cells": [list(divmod(a, SIDE)), list(divmod(b, SIDE))],
                    "anchor": list(divmod(a, SIDE)),
                }
                for a, b in self.white_dots
            ],
            "black_dots": [
                {
                    "cells": [list(divmod(a, SIDE)), list(divmod(b, SIDE))],
                    "anchor": list(divmod(a, SIDE)),
                }
                for a, b in self.black_dots
            ],
        }


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _orthogonal_edges() -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a in range(CELL_COUNT):
        for b in orthogonal_neighbors(a):
            if a < b:
                edges.add((a, b))
    return sorted(edges)


ORTHOGONAL_EDGES = _orthogonal_edges()


def _is_white_pair(a: int, b: int) -> bool:
    return abs(a - b) == 1


def _is_black_pair(a: int, b: int) -> bool:
    lo, hi = sorted((a, b))
    return hi == 2 * lo


def solution_satisfies_partial_kropki(
    solution_grid: list[int],
    candidate: PartialKropki,
) -> bool:
    for a, b in candidate.white_dots:
        va, vb = solution_grid[a], solution_grid[b]
        if va == 0 or vb == 0 or not _is_white_pair(va, vb):
            return False
    for a, b in candidate.black_dots:
        va, vb = solution_grid[a], solution_grid[b]
        if va == 0 or vb == 0 or not _is_black_pair(va, vb):
            return False
    return True


def enumerate_candidate_partial_kropki(solution_grid: list[int]) -> list[PartialKropki]:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    if any(v == 0 for v in solution_grid):
        raise ValueError("Solution grid must be fully solved")

    white_edges: list[tuple[int, int]] = []
    black_edges: list[tuple[int, int]] = []
    for a, b in ORTHOGONAL_EDGES:
        va, vb = solution_grid[a], solution_grid[b]
        if _is_white_pair(va, vb):
            white_edges.append((a, b))
        if _is_black_pair(va, vb):
            black_edges.append((a, b))

    candidates: set[PartialKropki] = set()

    max_white = min(MAX_WHITE_DOTS_PER_RULE, len(white_edges))
    max_black = min(MAX_BLACK_DOTS_PER_RULE, len(black_edges))

    # At least one dot total.
    for w_count in range(0, max_white + 1):
        for b_count in range(0, max_black + 1):
            if w_count + b_count == 0:
                continue
            for white_choice in combinations(white_edges, w_count):
                white_set = set(white_choice)
                for black_choice in combinations(black_edges, b_count):
                    black_set = set(black_choice)
                    if white_set & black_set:
                        continue
                    candidates.add(
                        PartialKropki(
                            white_dots=tuple(sorted(white_choice)),
                            black_dots=tuple(sorted(black_choice)),
                        )
                    )

    return sorted(
        candidates,
        key=lambda item: (
            len(item.white_dots) + len(item.black_dots),
            len(item.white_dots),
            len(item.black_dots),
            item.white_dots,
            item.black_dots,
        ),
    )


def get_candidate_partial_kropki(
    solution_grid: list[int],
    *,
    rng: random.Random | None = None,
    max_candidates: int | None = None,
) -> list[PartialKropki]:
    candidates = enumerate_candidate_partial_kropki(solution_grid)
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


def _allowed_pairs_for_white(domains: list[int], a: int, b: int, *, ctx: Any | None = None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for va in _mask_to_values(domains[a], ctx=ctx):
        for vb in _mask_to_values(domains[b], ctx=ctx):
            if _is_white_pair(va, vb):
                out.append((va, vb))
    return out


def _allowed_pairs_for_black(domains: list[int], a: int, b: int, *, ctx: Any | None = None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for va in _mask_to_values(domains[a], ctx=ctx):
        for vb in _mask_to_values(domains[b], ctx=ctx):
            if _is_black_pair(va, vb):
                out.append((va, vb))
    return out


def propagate_partial_kropki(
    domains: list[int],
    candidate: PartialKropki,
    *,
    ctx: Any | None = None,
) -> bool:
    changed = True
    while changed:
        changed = False

        for a, b in candidate.white_dots:
            feasible = _allowed_pairs_for_white(domains, a, b, ctx=ctx)
            if not feasible:
                return False
            allowed_a = 0
            allowed_b = 0
            for va, vb in feasible:
                allowed_a |= 1 << va
                allowed_b |= 1 << vb
            new_a = domains[a] & allowed_a
            new_b = domains[b] & allowed_b
            if new_a == 0 or new_b == 0:
                return False
            if new_a != domains[a]:
                domains[a] = new_a
                changed = True
            if new_b != domains[b]:
                domains[b] = new_b
                changed = True

        for a, b in candidate.black_dots:
            feasible = _allowed_pairs_for_black(domains, a, b, ctx=ctx)
            if not feasible:
                return False
            allowed_a = 0
            allowed_b = 0
            for va, vb in feasible:
                allowed_a |= 1 << va
                allowed_b |= 1 << vb
            new_a = domains[a] & allowed_a
            new_b = domains[b] & allowed_b
            if new_a == 0 or new_b == 0:
                return False
            if new_a != domains[a]:
                domains[a] = new_a
                changed = True
            if new_b != domains[b]:
                domains[b] = new_b
                changed = True

    return True


def partial_kropki_candidate_key(candidate: PartialKropki) -> tuple:
    return (
        len(candidate.white_dots) + len(candidate.black_dots),
        len(candidate.white_dots),
        len(candidate.black_dots),
        candidate.white_dots,
        candidate.black_dots,
    )


def partial_kropki_describe(candidate: PartialKropki) -> str:
    return (
        "partial_kropki="
        + str(
            {
                "white_dots": tuple(tuple(divmod(c, SIDE) for c in edge) for edge in candidate.white_dots),
                "black_dots": tuple(tuple(divmod(c, SIDE) for c in edge) for edge in candidate.black_dots),
            }
        )
    )


def partial_kropki_to_jsonable(candidate: PartialKropki) -> dict:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_partial_kropki,
    solution_satisfies_fn=solution_satisfies_partial_kropki,
    propagate_fn=lambda domains, candidate, ctx=None: propagate_partial_kropki(domains, candidate, ctx=ctx),
    candidate_key_fn=partial_kropki_candidate_key,
    describe_fn=partial_kropki_describe,
    to_jsonable_fn=partial_kropki_to_jsonable,
    metadata={"kind": "edge_relation_rule"},
)


ARCH_TEST_CASES = [
    {
        "name": "puzzle_0075_style_solution",
        "solution_grid": [
            3, 4, 2, 1,
            1, 2, 4, 3,
            2, 1, 3, 4,
            4, 3, 1, 2,
        ],
        "expect_candidates": True,
    }
]
