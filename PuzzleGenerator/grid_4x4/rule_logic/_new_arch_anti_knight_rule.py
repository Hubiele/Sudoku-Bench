from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        KNIGHT_NEIGHBORS,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import make_passthrough_rule
except ImportError:
    from .common.common_base_4x4 import (
        CELL_COUNT,
        DIGITS,
        KNIGHT_NEIGHBORS,
    )
    from .common.rule_api import make_passthrough_rule


RULE_NAME = "anti_knight"


@dataclass(frozen=True)
class AntiKnightConstraint:
    active: bool = True

    def to_json_dict(self) -> dict[str, bool]:
        return {"active": self.active}


def _mask_to_values(mask: int) -> tuple[int, ...]:
    return tuple(d for d in DIGITS if mask & (1 << d))


def _popcount(mask: int) -> int:
    return mask.bit_count()


def solution_satisfies_anti_knight(solution_grid: list[int]) -> bool:
    if len(solution_grid) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, got {len(solution_grid)}")
    for cell, value in enumerate(solution_grid):
        if value == 0:
            return False
        for nb in KNIGHT_NEIGHBORS[cell]:
            if nb > cell and solution_grid[nb] == value:
                return False
    return True


def get_candidate_anti_knight(
    solution_grid: list[int],
    *,
    rng=None,
    max_candidates: int | None = None,
) -> list[AntiKnightConstraint]:
    candidates = [AntiKnightConstraint()] if solution_satisfies_anti_knight(solution_grid) else []
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return candidates


def propagate_anti_knight(
    domains: list[int],
    candidate: AntiKnightConstraint,
    *,
    ctx: Any | None = None,
) -> bool:
    if not candidate.active:
        return True
    if len(domains) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} domains, got {len(domains)}")

    changed = True
    while changed:
        changed = False
        for cell in range(CELL_COUNT):
            mask = domains[cell]
            if mask == 0:
                return False
            if _popcount(mask) != 1:
                continue

            forced_value = _mask_to_values(mask)[0]
            bit = 1 << forced_value

            for nb in KNIGHT_NEIGHBORS[cell]:
                nb_mask = domains[nb]
                if nb_mask & bit:
                    new_mask = nb_mask & ~bit
                    if new_mask == 0:
                        return False
                    if new_mask != nb_mask:
                        domains[nb] = new_mask
                        changed = True
    return True


def anti_knight_candidate_key(candidate: AntiKnightConstraint) -> tuple[int]:
    return (1 if candidate.active else 0,)


def anti_knight_describe(candidate: AntiKnightConstraint) -> str:
    return "anti_knight=on" if candidate.active else "anti_knight=off"


def anti_knight_to_jsonable(candidate: AntiKnightConstraint) -> dict[str, bool]:
    return candidate.to_json_dict()


RULE_SPEC = make_passthrough_rule(
    name=RULE_NAME,
    generate_candidates_fn=get_candidate_anti_knight,
    solution_satisfies_fn=lambda solution_grid, candidate: solution_satisfies_anti_knight(solution_grid),
    propagate_fn=lambda domains, candidate, ctx=None: propagate_anti_knight(domains, candidate, ctx=ctx),
    candidate_key_fn=anti_knight_candidate_key,
    describe_fn=anti_knight_describe,
    to_jsonable_fn=anti_knight_to_jsonable,
    metadata={"kind": "boolean_rule"},
)

# Valgfritt brukt av test_new_rule_architecture.py
ARCH_TEST_CASES = [
    {
        "name": "valid_anti_knight_solution",
        "solution_grid": [
            2, 1, 3, 4,
            4, 3, 1, 2,
            3, 4, 2, 1,
            1, 2, 4, 3,
        ],
        "expect_candidates": True,
    },
    {
        "name": "invalid_anti_knight_solution",
        "solution_grid": [
            2, 1, 3, 4,
            4, 3, 1, 1,  # knight-conflict with cell 1
            3, 4, 2, 1,
            1, 2, 4, 3,
        ],
        "expect_candidates": False,
    },
]