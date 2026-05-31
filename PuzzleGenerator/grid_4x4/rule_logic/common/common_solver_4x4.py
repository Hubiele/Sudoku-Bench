from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        ALL_VALUES_MASK,
        CELL_COUNT,
        CELL_TO_UNITS,
        DIGITS,
        PEERS,
        count_givens,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import ActiveRule
except ImportError:
    from .common_base_4x4 import (
        ALL_VALUES_MASK,
        CELL_COUNT,
        CELL_TO_UNITS,
        DIGITS,
        PEERS,
        count_givens,
    )
    from .rule_api import ActiveRule


@dataclass(frozen=True)
class SolverContext:
    cell_count: int = CELL_COUNT
    digits: tuple[int, ...] = DIGITS
    peers: dict[int, tuple[int, ...]] = field(default_factory=lambda: PEERS)
    cell_to_units: dict[int, list[tuple[int, ...]]] = field(default_factory=lambda: CELL_TO_UNITS)
    all_values_mask: int = ALL_VALUES_MASK


DEFAULT_CONTEXT = SolverContext()


def _mask_to_values(mask: int) -> tuple[int, ...]:
    return tuple(d for d in DIGITS if mask & (1 << d))


def _popcount(mask: int) -> int:
    return mask.bit_count()


def _propagate_plugins(
    domains: list[int],
    active_rules: tuple[ActiveRule, ...],
    *,
    ctx: SolverContext,
) -> bool:
    changed = True
    while changed:
        changed = False
        snapshot = tuple(domains)
        for active_rule in active_rules:
            if not active_rule.spec.propagate(domains, active_rule.candidate, ctx=ctx):
                return False
        if any(mask == 0 for mask in domains):
            return False
        changed = tuple(domains) != snapshot
    return True


def _propagate_all(
    domains: list[int],
    active_rules: tuple[ActiveRule, ...],
    *,
    ctx: SolverContext,
) -> bool:
    return _propagate_plugins(domains, active_rules, ctx=ctx)


def _eliminate(
    domains: list[int],
    cell: int,
    value: int,
    active_rules: tuple[ActiveRule, ...],
    *,
    ctx: SolverContext,
) -> bool:
    bit = 1 << value
    if not (domains[cell] & bit):
        return True

    domains[cell] &= ~bit
    mask = domains[cell]
    if mask == 0:
        return False

    if _popcount(mask) == 1:
        forced_value = _mask_to_values(mask)[0]
        for peer in ctx.peers[cell]:
            if not _eliminate(domains, peer, forced_value, active_rules, ctx=ctx):
                return False

    for unit in ctx.cell_to_units[cell]:
        places = [c for c in unit if domains[c] & bit]
        if not places:
            return False
        if len(places) == 1:
            if not _assign(domains, places[0], value, active_rules, ctx=ctx):
                return False

    return _propagate_all(domains, active_rules, ctx=ctx)


def _assign(
    domains: list[int],
    cell: int,
    value: int,
    active_rules: tuple[ActiveRule, ...],
    *,
    ctx: SolverContext,
) -> bool:
    other_values = [v for v in _mask_to_values(domains[cell]) if v != value]
    for other in other_values:
        if not _eliminate(domains, cell, other, active_rules, ctx=ctx):
            return False
    return True


def _initial_domains(
    givens_grid: list[int],
    active_rules: tuple[ActiveRule, ...],
    *,
    ctx: SolverContext,
) -> list[int] | None:
    domains = [ctx.all_values_mask] * ctx.cell_count
    if not _propagate_all(domains, active_rules, ctx=ctx):
        return None
    for cell, value in enumerate(givens_grid):
        if value:
            if not _assign(domains, cell, int(value), active_rules, ctx=ctx):
                return None
    if not _propagate_all(domains, active_rules, ctx=ctx):
        return None
    return domains


def count_solutions(
    givens_grid: list[int],
    active_rules: Iterable[ActiveRule] = (),
    *,
    limit: int = 2,
    ctx: SolverContext = DEFAULT_CONTEXT,
) -> int:
    active_rules = tuple(active_rules)
    domains = _initial_domains(givens_grid, active_rules, ctx=ctx)
    if domains is None:
        return 0

    def search(domains: list[int], found: int) -> int:
        if found >= limit:
            return found
        unsolved = [c for c in range(ctx.cell_count) if _popcount(domains[c]) > 1]
        if not unsolved:
            return found + 1

        cell = min(unsolved, key=lambda c: _popcount(domains[c]))
        for value in _mask_to_values(domains[cell]):
            next_domains = domains.copy()
            if _assign(next_domains, cell, value, active_rules, ctx=ctx):
                found = search(next_domains, found)
                if found >= limit:
                    return found
        return found

    return search(domains, 0)


def has_unique_solution(
    givens_grid: list[int],
    active_rules: Iterable[ActiveRule] = (),
    *,
    ctx: SolverContext = DEFAULT_CONTEXT,
) -> bool:
    return count_solutions(givens_grid, active_rules, limit=2, ctx=ctx) == 1


def _subset_grid_from_positions(original_grid: list[int], keep_positions: set[int]) -> list[int]:
    return [original_grid[i] if i in keep_positions else 0 for i in range(CELL_COUNT)]


def find_minimal_unique_subset(
    original_puzzle_grid: list[int],
    active_rules: Iterable[ActiveRule] = (),
    *,
    ctx: SolverContext = DEFAULT_CONTEXT,
) -> tuple[list[int], int]:
    given_positions = [i for i, v in enumerate(original_puzzle_grid) if int(v) != 0]
    active_rules = tuple(active_rules)

    for keep_count in range(len(given_positions) + 1):
        for keep_positions in combinations(given_positions, keep_count):
            subset_grid = _subset_grid_from_positions(original_puzzle_grid, set(keep_positions))
            if has_unique_solution(subset_grid, active_rules, ctx=ctx):
                return subset_grid, count_givens(subset_grid)

    return list(original_puzzle_grid), count_givens(original_puzzle_grid)