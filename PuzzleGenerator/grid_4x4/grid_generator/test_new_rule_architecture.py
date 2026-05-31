from __future__ import annotations

import json
import random

from TEST.grid_4x4.rule_logic.common.common_base_4x4 import ALL_VALUES_MASK, CELL_COUNT
from TEST.grid_4x4.rule_logic.common.common_solver_4x4 import count_solutions
from TEST.grid_4x4.rule_logic.common.rule_api import ActiveRule
from TEST.grid_4x4.rule_logic.common.rule_registry import load_registered_rules


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _jsonable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def _default_arch_test_cases():
    return [
        {
            "name": "fallback_smoke",
            "solution_grid": [
                1, 2, 3, 4,
                3, 4, 1, 2,
                2, 1, 4, 3,
                4, 3, 2, 1,
            ],
            "expect_candidates": None,
        }
    ]


def _iter_test_cases(module):
    return getattr(module, "ARCH_TEST_CASES", _default_arch_test_cases())


def _test_registered_rule(registered) -> None:
    module = registered.module
    spec = registered.rule_spec
    rule_name = registered.rule_name

    print(f"[test] Regel: {rule_name}")
    rng = random.Random(0)

    for case in _iter_test_cases(module):
        case_name = case["name"]
        solution_grid = case["solution_grid"]
        expect_candidates = case.get("expect_candidates", None)

        candidates = spec.generate_candidates(solution_grid, rng=rng, max_candidates=5)
        _assert(isinstance(candidates, list), f"{rule_name}/{case_name}: generate_candidates returnerte ikke liste")

        if expect_candidates is True:
            _assert(len(candidates) > 0, f"{rule_name}/{case_name}: forventet minst én kandidat")
        elif expect_candidates is False:
            _assert(len(candidates) == 0, f"{rule_name}/{case_name}: forventet ingen kandidater")

        for idx, candidate in enumerate(candidates):
            _assert(spec.solution_satisfies(solution_grid, candidate), f"{rule_name}/{case_name}: kandidat {idx} tilfredsstiller ikke løsningen")

            key = spec.candidate_key(candidate)
            desc = spec.describe(candidate)
            js = spec.to_jsonable(candidate)

            _assert(isinstance(key, tuple), f"{rule_name}/{case_name}: candidate_key er ikke tuple")
            _assert(isinstance(desc, str), f"{rule_name}/{case_name}: describe er ikke str")
            _assert(_jsonable(js), f"{rule_name}/{case_name}: to_jsonable er ikke JSON-serialiserbar")

            domains = [ALL_VALUES_MASK] * CELL_COUNT
            ok = spec.propagate(domains, candidate, ctx=None)
            _assert(isinstance(ok, bool), f"{rule_name}/{case_name}: propagate returnerte ikke bool")
            _assert(len(domains) == CELL_COUNT, f"{rule_name}/{case_name}: propagate endret domain-lengde")

            sol_count = count_solutions(solution_grid, active_rules=[ActiveRule(spec, candidate)], limit=2)
            _assert(sol_count == 1, f"{rule_name}/{case_name}: generisk solver ga {sol_count} i stedet for 1 på komplett gyldig grid")

    print(f"[test] {rule_name}: OK")


def main() -> None:
    registered = load_registered_rules()
    _assert(registered, "Fant ingen _new_arch_-regler")

    print(f"[test] Fant {len(registered)} _new_arch_-regler")
    for _, reg in sorted(registered.items(), key=lambda kv: kv[0]):
        _test_registered_rule(reg)

    print("[test] Alle _new_arch_-regler besto den generelle arkitekturtesten")


if __name__ == "__main__":
    main()