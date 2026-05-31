from __future__ import annotations

import csv
import io
import itertools
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.common_base_4x4 import (
        SIDE,
        grid_to_string,
        parse_grid,
    )
    from TEST.grid_4x4.rule_logic.common.common_solver_4x4 import (
        count_solutions,
        find_minimal_unique_subset,
    )
    from TEST.grid_4x4.rule_logic.common.rule_api import ActiveRule
    from TEST.grid_4x4.rule_logic.common.rule_registry import load_registered_rules
except ImportError:
    from common_base_4x4 import (
        SIDE,
        grid_to_string,
        parse_grid,
    )
    from common_solver_4x4 import (
        count_solutions,
        find_minimal_unique_subset,
    )
    from rule_api import ActiveRule
    from rule_registry import load_registered_rules


ZIP_PATH = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench/TEST/grid_4x4/grid_generator/4x4_sudoku.zip")
OUTPUT_DIR = Path("train_generated_puzzle_4x4")
SCHEMA_OUTPUT_NAME = "shared_rule_channel_schema_4x4_new_arch_only.json"

NUM_BOARDS_TO_FIND = 200
RANDOM_SEED = 1
MAX_RULES_TO_ADD = 2
MAX_DATASET_ROWS: int | None = None

LOAD_LOG_EVERY = 200_000
CHECK_LOG_EVERY = 500
RULESET_LOG_ALWAYS = True
DETAILED_LOG_FIRST_N_BOARDS = 5
DETAILED_LOG_ALWAYS = False

DEFAULT_MAX_CANDIDATES_PER_RULE = 12

RULE_MAX_CANDIDATES: dict[str, int] = {
    "anti_knight": 1,
    "counting_circles": 16,
    "differences_count_lines": 12,
    "even_cells": 12,
    "killer_cages": 16,
    "little_killer": 10,
    "mean_baby_snake": 1,
    "odd_cells": 12,
    "partial_kropki": 12,
    "region_sum_lines": 8,
    "renban_lines": 10,
    "sunburn_cells": 8,
    "thermometers": 10,
    "zipper_lines": 8,
}
MAX_COMBINED_CANDIDATE_COMBOS = 96

CANDIDATE_REFRESH_ROUNDS = 2
BOARDS_PER_CANDIDATE_ROUND = 2000

PLANE_SIDE = 9
BOARD_OFFSET = (0, 0)

REGISTERED_RULES = load_registered_rules()
AVAILABLE_RULES = sorted(REGISTERED_RULES.keys())


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _should_detail_log(board_index_in_search: int) -> bool:
    if DETAILED_LOG_ALWAYS:
        return True
    return board_index_in_search <= DETAILED_LOG_FIRST_N_BOARDS


def _format_stage(stage: str, seconds: float) -> str:
    return f"{stage}={seconds:.3f}s"


# Read puzzle and solution pairs from one CSV file inside the zip archive.
def _read_csv_from_zip(zf: zipfile.ZipFile, name: str, delimiter: str) -> list[tuple[str, str]]:
    print(f"[load] Reading {name} with delimiter={delimiter!r}")
    t0 = time.perf_counter()
    raw = zf.read(name).decode("utf-8")
    rows: list[tuple[str, str]] = []
    reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
    for i, row in enumerate(reader, start=1):
        puzzle = (row.get("Puzzle") or row.get("puzzle") or "").strip()
        solution = (row.get("Solution") or row.get("solution") or "").strip()
        if not puzzle or not solution:
            continue
        rows.append((puzzle, solution))
        if LOAD_LOG_EVERY and i % LOAD_LOG_EVERY == 0:
            elapsed = time.perf_counter() - t0
            print(f"[load] {name}: {i} rows read | elapsed={_format_seconds(elapsed)}")

    elapsed = time.perf_counter() - t0
    print(f"[load] Done with {name}: {len(rows)} rows | elapsed={_format_seconds(elapsed)}")
    return rows


def load_dataset_from_zip(zip_path: Path) -> list[tuple[str, str]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find zip file: {zip_path}")

    print(f"[load] Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        comma_rows = _read_csv_from_zip(zf, "4x4_sudoku_unique_puzzles.csv", delimiter=",")
        tab_rows = _read_csv_from_zip(zf, "4x4_sudoku_unique_solution.csv", delimiter="\t")

    if not comma_rows:
        raise ValueError("Could not find any rows in comma-separated CSV")
    if not tab_rows:
        raise ValueError("Could not find any rows in tab-separated CSV")

    if comma_rows == tab_rows:
        print("[load] CSV check: the files are identical.")
        rows = comma_rows
    else:
        comma_set = set(comma_rows)
        tab_set = set(tab_rows)
        if tab_set.issubset(comma_set):
            print(
                "[load] CSV check: the tab-separated file is a subset of the comma-separated file. "
                "The generator therefore uses the full comma-separated dataset."
            )
            rows = comma_rows
        else:
            raise ValueError("The CSV files are inconsistent in a way the generator cannot handle automatically.")

    if MAX_DATASET_ROWS is not None:
        rows = rows[:MAX_DATASET_ROWS]
        print(f"[load] MAX_DATASET_ROWS active: using only the first {len(rows)} rows")

    return rows


def grid16_to_rows(flat: str) -> list[str]:
    return [flat[i:i + SIDE] for i in range(0, len(flat), SIDE)]


def print_grid(label: str, flat: str) -> None:
    print(label)
    for row in grid16_to_rows(flat):
        print("  " + " ".join(row))


def _format_rule_result(result: dict) -> str:
    parts = [" + ".join(result["rule_names"])]
    for rule_name in result["rule_names"]:
        candidate = result["rule_candidates"][rule_name]
        spec = REGISTERED_RULES[rule_name].rule_spec
        parts.append(spec.describe(candidate))
    return " | ".join(parts)


def _build_active_rules_from_mapping(rule_candidates: dict[str, Any]) -> tuple[ActiveRule, ...]:
    return tuple(
        ActiveRule(spec=REGISTERED_RULES[rule_name].rule_spec, candidate=rule_candidates[rule_name])
        for rule_name in sorted(rule_candidates)
    )


def _candidate_combo_sort_key(reduced_count: int, rule_candidates: dict[str, Any]) -> tuple:
    per_rule = []
    for rule_name in sorted(rule_candidates):
        spec = REGISTERED_RULES[rule_name].rule_spec
        per_rule.append((rule_name, spec.candidate_key(rule_candidates[rule_name])))
    return (reduced_count, tuple(per_rule))


# Compute the minimal givens needed when only the standard Sudoku rule is used.
def evaluate_standard_baseline(puzzle_grid: list[int]) -> dict:
    std_grid, std_min_count = find_minimal_unique_subset(
        puzzle_grid,
        active_rules=(),
    )
    return {"standard_min_grid": std_grid, "standard_min_count": std_min_count}


def ensure_standard_result(board: dict) -> dict:
    if board.get("standard_result") is None:
        board["standard_result"] = evaluate_standard_baseline(board["puzzle_grid"])
    return board["standard_result"]


def _max_candidates_for_rule(rule_name: str) -> int:
    return RULE_MAX_CANDIDATES.get(rule_name, DEFAULT_MAX_CANDIDATES_PER_RULE)


def _candidate_lists_for_rule_set(
    rule_names: list[str],
    *,
    solution_grid: list[int],
    rng: random.Random,
) -> list[tuple[str, list[Any]]] | None:
    out: list[tuple[str, list[Any]]] = []
    for rule_name in rule_names:
        spec = REGISTERED_RULES[rule_name].rule_spec
        limit = _max_candidates_for_rule(rule_name)
        candidates = list(spec.generate_candidates(solution_grid, rng=rng, max_candidates=limit))
        if not candidates:
            return None
        out.append((rule_name, candidates))
    return out


def _iter_candidate_combos(
    ordered_rule_names: list[str],
    ordered_candidate_lists: list[list[Any]],
    *,
    rng: random.Random,
):
    total = 1
    for lst in ordered_candidate_lists:
        total *= len(lst)

    if total <= MAX_COMBINED_CANDIDATE_COMBOS:
        for combo in itertools.product(*ordered_candidate_lists):
            yield {rule_name: candidate for rule_name, candidate in zip(ordered_rule_names, combo)}
        return

    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = MAX_COMBINED_CANDIDATE_COMBOS * 10

    while len(seen) < MAX_COMBINED_CANDIDATE_COMBOS and attempts < max_attempts:
        attempts += 1
        idx_tuple = tuple(rng.randrange(len(lst)) for lst in ordered_candidate_lists)
        if idx_tuple in seen:
            continue
        seen.add(idx_tuple)
        yield {
            rule_name: ordered_candidate_lists[i][idx_tuple[i]]
            for i, rule_name in enumerate(ordered_rule_names)
        }


def evaluate_combined_rules(
    rule_names: list[str],
    *,
    puzzle_grid: list[int],
    solution_grid: list[int],
    standard_result: dict,
    rng: random.Random,
) -> dict | None:
    std_min_count = standard_result["standard_min_count"]
    std_min_grid = standard_result["standard_min_grid"]

    candidate_lists = _candidate_lists_for_rule_set(
        rule_names,
        solution_grid=solution_grid,
        rng=rng,
    )
    if candidate_lists is None:
        return None

    ordered_rule_names = [name for name, _ in candidate_lists]
    ordered_candidate_lists = [cands for _, cands in candidate_lists]

    best: dict | None = None
    best_key: tuple | None = None

    for rule_candidates in _iter_candidate_combos(
        ordered_rule_names,
        ordered_candidate_lists,
        rng=rng,
    ):
        active_rules = _build_active_rules_from_mapping(rule_candidates)

        reduced_grid, reduced_count = find_minimal_unique_subset(
            puzzle_grid,
            active_rules=active_rules,
        )

        result = {
            "rule_names": sorted(rule_candidates.keys()),
            "rule_candidates": rule_candidates,
            "reduced_grid": reduced_grid,
            "reduced_count": reduced_count,
            "standard_min_grid": std_min_grid,
            "standard_min_count": std_min_count,
            "improvement_vs_standard_min": std_min_count - reduced_count,
        }

        candidate_key = _candidate_combo_sort_key(reduced_count, rule_candidates)
        if best is None or candidate_key < best_key:
            best = result
            best_key = candidate_key

    return best


def _without_rule(selected: dict, removed_rule: str) -> dict:
    new_candidates = {
        rule_name: candidate
        for rule_name, candidate in selected["rule_candidates"].items()
        if rule_name != removed_rule
    }
    return {
        "rule_names": sorted(new_candidates.keys()),
        "rule_candidates": new_candidates,
    }


def _rule_is_necessary_for_reduced_grid(selected: dict, removed_rule: str) -> bool:
    reduced_grid = selected["reduced_grid"]
    ablated = _without_rule(selected, removed_rule)
    active_rules = _build_active_rules_from_mapping(ablated["rule_candidates"])
    sol_count = count_solutions(
        reduced_grid,
        active_rules=active_rules,
        limit=2,
    )
    return sol_count != 1


def evaluate_rule_set(
    rule_names: list[str],
    *,
    puzzle_grid: list[int],
    solution_grid: list[int],
    standard_result: dict,
    rng: random.Random,
) -> tuple[dict | None, dict]:
    selected = evaluate_combined_rules(
        rule_names,
        puzzle_grid=puzzle_grid,
        solution_grid=solution_grid,
        standard_result=standard_result,
        rng=rng,
    )
    if selected is None:
        return None, {"reason": "no_valid_combination"}
    if selected["improvement_vs_standard_min"] <= 0:
        return None, {"reason": "no_positive_improvement", "selected": selected}

    necessity = {}
    if len(rule_names) >= 2:
        for rule_name in rule_names:
            necessity[rule_name] = _rule_is_necessary_for_reduced_grid(selected, rule_name)
        if not all(necessity.values()):
            return None, {
                "reason": "ablation_failed",
                "selected": selected,
                "necessity": necessity,
            }

    return selected, {"reason": "accepted", "necessity": necessity}


def _choose_rule_set(rng: random.Random) -> tuple[str, ...]:
    upper = min(MAX_RULES_TO_ADD, len(AVAILABLE_RULES))
    desired_rule_count = rng.randint(1, upper)
    return tuple(rng.sample(AVAILABLE_RULES, desired_rule_count))


# ----------------------------- dataset helpers -----------------------------


def _digits() -> list[int]:
    return list(range(1, SIDE + 1))


def _grid_to_matrix(grid: list[int]) -> list[list[int]]:
    return [grid[r * SIDE:(r + 1) * SIDE] for r in range(SIDE)]


def _empty_plane() -> list[list[float]]:
    return [[0.0 for _ in range(PLANE_SIDE)] for _ in range(PLANE_SIDE)]


def _place(plane: list[list[float]], coord: tuple[int, int], value: float) -> None:
    rr = coord[0] + BOARD_OFFSET[0]
    cc = coord[1] + BOARD_OFFSET[1]
    plane[rr][cc] = float(value)


def _cell_to_coord(cell: int) -> tuple[int, int]:
    return divmod(cell, SIDE)


def _norm_coord(v: int) -> float:
    return (v + 1) / (SIDE + 1)


def _progress_values(length: int) -> list[float]:
    return [(i + 1) / (length + 1) for i in range(length)]


def _global_active_plane() -> list[list[float]]:
    plane = _empty_plane()
    for r in range(SIDE):
        for c in range(SIDE):
            _place(plane, (r, c), 1.0)
    return plane


def _channel_entry(name: str, index_hint: int, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "index_hint": index_hint,
        "description": description,
    }


def build_schema() -> dict[str, Any]:
    return {
        "schema_name": "shared_rule_channel_schema_4x4_new_arch_only",
        "version": 1,
        "stack_side": SIDE,
        "notes": [
            "Standalone 4x4 schema emitted by dataset_generator.py.",
            "index_hint is the stack channel used by each rule plane.",
            "rule_planes_9x9 contains padded 9x9 planes with the 4x4 board embedded at offset (0, 0).",
        ],
        "rules": {
            "standard_sudoku": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("given_mask", 0, "1.0 on givens in the selected puzzle.")
                ],
            },
            "anti_knight": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("anti_knight_active", 1, "Global activation plane for anti-knight.")
                ],
            },
            "counting_circles": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("counting_circles_active", 2, "1.0 on counting circle cells.")
                ],
            },
            "differences_count_lines": {
                "num_channels": 4,
                "channels": [
                    _channel_entry("differences_count_active", 3, "1.0 on cells in a differences-count line."),
                    _channel_entry("differences_count_progress", 4, "Normalized progress along the line."),
                    _channel_entry("differences_count_anchor_row", 5, "Normalized row code of the line anchor."),
                    _channel_entry("differences_count_anchor_col", 6, "Normalized column code of the line anchor."),
                ],
            },
            "even_cells": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("even_cells_active", 7, "1.0 on cells constrained to even digits.")
                ],
            },
            "killer_cages": {
                "num_channels": 2,
                "channels": [
                    _channel_entry("killer_active", 8, "1.0 on killer cage cells."),
                    _channel_entry("killer_clue_sum", 9, "Cage clue sum repeated on every cage cell."),
                ],
            },
            "little_killer": {
                "num_channels": 2,
                "channels": [
                    _channel_entry("little_killer_progress", 10, "Normalized progress along a little-killer path."),
                    _channel_entry("little_killer_clue_sum", 11, "Little-killer clue sum repeated on the path."),
                ],
            },
            "mean_baby_snake": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("mean_baby_snake_active", 12, "Global activation plane for Mean Baby Snake.")
                ],
            },
            "odd_cells": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("odd_cells_active", 13, "1.0 on cells constrained to odd digits.")
                ],
            },
            "partial_kropki": {
                "num_channels": 6,
                "channels": [
                    _channel_entry("white_kropki_active", 14, "1.0 on cells touched by a white Kropki dot."),
                    _channel_entry("white_kropki_anchor_row", 15, "Normalized row code of the white-dot anchor."),
                    _channel_entry("white_kropki_anchor_col", 16, "Normalized column code of the white-dot anchor."),
                    _channel_entry("black_kropki_active", 17, "1.0 on cells touched by a black Kropki dot."),
                    _channel_entry("black_kropki_anchor_row", 18, "Normalized row code of the black-dot anchor."),
                    _channel_entry("black_kropki_anchor_col", 19, "Normalized column code of the black-dot anchor."),
                ],
            },
            "region_sum_lines": {
                "num_channels": 4,
                "channels": [
                    _channel_entry("region_sum_active", 20, "1.0 on region-sum line cells."),
                    _channel_entry("region_sum_segment_index", 21, "Normalized segment index inside the line."),
                    _channel_entry("region_sum_anchor_row", 22, "Normalized row code of the line anchor."),
                    _channel_entry("region_sum_anchor_col", 23, "Normalized column code of the line anchor."),
                ],
            },
            "renban_lines": {
                "num_channels": 3,
                "channels": [
                    _channel_entry("renban_active", 24, "1.0 on Renban line cells."),
                    _channel_entry("renban_anchor_row", 25, "Normalized row code of the line anchor."),
                    _channel_entry("renban_anchor_col", 26, "Normalized column code of the line anchor."),
                ],
            },
            "sunburn_cells": {
                "num_channels": 1,
                "channels": [
                    _channel_entry("sunburn_active", 27, "1.0 on sunburn cells.")
                ],
            },
            "thermometers": {
                "num_channels": 4,
                "channels": [
                    _channel_entry("thermo_active", 28, "1.0 on thermometer cells."),
                    _channel_entry("thermo_progress", 29, "Normalized progress from bulb to tip."),
                    _channel_entry("thermo_anchor_row", 30, "Normalized row code of the bulb."),
                    _channel_entry("thermo_anchor_col", 31, "Normalized column code of the bulb."),
                ],
            },
            "zipper_lines": {
                "num_channels": 4,
                "channels": [
                    _channel_entry("zipper_active", 32, "1.0 on zipper cells."),
                    _channel_entry("zipper_distance", 33, "Normalized distance from the zipper center."),
                    _channel_entry("zipper_anchor_row", 34, "Normalized row code of the zipper center."),
                    _channel_entry("zipper_anchor_col", 35, "Normalized column code of the zipper center."),
                ],
            },
        },
    }


def _selected_channel_specs(schema: dict[str, Any], rule_names: list[str]) -> list[dict[str, Any]]:
    names = ["standard_sudoku"] + list(rule_names)
    out: list[dict[str, Any]] = []
    for rule_name in names:
        out.extend(schema["rules"][rule_name]["channels"])
    out.sort(key=lambda item: item["index_hint"])
    return out


def _encode_standard_only(givens_grid: list[int]) -> dict[str, list[list[float]]]:
    plane = _empty_plane()
    for cell, value in enumerate(givens_grid):
        if value != 0:
            _place(plane, _cell_to_coord(cell), 1.0)
    return {"given_mask": plane}


def _encode_anti_knight(_: Any) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    return {"active": True}, {"anti_knight_active": _global_active_plane()}


def _encode_mean_baby_snake(_: Any) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    return {"active": True}, {"mean_baby_snake_active": _global_active_plane()}


def _encode_cell_set(candidate: Any, active_name: str) -> tuple[Any, dict[str, list[list[float]]]]:
    plane = _empty_plane()
    coords = []
    for cell in candidate.cells:
        coord = _cell_to_coord(cell)
        coords.append([coord[0], coord[1]])
        _place(plane, coord, 1.0)
    return coords, {active_name: plane}


def _encode_counting_circles(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_cell_set(candidate, "counting_circles_active")


def _encode_even_cells(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_cell_set(candidate, "even_cells_active")


def _encode_odd_cells(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_cell_set(candidate, "odd_cells_active")


def _encode_sunburn_cells(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    payload = []
    plane = _empty_plane()
    for cell in candidate.cells:
        coord = _cell_to_coord(cell)
        payload.append({"cells": [[coord[0], coord[1]]], "anchor": [coord[0], coord[1]]})
        _place(plane, coord, 1.0)
    return payload, {"sunburn_active": plane}


def _encode_killer_cages(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    active = _empty_plane()
    clue_sum = _empty_plane()
    cells_payload = []
    for cell in candidate.cells:
        coord = _cell_to_coord(cell)
        cells_payload.append([coord[0], coord[1]])
        _place(active, coord, 1.0)
        _place(clue_sum, coord, float(candidate.clue_sum))
    payload = [{"cells": cells_payload, "clue_sum": int(candidate.clue_sum), "anchor": cells_payload[0]}]
    return payload, {"killer_active": active, "killer_clue_sum": clue_sum}


def _encode_little_killer(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    progress = _empty_plane()
    clue_sum = _empty_plane()
    payload = []
    for clue in candidate.clues:
        prog_vals = _progress_values(len(clue.cells))
        cells_payload = []
        for idx, cell in enumerate(clue.cells):
            coord = _cell_to_coord(cell)
            cells_payload.append([coord[0], coord[1]])
            _place(progress, coord, prog_vals[idx])
            _place(clue_sum, coord, float(clue.clue_sum))
        payload.append(
            {
                "cells": cells_payload,
                "clue_sum": int(clue.clue_sum),
                "outside_coord": [int(clue.outside_coord[0]), int(clue.outside_coord[1])],
                "direction": clue.direction,
                "anchor": cells_payload[0],
            }
        )
    return payload, {"little_killer_progress": progress, "little_killer_clue_sum": clue_sum}


def _encode_path_rule(
    lines: tuple[tuple[int, ...], ...],
    *,
    active_name: str | None = None,
    progress_name: str | None = None,
    anchor_row_name: str | None = None,
    anchor_col_name: str | None = None,
    distance_name: str | None = None,
    segment_name: str | None = None,
    center_anchor: bool = False,
    use_progress: bool = False,
    use_distance: bool = False,
    use_segment: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    planes: dict[str, list[list[float]]] = {}
    for name in [active_name, progress_name, anchor_row_name, anchor_col_name, distance_name, segment_name]:
        if name is not None:
            planes[name] = _empty_plane()

    payload = []
    for line in lines:
        coords = [_cell_to_coord(cell) for cell in line]
        anchor = coords[len(coords) // 2] if center_anchor else coords[0]

        if use_progress:
            aux_vals = _progress_values(len(line))
        elif use_distance:
            center = len(line) // 2
            aux_vals = [abs(i - center) / (center + 1 if center + 1 > 0 else 1) for i in range(len(line))]
        elif use_segment:
            box_map: dict[tuple[int, int], int] = {}
            aux_vals = []
            next_idx = 1
            for r, c in coords:
                box = (r // 2, c // 2)
                if box not in box_map:
                    box_map[box] = next_idx
                    next_idx += 1
                aux_vals.append(box_map[box] / next_idx)
        else:
            aux_vals = [0.0] * len(line)

        for i, coord in enumerate(coords):
            if active_name is not None:
                _place(planes[active_name], coord, 1.0)
            if progress_name is not None:
                _place(planes[progress_name], coord, aux_vals[i] if use_progress else 0.0)
            if distance_name is not None:
                _place(planes[distance_name], coord, aux_vals[i] if use_distance else 0.0)
            if segment_name is not None:
                _place(planes[segment_name], coord, aux_vals[i] if use_segment else 0.0)
            if anchor_row_name is not None:
                _place(planes[anchor_row_name], coord, _norm_coord(anchor[0]))
            if anchor_col_name is not None:
                _place(planes[anchor_col_name], coord, _norm_coord(anchor[1]))

        payload.append({"cells": [[r, c] for r, c in coords], "anchor": [anchor[0], anchor[1]]})

    return payload, planes


def _encode_differences_count_lines(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_path_rule(
        candidate.lines,
        active_name="differences_count_active",
        progress_name="differences_count_progress",
        anchor_row_name="differences_count_anchor_row",
        anchor_col_name="differences_count_anchor_col",
        use_progress=True,
    )


def _encode_renban_lines(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_path_rule(
        candidate.lines,
        active_name="renban_active",
        anchor_row_name="renban_anchor_row",
        anchor_col_name="renban_anchor_col",
    )


def _encode_thermometers(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_path_rule(
        candidate.lines,
        active_name="thermo_active",
        progress_name="thermo_progress",
        anchor_row_name="thermo_anchor_row",
        anchor_col_name="thermo_anchor_col",
        use_progress=True,
    )


def _encode_zipper_lines(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_path_rule(
        candidate.lines,
        active_name="zipper_active",
        distance_name="zipper_distance",
        anchor_row_name="zipper_anchor_row",
        anchor_col_name="zipper_anchor_col",
        center_anchor=True,
        use_distance=True,
    )


def _encode_region_sum_lines(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    return _encode_path_rule(
        candidate.lines,
        active_name="region_sum_active",
        segment_name="region_sum_segment_index",
        anchor_row_name="region_sum_anchor_row",
        anchor_col_name="region_sum_anchor_col",
        use_segment=True,
    )


def _encode_partial_kropki(candidate: Any) -> tuple[Any, dict[str, list[list[float]]]]:
    white_active = _empty_plane()
    white_r = _empty_plane()
    white_c = _empty_plane()
    black_active = _empty_plane()
    black_r = _empty_plane()
    black_c = _empty_plane()

    white_payload = []
    for a, b in candidate.white_dots:
        ca, cb = _cell_to_coord(a), _cell_to_coord(b)
        anchor = min(ca, cb)
        white_payload.append({"cells": [[ca[0], ca[1]], [cb[0], cb[1]]], "anchor": [anchor[0], anchor[1]]})
        for coord in [ca, cb]:
            _place(white_active, coord, 1.0)
            _place(white_r, coord, _norm_coord(anchor[0]))
            _place(white_c, coord, _norm_coord(anchor[1]))

    black_payload = []
    for a, b in candidate.black_dots:
        ca, cb = _cell_to_coord(a), _cell_to_coord(b)
        anchor = min(ca, cb)
        black_payload.append({"cells": [[ca[0], ca[1]], [cb[0], cb[1]]], "anchor": [anchor[0], anchor[1]]})
        for coord in [ca, cb]:
            _place(black_active, coord, 1.0)
            _place(black_r, coord, _norm_coord(anchor[0]))
            _place(black_c, coord, _norm_coord(anchor[1]))

    payload = {"white_dots": white_payload, "black_dots": black_payload}
    planes = {
        "white_kropki_active": white_active,
        "white_kropki_anchor_row": white_r,
        "white_kropki_anchor_col": white_c,
        "black_kropki_active": black_active,
        "black_kropki_anchor_row": black_r,
        "black_kropki_anchor_col": black_c,
    }
    return payload, planes


RULE_ENCODERS = {
    "anti_knight": _encode_anti_knight,
    "counting_circles": _encode_counting_circles,
    "differences_count_lines": _encode_differences_count_lines,
    "even_cells": _encode_even_cells,
    "killer_cages": _encode_killer_cages,
    "little_killer": _encode_little_killer,
    "mean_baby_snake": _encode_mean_baby_snake,
    "odd_cells": _encode_odd_cells,
    "partial_kropki": _encode_partial_kropki,
    "region_sum_lines": _encode_region_sum_lines,
    "renban_lines": _encode_renban_lines,
    "sunburn_cells": _encode_sunburn_cells,
    "thermometers": _encode_thermometers,
    "zipper_lines": _encode_zipper_lines,
}


def build_dataset_record(
    puzzle_id: int,
    board: dict,
    selected: dict,
    meta: dict | None,
    schema: dict[str, Any],
) -> dict[str, Any]:
    givens_grid = selected["reduced_grid"]
    solution_grid = board["solution_grid"]

    record: dict[str, Any] = {
        "variant_id": f"generated_puzzle_{puzzle_id:05d}",
        "active_rule_names": ["standard_sudoku_4x4"] + selected["rule_names"],
        "side": SIDE,
        "digits": _digits(),
        "board_offset": [BOARD_OFFSET[0], BOARD_OFFSET[1]],
        "solution": _grid_to_matrix(solution_grid),
        "givens": _grid_to_matrix(givens_grid),
    }

    rule_planes = _encode_standard_only(givens_grid)
    for rule_name in selected["rule_names"]:
        payload, planes = RULE_ENCODERS[rule_name](selected["rule_candidates"][rule_name])
        record[rule_name] = payload
        rule_planes.update(planes)

    selected_channels = _selected_channel_specs(schema, selected["rule_names"])
    record["rule_channel_names"] = [item["name"] for item in selected_channels if item["name"] != "given_mask"]
    record["rule_planes_9x9"] = {
        item["name"]: rule_planes[item["name"]]
        for item in selected_channels
        if item["name"] != "given_mask"
    }
    record["metadata"] = {
        "chosen_rule_set": selected["rule_names"],
        "selected_rule_text": _format_rule_result(selected),
        "standard_min_count": selected["standard_min_count"],
        "selected_min_count": selected["reduced_count"],
        "improvement_vs_standard_min": selected["improvement_vs_standard_min"],
        "original_puzzle": _grid_to_matrix(board["puzzle_grid"]),
        "minimal_standard_only_subset": _grid_to_matrix(selected["standard_min_grid"]),
        "necessity": {} if meta is None else meta.get("necessity", {}),
    }
    return record


def write_dataset_record(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['variant_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def write_schema_file(schema: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SCHEMA_OUTPUT_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


# ----------------------------- search logic -----------------------------


# Search for a board where the selected extra rules reduce the number of givens.
def _find_board_for_rule_set(
    remaining_dataset: list[dict],
    rule_set: tuple[str, ...],
    rng: random.Random,
    *,
    search_started_at: float,
) -> tuple[int | None, dict | None, dict | None, int]:
    order = list(range(len(remaining_dataset)))
    rng.shuffle(order)
    boards_checked = 0

    if RULESET_LOG_ALWAYS:
        print(f"[search] Starting search for rule set {list(rule_set)} | candidate_boards={len(order)}")

    if BOARDS_PER_CANDIDATE_ROUND <= 0:
        raise ValueError("BOARDS_PER_CANDIDATE_ROUND must be >= 1")
    if CANDIDATE_REFRESH_ROUNDS <= 0:
        raise ValueError("CANDIDATE_REFRESH_ROUNDS must be >= 1")

    offset = 0
    for round_idx in range(CANDIDATE_REFRESH_ROUNDS):
        if offset >= len(order):
            break

        round_seed = rng.randrange(2**63)
        round_rng = random.Random(round_seed)
        round_start = offset
        round_end = min(offset + BOARDS_PER_CANDIDATE_ROUND, len(order))
        round_order = order[round_start:round_end]
        offset = round_end

        if RULESET_LOG_ALWAYS:
            print(
                f"[search] Rule set {list(rule_set)} | candidate round {round_idx + 1}/{CANDIDATE_REFRESH_ROUNDS} "
                f"| boards_in_round={len(round_order)}"
            )

        for idx in round_order:
            board = remaining_dataset[idx]
            boards_checked += 1
            detail_log = _should_detail_log(boards_checked)

            if detail_log:
                print(
                    f"[search][board {boards_checked}] Starting evaluation | "
                    f"puzzle={board['puzzle_flat']} | rules={list(rule_set)}"
                )

            stage_t0 = time.perf_counter()
            standard_result = ensure_standard_result(board)
            baseline_dt = time.perf_counter() - stage_t0

            if detail_log:
                print(
                    f"[search][board {boards_checked}] Standard baseline done | "
                    f"{_format_stage('baseline', baseline_dt)} | "
                    f"standard_min={standard_result['standard_min_count']}"
                )

            stage_t0 = time.perf_counter()
            board_rng = random.Random(round_rng.randrange(2**63))
            selected, meta = evaluate_rule_set(
                list(rule_set),
                puzzle_grid=board["puzzle_grid"],
                solution_grid=board["solution_grid"],
                standard_result=standard_result,
                rng=board_rng,
            )
            rules_dt = time.perf_counter() - stage_t0

            if detail_log:
                print(f"[search][board {boards_checked}] Rule evaluation done | {_format_stage('rules', rules_dt)}")
                if selected is None:
                    print(f"[search][board {boards_checked}] No valid/useful combination found for this board.")
                    if meta is not None and meta.get("reason") == "ablation_failed":
                        print(f"[search][board {boards_checked}] Ablation failed | necessity={meta.get('necessity', {})}")
                else:
                    print(
                        f"[search][board {boards_checked}] Candidate found | "
                        f"selected={_format_rule_result(selected)} | "
                        f"delta={selected['improvement_vs_standard_min']}"
                    )
                    if meta is not None and meta.get("necessity"):
                        print(f"[search][board {boards_checked}] Necessity={meta['necessity']}")

            if CHECK_LOG_EVERY and boards_checked % CHECK_LOG_EVERY == 0:
                elapsed = time.perf_counter() - search_started_at
                print(
                    f"[search] Rule set {list(rule_set)} | checked {boards_checked} boards in this round "
                    f"| total elapsed={_format_seconds(elapsed)}"
                )

            if selected is not None:
                elapsed = time.perf_counter() - search_started_at
                print(
                    f"[search] Match for rule set {list(rule_set)} after {boards_checked} boards "
                    f"| elapsed={_format_seconds(elapsed)}"
                )
                return idx, selected, meta, boards_checked

        if RULESET_LOG_ALWAYS and round_idx + 1 < CANDIDATE_REFRESH_ROUNDS and offset < len(order):
            elapsed = time.perf_counter() - search_started_at
            print(
                f"[search] No match for rule set {list(rule_set)} in candidate round {round_idx + 1} "
                f"after {boards_checked} boards | elapsed={_format_seconds(elapsed)}. Drawing new candidates."
            )

    elapsed = time.perf_counter() - search_started_at
    print(
        f"[search] No match for rule set {list(rule_set)} after {boards_checked} boards "
        f"| elapsed={_format_seconds(elapsed)}"
    )
    return None, None, None, boards_checked


# Generate dataset records until the requested number of boards is found.
def main() -> None:
    t0 = time.perf_counter()
    dataset = load_dataset_from_zip(ZIP_PATH)
    print(f"Read {len(dataset)} puzzles from {ZIP_PATH}")
    print(f"[init] Loaded {len(AVAILABLE_RULES)} rules from registry: {AVAILABLE_RULES}")
    print(f"[init] Candidate caps per rule: {RULE_MAX_CANDIDATES}")
    print(f"[init] MAX_COMBINED_CANDIDATE_COMBOS={MAX_COMBINED_CANDIDATE_COMBOS}")
    print(
        f"[init] Candidate rounds per rule set: {CANDIDATE_REFRESH_ROUNDS} "
        f"| boards per round: {BOARDS_PER_CANDIDATE_ROUND}"
    )
    print(f"[init] Output folder: {OUTPUT_DIR}")

    if NUM_BOARDS_TO_FIND <= 0:
        raise ValueError("NUM_BOARDS_TO_FIND must be >= 1")
    if MAX_RULES_TO_ADD <= 0:
        raise ValueError("MAX_RULES_TO_ADD must be >= 1")
    if not AVAILABLE_RULES:
        raise ValueError("No available rules in registry")

    rng = random.Random(RANDOM_SEED)
    schema = build_schema()
    schema_path = write_schema_file(schema, OUTPUT_DIR)
    print(f"[init] Wrote schema: {schema_path}")

    print("[init] Parsing boards to internal representation ...")
    parse_t0 = time.perf_counter()
    remaining_dataset: list[dict] = []
    for i, (puzzle_flat, solution_flat) in enumerate(dataset, start=1):
        puzzle_grid = parse_grid(puzzle_flat)
        solution_grid = parse_grid(solution_flat)
        remaining_dataset.append(
            {
                "puzzle_flat": puzzle_flat,
                "solution_flat": solution_flat,
                "puzzle_grid": puzzle_grid,
                "solution_grid": solution_grid,
                "standard_result": None,
            }
        )
        if LOAD_LOG_EVERY and i % LOAD_LOG_EVERY == 0:
            elapsed = time.perf_counter() - parse_t0
            print(f"[init] Parsed {i} boards | elapsed={_format_seconds(elapsed)}")

    print(
        f"[init] Done parsing {len(remaining_dataset)} boards "
        f"| elapsed={_format_seconds(time.perf_counter() - parse_t0)}"
    )
    print("[init] Note: the standard baseline is computed lazily for each board when it is tested.")

    successful = 0
    ruleset_draws = 0
    boards_checked_total = 0
    selected_rule_histogram: dict[str, int] = {}

    while successful < NUM_BOARDS_TO_FIND and remaining_dataset:
        rule_set = _choose_rule_set(rng)
        ruleset_draws += 1
        print("\n" + "-" * 80)
        print(f"[main] Draw {ruleset_draws}: selected rule set {list(rule_set)}")
        print(f"[main] Remaining boards: {len(remaining_dataset)}")

        selected_index, selected, meta, boards_checked = _find_board_for_rule_set(
            remaining_dataset,
            rule_set,
            rng,
            search_started_at=t0,
        )
        boards_checked_total += boards_checked

        if selected_index is None or selected is None:
            print("[main] Could not find a suitable board for this rule set. Drawing a new rule set.")
            continue

        board = remaining_dataset.pop(selected_index)
        successful += 1
        rule_key = "+".join(selected["rule_names"])
        selected_rule_histogram[rule_key] = selected_rule_histogram.get(rule_key, 0) + 1

        record = build_dataset_record(successful, board, selected, meta, schema)
        out_path = write_dataset_record(record, OUTPUT_DIR)

        print("\n" + "=" * 80)
        print(f"[FOUND {successful}/{NUM_BOARDS_TO_FIND}]")
        print(f"Chosen rule set:            {list(rule_set)}")
        print(f"Selected rules:             {_format_rule_result(selected)}")
        print(f"Minimal givens, standard:   {selected['standard_min_count']}")
        print(f"Minimal givens, selected:   {selected['reduced_count']}")
        print(f"Reduction vs standard-min:  {selected['improvement_vs_standard_min']}")
        if meta is not None and meta.get("necessity"):
            print(f"Rule necessity on reduced grid: {meta['necessity']}")
        print(f"Wrote JSON:                 {out_path}")

        print_grid("Original puzzle:", board["puzzle_flat"])
        print_grid("Solution:", board["solution_flat"])
        print_grid("Minimal standard-only subset:", grid_to_string(selected["standard_min_grid"]))
        print_grid("Minimal subset with selected rules:", grid_to_string(selected["reduced_grid"]))

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Rule-set draws attempted:    {ruleset_draws}")
    print(f"Boards checked total:        {boards_checked_total}")
    print(f"Boards found / files saved:  {successful}")
    print(f"Selected rule histogram:     {selected_rule_histogram}")
    print(f"Remaining unused boards:     {len(remaining_dataset)}")
    print(f"Elapsed time (seconds):      {elapsed:.3f}")

    if successful < NUM_BOARDS_TO_FIND:
        print("Note: the generator found fewer boards than requested before the search became too difficult or the dataset was exhausted.")


if __name__ == "__main__":
    main()