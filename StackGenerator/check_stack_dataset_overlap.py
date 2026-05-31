#!/usr/bin/env python3
"""
Check overlap between training stacks and test/Sudoku-Bench stacks.

This script checks whether any puzzles in a test set are identical to puzzles in
the training set. It is intended for .npy stack datasets such as:

    train_stacks.npy
    validation_stacks.npy
    test_stacks.npy

The most useful comparison mode is usually "input", which compares the puzzle
input representation:
    - number channels
    - given mask
    - board mask
    - rule-geometry channels

It excludes hidden channels and target channels. This is usually what you want
when checking whether a test puzzle was present in the training data.

Output:
    OUT_DIR/overlap_input.csv
    OUT_DIR/overlap_full.csv
    OUT_DIR/overlap_summary.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench")

TRAIN_STACKS_PATH = PROJECT_ROOT / "StackGenerator" / "stack_dataset" / "train_stacks.npy"
TEST_STACKS_PATH = PROJECT_ROOT / "StackGenerator" / "stack_test" / "test_stacks.npy"

TRAIN_METADATA_PATH = TRAIN_STACKS_PATH.with_name("train_metadata.json")
TEST_METADATA_PATH = TEST_STACKS_PATH.with_name("test_metadata.json")

OUT_DIR = Path("dataset_overlap_check")

# Recommended:
#   "input" compares puzzle input only, excluding hidden and target channels.
#   "full" compares the full stack.
#   "both" runs both comparisons.
COMPARE_MODE = "input"  # "input", "full", "input_no_rules", "target", or "both"

# Crop to the active board area before hashing. This makes comparison less
# sensitive to unused padding cells outside 4x4/6x6 boards.
CROP_TO_ACTIVE_BOARD = True

# Round float values before hashing. This protects against tiny floating point
# differences in rule geometry. For exact comparison, set ROUND_DECIMALS = None.
ROUND_DECIMALS: int | None = 6

# Only used if import from NCA.NCA_model fails.
FALLBACK_NUMBER_CHANNELS = 9
FALLBACK_GIVEN_MASK_CHANNELS = 1
FALLBACK_BOARD_MASK_CHANNELS = 1
FALLBACK_RULE_CHANNELS = 30
FALLBACK_HIDDEN_CHANNELS = 96
FALLBACK_TARGET_CHANNELS = 9


# =============================================================================
# Channel layout
# =============================================================================

def load_channel_constants() -> dict[str, Any]:
    """Load channel constants from the project if possible; otherwise use fallbacks."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from NCA.NCA_model import (  # type: ignore
            NUMBER_CHANNELS,
            GIVEN_MASK_CHANNELS,
            BOARD_MASK_CHANNELS,
            RULE_CHANNELS,
            HIDDEN_CHANNELS,
            TARGET_CHANNELS,
        )

        return {
            "NUMBER_CHANNELS": int(NUMBER_CHANNELS),
            "GIVEN_MASK_CHANNELS": int(GIVEN_MASK_CHANNELS),
            "BOARD_MASK_CHANNELS": int(BOARD_MASK_CHANNELS),
            "RULE_CHANNELS": int(RULE_CHANNELS),
            "HIDDEN_CHANNELS": int(HIDDEN_CHANNELS),
            "TARGET_CHANNELS": int(TARGET_CHANNELS),
            "source": "NCA.NCA_model",
        }
    except Exception as e:
        print(f"[WARN] Could not import channel constants from NCA.NCA_model: {e}")
        print("[WARN] Using fallback channel constants from this script.")

        return {
            "NUMBER_CHANNELS": int(FALLBACK_NUMBER_CHANNELS),
            "GIVEN_MASK_CHANNELS": int(FALLBACK_GIVEN_MASK_CHANNELS),
            "BOARD_MASK_CHANNELS": int(FALLBACK_BOARD_MASK_CHANNELS),
            "RULE_CHANNELS": int(FALLBACK_RULE_CHANNELS),
            "HIDDEN_CHANNELS": int(FALLBACK_HIDDEN_CHANNELS),
            "TARGET_CHANNELS": int(FALLBACK_TARGET_CHANNELS),
            "source": "fallback",
        }


def channel_slices(constants: dict[str, Any]) -> dict[str, slice]:
    number_start = 0
    number_end = number_start + int(constants["NUMBER_CHANNELS"])

    given_start = number_end
    given_end = given_start + int(constants["GIVEN_MASK_CHANNELS"])

    board_start = given_end
    board_end = board_start + int(constants["BOARD_MASK_CHANNELS"])

    rule_start = board_end
    rule_end = rule_start + int(constants["RULE_CHANNELS"])

    hidden_start = rule_end
    hidden_end = hidden_start + int(constants["HIDDEN_CHANNELS"])

    target_start = hidden_end
    target_end = target_start + int(constants["TARGET_CHANNELS"])

    return {
        "number": slice(number_start, number_end),
        "given": slice(given_start, given_end),
        "board": slice(board_start, board_end),
        "rule": slice(rule_start, rule_end),
        "hidden": slice(hidden_start, hidden_end),
        "target": slice(target_start, target_end),
    }


# =============================================================================
# Helpers
# =============================================================================

def load_metadata(path: Path, n_expected: int) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Metadata file not found: {path}")
        return [{} for _ in range(n_expected)]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[WARN] Metadata file is not a list: {path}")
        return [{} for _ in range(n_expected)]

    out = []
    for i in range(n_expected):
        if i < len(data) and isinstance(data[i], dict):
            out.append(data[i])
        else:
            out.append({})
    return out


def active_crop(stack: np.ndarray, board_slice: slice) -> tuple[slice, slice]:
    board = stack[..., board_slice]
    board_mask = board[..., 0] > 0.5 if board.ndim == 3 else board > 0.5

    coords = np.argwhere(board_mask)
    if len(coords) == 0:
        return slice(0, stack.shape[0]), slice(0, stack.shape[1])

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return slice(int(r0), int(r1) + 1), slice(int(c0), int(c1) + 1)


def make_signature_array(
    stack: np.ndarray,
    slices: dict[str, slice],
    mode: str,
    crop_to_active: bool,
    round_decimals: int | None,
) -> np.ndarray:
    if crop_to_active:
        rs, cs = active_crop(stack, slices["board"])
        stack_view = stack[rs, cs, :]
    else:
        stack_view = stack

    if mode == "full":
        sig = stack_view
    elif mode == "input":
        # Puzzle input: givens/number channels, masks, and rule geometry.
        # Excludes hidden state and target solution.
        sig = np.concatenate(
            [
                stack_view[..., slices["number"]],
                stack_view[..., slices["given"]],
                stack_view[..., slices["board"]],
                stack_view[..., slices["rule"]],
            ],
            axis=-1,
        )
    elif mode == "input_no_rules":
        sig = np.concatenate(
            [
                stack_view[..., slices["number"]],
                stack_view[..., slices["given"]],
                stack_view[..., slices["board"]],
            ],
            axis=-1,
        )
    elif mode == "target":
        sig = np.concatenate(
            [
                stack_view[..., slices["board"]],
                stack_view[..., slices["target"]],
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")

    sig = np.asarray(sig)

    if round_decimals is not None and np.issubdtype(sig.dtype, np.floating):
        sig = np.round(sig.astype(np.float32, copy=False), decimals=round_decimals)

    return np.ascontiguousarray(sig)


def hash_signature(sig: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(sig.shape).encode("utf-8"))
    h.update(str(sig.dtype).encode("utf-8"))
    h.update(sig.tobytes(order="C"))
    return h.hexdigest()


def short_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "puzzle_id",
        "variant_id",
        "side",
        "source_file",
        "source",
        "active_rule_names",
        "rules",
        "name",
    ]
    return {k: meta.get(k, "") for k in keys if k in meta}


def build_train_hash_index(
    train_stacks: np.ndarray,
    slices: dict[str, slice],
    mode: str,
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    n = len(train_stacks)
    print(f"[{mode}] Hashing training set: {n:,} stacks")

    for i in range(n):
        sig = make_signature_array(
            train_stacks[i],
            slices=slices,
            mode=mode,
            crop_to_active=CROP_TO_ACTIVE_BOARD,
            round_decimals=ROUND_DECIMALS,
        )
        digest = hash_signature(sig)
        index.setdefault(digest, []).append(i)

        if (i + 1) % 50000 == 0 or (i + 1) == n:
            print(f"[{mode}]   hashed {i + 1:,}/{n:,}")

    return index


def check_test_against_train(
    test_stacks: np.ndarray,
    test_metadata: list[dict[str, Any]],
    train_hash_index: dict[str, list[int]],
    train_metadata: list[dict[str, Any]],
    slices: dict[str, slice],
    mode: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    n = len(test_stacks)
    print(f"[{mode}] Checking test set: {n:,} stacks")

    for i in range(n):
        sig = make_signature_array(
            test_stacks[i],
            slices=slices,
            mode=mode,
            crop_to_active=CROP_TO_ACTIVE_BOARD,
            round_decimals=ROUND_DECIMALS,
        )
        digest = hash_signature(sig)

        if digest in train_hash_index:
            for train_i in train_hash_index[digest]:
                matches.append(
                    {
                        "mode": mode,
                        "test_index_0based": i,
                        "test_index_1based": i + 1,
                        "train_index_0based": train_i,
                        "train_index_1based": train_i + 1,
                        "hash": digest,
                        "test_metadata": json.dumps(short_meta(test_metadata[i]), ensure_ascii=False),
                        "train_metadata": json.dumps(short_meta(train_metadata[train_i]), ensure_ascii=False),
                    }
                )

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"[{mode}]   checked {i + 1:,}/{n:,}")

    return matches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "test_index_0based",
        "test_index_1based",
        "train_index_0based",
        "train_index_1based",
        "hash",
        "test_metadata",
        "train_metadata",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_mode(
    *,
    mode: str,
    train_stacks: np.ndarray,
    test_stacks: np.ndarray,
    train_metadata: list[dict[str, Any]],
    test_metadata: list[dict[str, Any]],
    slices: dict[str, slice],
) -> list[dict[str, Any]]:
    train_index = build_train_hash_index(train_stacks, slices, mode)
    matches = check_test_against_train(
        test_stacks,
        test_metadata,
        train_index,
        train_metadata,
        slices,
        mode,
    )

    out_csv = OUT_DIR / f"overlap_{mode}.csv"
    write_csv(out_csv, matches)

    print(f"[{mode}] Matches found: {len(matches)}")
    print(f"[{mode}] Wrote: {out_csv.resolve()}")
    print()
    return matches


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    constants = load_channel_constants()
    slices = channel_slices(constants)

    print("Configuration")
    print("-" * 80)
    print(f"TRAIN_STACKS_PATH     = {TRAIN_STACKS_PATH}")
    print(f"TEST_STACKS_PATH      = {TEST_STACKS_PATH}")
    print(f"TRAIN_METADATA_PATH   = {TRAIN_METADATA_PATH}")
    print(f"TEST_METADATA_PATH    = {TEST_METADATA_PATH}")
    print(f"OUT_DIR               = {OUT_DIR}")
    print(f"COMPARE_MODE          = {COMPARE_MODE}")
    print(f"CROP_TO_ACTIVE_BOARD  = {CROP_TO_ACTIVE_BOARD}")
    print(f"ROUND_DECIMALS        = {ROUND_DECIMALS}")
    print(f"channel constants     = {constants}")
    print(f"channel slices        = {slices}")
    print()

    print("Loading stacks with mmap_mode='r'")
    train_stacks = np.load(TRAIN_STACKS_PATH, mmap_mode="r")
    test_stacks = np.load(TEST_STACKS_PATH, mmap_mode="r")

    print(f"Train shape: {train_stacks.shape} | dtype={train_stacks.dtype}")
    print(f"Test shape:  {test_stacks.shape} | dtype={test_stacks.dtype}")
    print()

    train_metadata = load_metadata(TRAIN_METADATA_PATH, len(train_stacks))
    test_metadata = load_metadata(TEST_METADATA_PATH, len(test_stacks))

    modes = ["input", "full"] if COMPARE_MODE == "both" else [COMPARE_MODE]

    summary: dict[str, Any] = {
        "train_stacks_path": str(TRAIN_STACKS_PATH),
        "test_stacks_path": str(TEST_STACKS_PATH),
        "train_shape": list(train_stacks.shape),
        "test_shape": list(test_stacks.shape),
        "compare_mode": COMPARE_MODE,
        "crop_to_active_board": CROP_TO_ACTIVE_BOARD,
        "round_decimals": ROUND_DECIMALS,
        "channel_constants": constants,
        "matches": {},
    }

    for mode in modes:
        matches = run_one_mode(
            mode=mode,
            train_stacks=train_stacks,
            test_stacks=test_stacks,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            slices=slices,
        )
        summary["matches"][mode] = {
            "num_matches": len(matches),
            "unique_test_matches": sorted({m["test_index_1based"] for m in matches}),
            "unique_train_matches_count": len({m["train_index_1based"] for m in matches}),
            "csv": str((OUT_DIR / f"overlap_{mode}.csv").resolve()),
        }

    summary_path = OUT_DIR / "overlap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Summary")
    print("-" * 80)
    print(json.dumps(summary["matches"], indent=2, ensure_ascii=False))
    print(f"Wrote summary: {summary_path.resolve()}")
    print()
    print("Interpretation")
    print("-" * 80)
    print("If overlap_input.csv has matches, a test puzzle input appears in the training set.")
    print("If overlap_full.csv has matches, the entire stack appears in the training set.")
    print("For leakage checks, input-mode matches are usually the most important.")


if __name__ == "__main__":
    main()
