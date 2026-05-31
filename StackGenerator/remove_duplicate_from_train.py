#!/usr/bin/env python3
"""
Remove training samples that overlap with the test/Sudoku-Bench set.

This script reads overlap_input.csv, finds the training indices listed there,
and writes a new training dataset without those rows.

It is intentionally conservative:
- It does NOT modify train_stacks.npy in place.
- It writes a new .npy file and, if present, a matching metadata .json file.
- It writes a summary file documenting exactly what was removed.

Expected overlap CSV columns:
    train_index_0based
or:
    train_index_1based

Recommended workflow:
    1. Run check_stack_dataset_overlap.py.
    2. Inspect overlap_input.csv.
    3. Run this script.
    4. Update your training path to point to the deduplicated train file.
    5. Optionally rerun check_stack_dataset_overlap.py using the new train file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench")

TRAIN_STACKS_PATH = PROJECT_ROOT / "StackGenerator" / "stack_dataset" / "train_stacks.npy"
TRAIN_METADATA_PATH = TRAIN_STACKS_PATH.with_name("train_metadata.json")

# Point this to the overlap_input.csv produced by check_stack_dataset_overlap.py.
OVERLAP_CSV_PATH = Path("dataset_overlap_check") / "overlap_input.csv"

# Output files. The original train files are left untouched.
OUTPUT_TRAIN_STACKS_PATH = TRAIN_STACKS_PATH.with_name("train_stacks_deduplicated.npy")
OUTPUT_TRAIN_METADATA_PATH = TRAIN_METADATA_PATH.with_name("train_metadata_deduplicated.json")

# Extra documentation outputs.
OUTPUT_SUMMARY_PATH = TRAIN_STACKS_PATH.with_name("train_deduplication_summary.json")
OUTPUT_REMOVED_ROWS_CSV_PATH = TRAIN_STACKS_PATH.with_name("removed_train_duplicates.csv")

# Copy this many rows at a time. Increase if you have enough RAM.
CHUNK_SIZE = 8192

# Safety settings.
DRY_RUN = False
OVERWRITE_OUTPUT = False


# =============================================================================
# Helpers
# =============================================================================

def load_remove_indices(overlap_csv_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load unique 0-based train indices to remove from overlap CSV."""
    if not overlap_csv_path.exists():
        raise FileNotFoundError(f"Could not find overlap CSV: {overlap_csv_path}")

    overlap_df = pd.read_csv(overlap_csv_path)

    if overlap_df.empty:
        return np.array([], dtype=np.int64), overlap_df

    if "train_index_0based" in overlap_df.columns:
        indices = overlap_df["train_index_0based"].dropna().astype(np.int64).to_numpy()
    elif "train_index_1based" in overlap_df.columns:
        indices = overlap_df["train_index_1based"].dropna().astype(np.int64).to_numpy() - 1
    else:
        raise ValueError(
            "The overlap CSV must contain either 'train_index_0based' or 'train_index_1based'. "
            f"Found columns: {list(overlap_df.columns)}"
        )

    unique_indices = np.array(sorted(set(int(i) for i in indices)), dtype=np.int64)
    return unique_indices, overlap_df


def check_output_path(path: Path) -> None:
    if path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Set OVERWRITE_OUTPUT = True if you intentionally want to overwrite it."
        )


def load_metadata(metadata_path: Path, expected_len: int) -> list[dict[str, Any]] | None:
    if not metadata_path.exists():
        print(f"[WARN] Metadata file not found: {metadata_path}")
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError(f"Metadata file must contain a list: {metadata_path}")

    if len(metadata) != expected_len:
        raise ValueError(
            f"Metadata length does not match train stacks: "
            f"{len(metadata)} metadata rows vs. {expected_len} stacks."
        )

    return metadata


def write_removed_rows_csv(
    output_path: Path,
    overlap_df: pd.DataFrame,
    remove_indices: np.ndarray,
) -> None:
    """Write the overlap rows corresponding to removed train indices."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overlap_df.empty:
        overlap_df.to_csv(output_path, index=False)
        return

    if "train_index_0based" in overlap_df.columns:
        removed_df = overlap_df[overlap_df["train_index_0based"].isin(remove_indices)]
    elif "train_index_1based" in overlap_df.columns:
        removed_df = overlap_df[(overlap_df["train_index_1based"] - 1).isin(remove_indices)]
    else:
        removed_df = overlap_df.iloc[0:0]

    removed_df.to_csv(output_path, index=False)


def copy_kept_stacks(
    train_stacks: np.ndarray,
    keep_indices: np.ndarray,
    output_path: Path,
    chunk_size: int,
) -> None:
    """Copy kept rows to a new .npy file without loading the full dataset into RAM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    check_output_path(output_path)

    new_shape = (len(keep_indices),) + tuple(train_stacks.shape[1:])
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=train_stacks.dtype,
        shape=new_shape,
    )

    write_pos = 0
    total = len(keep_indices)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        idx = keep_indices[start:end]

        out[write_pos:write_pos + len(idx)] = train_stacks[idx]
        write_pos += len(idx)

        print(f"Copied {write_pos:,}/{total:,} kept stacks")

    # Ensure data is flushed to disk.
    del out


def write_filtered_metadata(
    metadata: list[dict[str, Any]] | None,
    keep_indices: np.ndarray,
    output_path: Path,
) -> None:
    if metadata is None:
        return

    check_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [metadata[int(i)] for i in keep_indices]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)


def main() -> None:
    print("Configuration")
    print("-" * 80)
    print(f"TRAIN_STACKS_PATH           = {TRAIN_STACKS_PATH}")
    print(f"TRAIN_METADATA_PATH         = {TRAIN_METADATA_PATH}")
    print(f"OVERLAP_CSV_PATH            = {OVERLAP_CSV_PATH}")
    print(f"OUTPUT_TRAIN_STACKS_PATH    = {OUTPUT_TRAIN_STACKS_PATH}")
    print(f"OUTPUT_TRAIN_METADATA_PATH  = {OUTPUT_TRAIN_METADATA_PATH}")
    print(f"OUTPUT_SUMMARY_PATH         = {OUTPUT_SUMMARY_PATH}")
    print(f"OUTPUT_REMOVED_ROWS_CSV_PATH= {OUTPUT_REMOVED_ROWS_CSV_PATH}")
    print(f"CHUNK_SIZE                  = {CHUNK_SIZE}")
    print(f"DRY_RUN                     = {DRY_RUN}")
    print(f"OVERWRITE_OUTPUT            = {OVERWRITE_OUTPUT}")
    print()

    remove_indices, overlap_df = load_remove_indices(OVERLAP_CSV_PATH)

    print(f"Overlap CSV rows: {len(overlap_df):,}")
    print(f"Unique train rows to remove: {len(remove_indices):,}")

    if not TRAIN_STACKS_PATH.exists():
        raise FileNotFoundError(f"Could not find train stacks: {TRAIN_STACKS_PATH}")

    train_stacks = np.load(TRAIN_STACKS_PATH, mmap_mode="r")
    n_train = len(train_stacks)

    print(f"Train stack shape: {train_stacks.shape}")
    print(f"Train stack dtype: {train_stacks.dtype}")

    if len(remove_indices) > 0:
        if remove_indices.min() < 0 or remove_indices.max() >= n_train:
            raise IndexError(
                f"Remove indices are outside training range. "
                f"min={remove_indices.min()}, max={remove_indices.max()}, n_train={n_train}"
            )

    keep_mask = np.ones(n_train, dtype=bool)
    keep_mask[remove_indices] = False
    keep_indices = np.flatnonzero(keep_mask).astype(np.int64)

    print(f"Rows kept:    {len(keep_indices):,}")
    print(f"Rows removed: {n_train - len(keep_indices):,}")
    print()

    metadata = load_metadata(TRAIN_METADATA_PATH, n_train)

    summary = {
        "train_stacks_path": str(TRAIN_STACKS_PATH),
        "train_metadata_path": str(TRAIN_METADATA_PATH),
        "overlap_csv_path": str(OVERLAP_CSV_PATH),
        "output_train_stacks_path": str(OUTPUT_TRAIN_STACKS_PATH),
        "output_train_metadata_path": str(OUTPUT_TRAIN_METADATA_PATH),
        "original_train_shape": list(train_stacks.shape),
        "output_train_shape": [int(len(keep_indices))] + list(train_stacks.shape[1:]),
        "dtype": str(train_stacks.dtype),
        "overlap_csv_rows": int(len(overlap_df)),
        "unique_train_rows_removed": int(len(remove_indices)),
        "removed_train_indices_0based": [int(i) for i in remove_indices],
        "removed_train_indices_1based": [int(i) + 1 for i in remove_indices],
        "dry_run": bool(DRY_RUN),
    }

    if DRY_RUN:
        print("DRY_RUN=True, so no files were written.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    check_output_path(OUTPUT_SUMMARY_PATH)
    check_output_path(OUTPUT_REMOVED_ROWS_CSV_PATH)

    copy_kept_stacks(
        train_stacks=train_stacks,
        keep_indices=keep_indices,
        output_path=OUTPUT_TRAIN_STACKS_PATH,
        chunk_size=CHUNK_SIZE,
    )

    write_filtered_metadata(
        metadata=metadata,
        keep_indices=keep_indices,
        output_path=OUTPUT_TRAIN_METADATA_PATH,
    )

    write_removed_rows_csv(
        output_path=OUTPUT_REMOVED_ROWS_CSV_PATH,
        overlap_df=overlap_df,
        remove_indices=remove_indices,
    )

    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("Done.")
    print(f"Wrote filtered train stacks: {OUTPUT_TRAIN_STACKS_PATH.resolve()}")
    if metadata is not None:
        print(f"Wrote filtered metadata:     {OUTPUT_TRAIN_METADATA_PATH.resolve()}")
    print(f"Wrote removed rows CSV:      {OUTPUT_REMOVED_ROWS_CSV_PATH.resolve()}")
    print(f"Wrote summary:               {OUTPUT_SUMMARY_PATH.resolve()}")
    print()
    print("Recommended next step:")
    print("  Run check_stack_dataset_overlap.py again with TRAIN_STACKS_PATH pointing")
    print("  to the new train_stacks_deduplicated.npy file and verify that")
    print("  overlap_input.csv has zero matches.")


if __name__ == "__main__":
    main()
