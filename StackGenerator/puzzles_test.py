from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ==============================================================================
# Configuration: set the path to the JSON files here.
# ==============================================================================
# This can be any folder that contains the JSON files to use.
# By default, this script uses the "converted" folder from the conversion step.
# Example absolute path: INPUT_JSON_DIR = r"C:\Users\Name\Documents\Sudoku\converted"
INPUT_JSON_DIR = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench/Sudoku-Bench_json_conversion/converted")
# ==============================================================================


# ---------------------------------------------------------------------------
# Load sibling puzzles_train.py dynamically so we reuse the exact same stack logic
# ---------------------------------------------------------------------------

def _load_puzzles_module():
    here = Path(__file__).resolve().parent
    puzzles_path = here / "puzzles_train.py"
    if not puzzles_path.exists():
        raise FileNotFoundError(
            f"Could not find puzzles_train.py next to this script: {puzzles_path}"
        )

    module_name = "_stack_puzzles_module"
    spec = importlib.util.spec_from_file_location(module_name, puzzles_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {puzzles_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


puzzles = _load_puzzles_module()

NUMBER_CHANNELS = puzzles.NUMBER_CHANNELS
GIVEN_MASK_CHANNELS = puzzles.GIVEN_MASK_CHANNELS
BOARD_MASK_CHANNELS = puzzles.BOARD_MASK_CHANNELS
RULE_CHANNELS = puzzles.RULE_CHANNELS
HIDDEN_CHANNELS = puzzles.HIDDEN_CHANNELS
TARGET_CHANNELS = puzzles.TARGET_CHANNELS
STACK_SIDE = puzzles.STACK_SIDE

DEFAULT_LOG_EVERY = puzzles.DEFAULT_LOG_EVERY
DEFAULT_GRID_SIZES = ("4x4", "9x9")


def _normalize_grid_sizes(grid_sizes: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(grid_sizes, str):
        return (grid_sizes,)
    out = tuple(str(x) for x in grid_sizes)
    if not out:
        raise ValueError("grid_sizes cannot be empty.")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_and_save_test_stack_dataset(
    output_dir: str | Path,
    input_dir: str | Path | None = None,  # Allows the input path to be passed directly to the function.
    grid_sizes: str | Sequence[str] = DEFAULT_GRID_SIZES,
    project_root: str | Path | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    if log_every <= 0:
        raise ValueError("log_every must be >= 1.")

    # Select the input folder. Use the global path if no input folder is passed.
    target_input_dir = Path(INPUT_JSON_DIR if input_dir is None else input_dir)

    if not target_input_dir.exists():
        raise FileNotFoundError(f"Could not find input folder for JSON files: {target_input_dir.resolve()}")

    # Use the same schema lookup as the training stack.
    schema_dataset_dir = puzzles._default_dataset_dir()
    schema = puzzles._load_schema(schema_dataset_dir, schema_path)

    grid_sizes = _normalize_grid_sizes(grid_sizes)
    rule_channel_count = puzzles._schema_rule_channel_count(schema)
    channel_map = puzzles._schema_channel_map(schema, rule_channel_count)
    rule_channel_names = puzzles._schema_rule_channel_names(schema, rule_channel_count)
    channel_names = puzzles._all_channel_names(schema, rule_channel_count)

    # Read all JSON files directly from the selected input folder.
    # Each JSON file is treated as one test example.
    variant_files = sorted(list(target_input_dir.glob("*.json")))
    if not variant_files:
        raise ValueError(f"Could not find any test JSON files in folder: {target_input_dir.resolve()}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total_examples = len(variant_files)
    total_channels = len(channel_names)
    stack_shape = (total_examples, STACK_SIDE, STACK_SIDE, total_channels)
    target_shape = (total_examples, STACK_SIDE, STACK_SIDE)

    manifest = {
        "schema_name": schema.get("schema_name"),
        "schema_version": schema.get("version"),
        "schema_path": str(puzzles._schema_path(schema_dataset_dir, schema_path).resolve()),
        "grid_sizes": list(grid_sizes),
        "total_examples": total_examples,
        "number_channels": NUMBER_CHANNELS,
        "given_mask_channels": GIVEN_MASK_CHANNELS,
        "board_mask_channels": BOARD_MASK_CHANNELS,
        "rule_channels": rule_channel_count,
        "hidden_channels": HIDDEN_CHANNELS,
        "target_channels": TARGET_CHANNELS,
        "total_channels": total_channels,
        "stack_side": STACK_SIDE,
        "channel_names": channel_names,
        "rule_channel_names": rule_channel_names,
        "rule_channel_map": channel_map,
        "output_dir": str(output_root.resolve()),
        "test_stacks_file": "test_stacks.npy",
        "test_targets_file": "test_targets.npy",
        "test_metadata_file": "test_metadata.json",
    }

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    stack_path = output_root / "test_stacks.npy"
    target_path = output_root / "test_targets.npy"
    metadata_path = output_root / "test_metadata.json"

    # Use memory-mapped arrays so the dataset can be written without keeping all stacks in RAM.
    stacks_mm = np.lib.format.open_memmap(
        stack_path,
        mode="w+",
        dtype=np.float32,
        shape=stack_shape,
    )
    targets_mm = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.int32,
        shape=target_shape,
    )

    metadata_list: list[dict[str, Any]] = []

    print(f"Saving test dataset to: {output_root.resolve()}")
    print(f"Reading JSON files from: {target_input_dir.resolve()}")
    print(f"Schema: {manifest['schema_path']}")
    print(f"Total number of test files: {total_examples}")
    print("Format: .npy (test_stacks.npy / test_targets.npy)")

    t0 = time.time()
    for i, path in enumerate(variant_files, start=1):
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        # Reuse the same stack-construction logic as the training dataset.
        stack_hwc, target, meta = puzzles._make_example_arrays(
            path,
            record,
            schema,
            channel_names,
            rule_channel_count,
        )
        meta["split"] = "test"

        stacks_mm[i - 1] = stack_hwc
        targets_mm[i - 1] = target
        metadata_list.append(meta)

        if i == 1 or i == total_examples or i % log_every == 0:
            elapsed = time.time() - t0
            avg = elapsed / i
            remaining = avg * (total_examples - i)
            print(
                f"[test] {i}/{total_examples} saved | "
                f"elapsed={puzzles._format_seconds(elapsed)} | "
                f"eta={puzzles._format_seconds(remaining)} | "
                f"last={path.name}"
            )

    stacks_mm.flush()
    targets_mm.flush()

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"Done. Test dataset saved in: {output_root.resolve()}\n")
    return manifest


if __name__ == "__main__":
    output_root = Path(__file__).resolve().parent / "stack_test"
    manifest = build_and_save_test_stack_dataset(
        output_dir=output_root,
        input_dir=INPUT_JSON_DIR,  # Pass in the path from the configuration above.
        grid_sizes=DEFAULT_GRID_SIZES,
        log_every=DEFAULT_LOG_EVERY,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))