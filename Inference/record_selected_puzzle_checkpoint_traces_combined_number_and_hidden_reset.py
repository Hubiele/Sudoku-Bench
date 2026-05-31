#!/usr/bin/env python3
"""
Record full NCA rollout traces for selected puzzles across all checkpoints.

Place this script in the same folder as testing_test.py and testing_test_with_csv.py.

Purpose
-------
For a small list of selected puzzle indices, the script loads every checkpoint
inside CHECKPOINT_ROOT, runs inference on each selected puzzle, and saves the
entire channel stack after every NCA step. This makes it possible to analyze how
the model's internal state develops across training checkpoints and across NCA
steps.

Expected checkpoint structure
-----------------------------
CHECKPOINT_ROOT/
    step_001000/
        model.eqx
        meta.json
        ...
    step_002000/
        model.eqx
        meta.json
        ...
    ...
    step_100000/
        model.eqx
        meta.json
        ...

Output structure
----------------
OUT_DIR/
    trace_manifest.json
    selected_puzzles.json
    all_metrics.csv
    puzzle_0051/
        puzzle_metadata.json
        givens.npy
        target.npy
        board_mask.npy
        checkpoints.csv
        step_001000/
            trace.npy
            metrics.json
            final_prediction.npy
        step_002000/
            trace.npy
            metrics.json
            final_prediction.npy
        ...
    puzzle_0055/
        ...

trace.npy has shape:
    (saved_nca_steps, 9, 9, channels)

If SAVE_INITIAL_STATE = True and SAVE_EVERY_NCA_STEP = 1, saved_nca_steps will be
NCA_STEPS + 1, because step 0 is the input stack before any NCA update.

Dependencies are the same as the existing training/inference code:
    jax, equinox, numpy
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench")

# Directory containing test_stacks.npy, test_metadata.json, and manifest.json.
TEST_STACK_DIR = PROJECT_ROOT / "StackGenerator" / "stack_test"

# Directory containing step_001000, step_002000, ..., step_100000.
CHECKPOINT_ROOT = Path("/home/daniel/Documents/Skole/Masteroppgave/Cluster/spring_semester/20260507_train_search_LR/checkpoints/warmup_short/")

# Selected puzzle indices, matching the indices from the result CSV.
SELECTED_PUZZLE_INDICES: list[int] = [51, 55, 56, 60, 63, 81, 82, 83, 84, 85]

# NCA rollout length.
NCA_STEPS = 128

# =============================================================================
# Combined intervention experiment
# =============================================================================
# This variant tests whether the model is locked by a combination of:
#   1. hidden-channel state, and
#   2. large number/digit logits.
#
# After the NCA steps listed below, the script can:
#   - scale the number channels by a positive factor, preserving argmax
#   - reset the hidden channels to zero
#
# With NCA_STEPS=128 and intervention step 64:
#
#   run step 1, ..., run step 64,
#   scale number channels and/or reset hidden channels,
#   continue with step 65, ..., step 128.
#
# A positive scale factor preserves the ordering of the digit channels inside
# each cell, so the immediate argmax prediction is unchanged. However, the
# softmax confidence is reduced when the factor is between 0 and 1.
# =============================================================================
APPLY_COMBINED_INTERVENTION_DURING_ROLLOUT = True
INTERVENTION_AFTER_STEPS: list[int] = [64]

SCALE_NUMBER_CHANNELS_ON_INTERVENTION = True
NUMBER_CHANNEL_SCALE_FACTOR = 0.0

RESET_HIDDEN_CHANNELS_ON_INTERVENTION = True

# Number of stochastic inference runs per checkpoint/puzzle.
# Use 1 for deterministic-looking analysis. Increase if you want to study fire-rate randomness.
NUM_TRIALS_PER_PUZZLE = 1

# Random seed used for NCA update keys.
SEED = 0

# Save step 0, i.e. the initial channel stack before any NCA update.
SAVE_INITIAL_STATE = True

# Save every NCA step. Use 1 for full traces. Use 2, 4, 8, ... to reduce storage.
SAVE_EVERY_NCA_STEP = 1

# Store trace arrays as float32, float16, or bfloat16-like float32 fallback.
# float16 reduces disk usage, but may lose precision for later detailed analysis.
TRACE_DTYPE = "float32"  # "float32" or "float16"

# Output directory.
OUT_DIR = Path("selected_puzzle_checkpoint_traces_combined_intervention")

# If True, old files in OUT_DIR are not deleted, but files with same names are overwritten.
OVERWRITE_EXISTING_FILES = True


# =============================================================================
# Project imports
# =============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NCA.NCA_model import (  # type: ignore
    NCA,
    NUMBER_CHANNELS,
    GIVEN_MASK_CHANNELS,
    BOARD_MASK_CHANNELS,
    RULE_CHANNELS,
    HIDDEN_CHANNELS,
    TARGET_CHANNELS,
)
from Train.checkpoints import load_checkpoint  # type: ignore


# =============================================================================
# Dataset helpers
# =============================================================================

def load_test_dataset(test_stack_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    stacks_path = test_stack_dir / "test_stacks.npy"
    metadata_path = test_stack_dir / "test_metadata.json"
    manifest_path = test_stack_dir / "manifest.json"

    if not stacks_path.exists():
        raise FileNotFoundError(f"Could not find: {stacks_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find: {manifest_path}")

    stacks = np.load(stacks_path, mmap_mode=None)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if len(stacks) != len(metadata):
        raise ValueError(
            f"Mismatch between stacks and metadata: {len(stacks)} stacks vs. {len(metadata)} metadata entries."
        )

    return stacks, metadata, manifest


def channel_slices() -> dict[str, slice]:
    given_digit_start = 0
    given_digit_end = NUMBER_CHANNELS

    given_mask_start = given_digit_end
    given_mask_end = given_mask_start + GIVEN_MASK_CHANNELS

    board_mask_start = given_mask_end
    board_mask_end = board_mask_start + BOARD_MASK_CHANNELS

    rule_start = board_mask_end
    rule_end = rule_start + RULE_CHANNELS

    hidden_start = rule_end
    hidden_end = hidden_start + HIDDEN_CHANNELS

    target_start = hidden_end
    target_end = target_start + TARGET_CHANNELS

    return {
        "number_channels": slice(given_digit_start, given_digit_end),
        "given_mask": slice(given_mask_start, given_mask_end),
        "board_mask": slice(board_mask_start, board_mask_end),
        "rule_channels": slice(rule_start, rule_end),
        "hidden_channels": slice(hidden_start, hidden_end),
        "target_channels": slice(target_start, target_end),
    }


def slice_to_list(s: slice) -> list[int]:
    return [int(s.start), int(s.stop)]


def masks_from_stack(stack_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slices = channel_slices()
    given_mask = stack_hwc[..., slices["given_mask"]][..., 0] > 0.5
    board_mask = stack_hwc[..., slices["board_mask"]][..., 0] > 0.5
    predict_mask = board_mask & (~given_mask)
    return given_mask, board_mask, predict_mask


def active_window_from_stack(stack_hwc: np.ndarray) -> tuple[int, int, int]:
    _, board_mask, _ = masks_from_stack(stack_hwc)
    coords = np.argwhere(board_mask)

    if len(coords) == 0:
        return 0, 0, 0

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    side = int(max(r1 - r0 + 1, c1 - c0 + 1))
    return side, int(r0), int(c0)


def decode_given_grid(stack_hwc: np.ndarray) -> np.ndarray:
    slices = channel_slices()
    number_logits = stack_hwc[..., slices["number_channels"]]
    given_mask, board_mask, _ = masks_from_stack(stack_hwc)
    digits = np.argmax(number_logits, axis=-1) + 1
    return np.where(given_mask & board_mask, digits, 0).astype(np.int16)


def decode_target_grid(stack_hwc: np.ndarray) -> np.ndarray:
    slices = channel_slices()
    target_onehot = stack_hwc[..., slices["target_channels"]]
    return (np.argmax(target_onehot, axis=-1) + 1).astype(np.int16)


def decode_prediction_grid(stack_hwc: np.ndarray) -> np.ndarray:
    slices = channel_slices()
    logits = stack_hwc[..., slices["number_channels"]]
    return (np.argmax(logits, axis=-1) + 1).astype(np.int16)


def crop_to_active(grid_9x9: np.ndarray, reference_stack_hwc: np.ndarray) -> np.ndarray:
    side, r0, c0 = active_window_from_stack(reference_stack_hwc)
    return grid_9x9[r0:r0 + side, c0:c0 + side]


# =============================================================================
# Checkpoint helpers
# =============================================================================

def parse_step_from_checkpoint(path: Path) -> int:
    meta_path = path / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "step" in meta:
                return int(meta["step"])
        except Exception:
            pass

    match = re.search(r"step[_-]?(\d+)", path.name)
    if match:
        return int(match.group(1))

    return -1


def find_checkpoints(checkpoint_root: Path) -> list[Path]:
    checkpoint_root = checkpoint_root.resolve()

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    # If the user accidentally points directly to a checkpoint folder.
    if (checkpoint_root / "model.eqx").exists() and (checkpoint_root / "meta.json").exists():
        return [checkpoint_root]

    # Preferred expected structure: immediate step_XXXXXX directories.
    candidates = [
        p for p in checkpoint_root.iterdir()
        if p.is_dir() and p.name.startswith("step_")
        and (p / "model.eqx").exists()
        and (p / "meta.json").exists()
    ]

    # Fallback: recursive search.
    if not candidates:
        candidates = [
            p.parent for p in checkpoint_root.rglob("model.eqx")
            if (p.parent / "meta.json").exists()
        ]

    checkpoints = sorted(set(candidates), key=lambda p: (parse_step_from_checkpoint(p), str(p)))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {checkpoint_root}")

    return checkpoints


# =============================================================================
# NCA rollout
# =============================================================================

def make_rollout_trace_fn(nca_steps: int, save_initial_state: bool):
    """JIT-compiled rollout that returns final state and all intermediate states.

    This variant applies a combined intervention after selected NCA steps.
    The intervention can scale number channels and reset hidden channels.

    The intervention is applied *after* the model update for that step. For
    example, if INTERVENTION_AFTER_STEPS=[64], the model runs step 64 first,
    then applies the intervention, and then continues with step 65.

    The saved trace includes the intervened state at that step.
    """
    nca_steps = int(nca_steps)
    slices = channel_slices()
    number_slice = slices["number_channels"]
    hidden_slice = slices["hidden_channels"]

    intervention_steps = jnp.asarray(INTERVENTION_AFTER_STEPS, dtype=jnp.int32)
    number_scale_factor = jnp.asarray(NUMBER_CHANNEL_SCALE_FACTOR, dtype=jnp.float32)

    @eqx.filter_jit
    def rollout_trace(model: NCA, key: jax.Array, initial_stack: jnp.ndarray):
        def maybe_apply_intervention(stack_t: jnp.ndarray, step_number: jnp.ndarray) -> jnp.ndarray:
            if not APPLY_COMBINED_INTERVENTION_DURING_ROLLOUT:
                return stack_t

            should_intervene = jnp.any(step_number == intervention_steps)

            def do_intervention(x):
                y = x

                if SCALE_NUMBER_CHANNELS_ON_INTERVENTION:
                    # Positive scaling preserves argmax within each cell, but
                    # reduces softmax confidence when the factor is between 0 and 1.
                    y = y.at[..., number_slice].multiply(number_scale_factor)

                if RESET_HIDDEN_CHANNELS_ON_INTERVENTION:
                    y = y.at[..., hidden_slice].set(0.0)

                return y

            return jax.lax.cond(should_intervene, do_intervention, lambda x: x, stack_t)

        def step_fn(stack_t, inputs):
            step_key, step_number = inputs
            stack_next = model(step_key, stack_t)
            stack_next = maybe_apply_intervention(stack_next, step_number)
            return stack_next, stack_next

        step_keys = jax.random.split(key, nca_steps)
        step_numbers = jnp.arange(1, nca_steps + 1, dtype=jnp.int32)
        final_stack, states = jax.lax.scan(step_fn, initial_stack, (step_keys, step_numbers))

        if save_initial_state:
            states = jnp.concatenate([initial_stack[None, ...], states], axis=0)

        return final_stack, states

    return rollout_trace

def masked_cross_entropy(stack_hwc: jnp.ndarray) -> jnp.ndarray:
    slices = channel_slices()

    logits = stack_hwc[..., slices["number_channels"]]
    target_onehot = stack_hwc[..., slices["target_channels"]]
    given_mask = stack_hwc[..., slices["given_mask"]]
    board_mask = stack_hwc[..., slices["board_mask"]]
    predict_mask = board_mask * (1.0 - given_mask)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(target_onehot * log_probs, axis=-1, keepdims=True)
    numerator = jnp.sum(ce * predict_mask)
    denom = jnp.maximum(jnp.sum(predict_mask), 1.0)

    return numerator / denom


def evaluate_output(output_stack_hwc: np.ndarray, reference_stack_hwc: np.ndarray) -> dict[str, Any]:
    pred = decode_prediction_grid(output_stack_hwc)
    target = decode_target_grid(reference_stack_hwc)
    _, board_mask, predict_mask = masks_from_stack(reference_stack_hwc)

    board_total = int(board_mask.sum())
    predict_total = int(predict_mask.sum())

    board_wrong = int(np.sum((pred != target) & board_mask))
    predict_wrong = int(np.sum((pred != target) & predict_mask))

    return {
        "loss": float(masked_cross_entropy(jnp.asarray(output_stack_hwc))),
        "board_total": board_total,
        "board_wrong": board_wrong,
        "board_correct": board_total - board_wrong,
        "predict_total": predict_total,
        "predict_wrong": predict_wrong,
        "predict_correct": predict_total - predict_wrong,
        "solved": bool(board_wrong == 0),
    }


# =============================================================================
# Saving
# =============================================================================

def trace_dtype_np() -> np.dtype:
    if TRACE_DTYPE == "float16":
        return np.dtype(np.float16)
    if TRACE_DTYPE == "float32":
        return np.dtype(np.float32)
    raise ValueError("TRACE_DTYPE must be 'float32' or 'float16'.")


def save_trace(
    *,
    trace: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_every = max(int(SAVE_EVERY_NCA_STEP), 1)
    keep = np.arange(0, trace.shape[0], save_every, dtype=np.int32)
    if keep[-1] != trace.shape[0] - 1:
        keep = np.concatenate([keep, np.array([trace.shape[0] - 1], dtype=np.int32)])

    trace_saved = trace[keep].astype(trace_dtype_np(), copy=False)

    # .npy is simple and fast to load later.
    np.save(out_path, trace_saved)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_static_puzzle_files(puzzle_dir: Path, stack_hwc: np.ndarray, meta: dict[str, Any]) -> None:
    puzzle_dir.mkdir(parents=True, exist_ok=True)

    given_grid = decode_given_grid(stack_hwc)
    target_grid = decode_target_grid(stack_hwc)
    given_mask, board_mask, predict_mask = masks_from_stack(stack_hwc)

    np.save(puzzle_dir / "givens_9x9.npy", given_grid)
    np.save(puzzle_dir / "target_9x9.npy", target_grid)
    np.save(puzzle_dir / "board_mask_9x9.npy", board_mask.astype(np.bool_))
    np.save(puzzle_dir / "predict_mask_9x9.npy", predict_mask.astype(np.bool_))

    np.save(puzzle_dir / "givens_active.npy", crop_to_active(given_grid, stack_hwc))
    np.save(puzzle_dir / "target_active.npy", crop_to_active(target_grid, stack_hwc))

    write_json(puzzle_dir / "puzzle_metadata.json", meta)


def make_global_manifest(
    *,
    checkpoints: list[Path],
    manifest: dict[str, Any],
    selected_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    slices = channel_slices()
    saved_step_indices = list(range(0, NCA_STEPS + (1 if SAVE_INITIAL_STATE else 0), max(int(SAVE_EVERY_NCA_STEP), 1)))
    if saved_step_indices[-1] != NCA_STEPS:
        saved_step_indices.append(NCA_STEPS)

    return {
        "description": "NCA rollout traces for selected puzzles across checkpoints.",
        "project_root": str(PROJECT_ROOT),
        "test_stack_dir": str(TEST_STACK_DIR),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "out_dir": str(OUT_DIR),
        "selected_puzzle_indices_1based": SELECTED_PUZZLE_INDICES,
        "selected_puzzles": selected_entries,
        "nca_steps": NCA_STEPS,
        "save_initial_state": SAVE_INITIAL_STATE,
        "save_every_nca_step": SAVE_EVERY_NCA_STEP,
        "saved_step_indices": saved_step_indices,
        "trace_dtype": TRACE_DTYPE,
        "trace_shape_description": "[saved_steps, 9, 9, channels]",
        "num_trials_per_puzzle": NUM_TRIALS_PER_PUZZLE,
        "apply_combined_intervention_during_rollout": APPLY_COMBINED_INTERVENTION_DURING_ROLLOUT,
        "intervention_after_steps": INTERVENTION_AFTER_STEPS,
        "scale_number_channels_on_intervention": SCALE_NUMBER_CHANNELS_ON_INTERVENTION,
        "number_channel_scale_factor": NUMBER_CHANNEL_SCALE_FACTOR,
        "reset_hidden_channels_on_intervention": RESET_HIDDEN_CHANNELS_ON_INTERVENTION,
        "channel_slices": {name: slice_to_list(s) for name, s in slices.items()},
        "model_channel_constants": {
            "NUMBER_CHANNELS": NUMBER_CHANNELS,
            "GIVEN_MASK_CHANNELS": GIVEN_MASK_CHANNELS,
            "BOARD_MASK_CHANNELS": BOARD_MASK_CHANNELS,
            "RULE_CHANNELS": RULE_CHANNELS,
            "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
            "TARGET_CHANNELS": TARGET_CHANNELS,
        },
        "dataset_manifest": manifest,
        "checkpoints": [
            {
                "step": parse_step_from_checkpoint(p),
                "path": str(p),
                "name": p.name,
            }
            for p in checkpoints
        ],
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Configuration")
    print("-" * 80)
    print(f"PROJECT_ROOT            = {PROJECT_ROOT}")
    print(f"TEST_STACK_DIR          = {TEST_STACK_DIR}")
    print(f"CHECKPOINT_ROOT         = {CHECKPOINT_ROOT}")
    print(f"SELECTED_PUZZLE_INDICES = {SELECTED_PUZZLE_INDICES}")
    print(f"NCA_STEPS               = {NCA_STEPS}")
    print(f"COMBINED_INTERVENTION   = {APPLY_COMBINED_INTERVENTION_DURING_ROLLOUT}")
    print(f"INTERVENTION_AFTER      = {INTERVENTION_AFTER_STEPS}")
    print(f"SCALE_NUMBER_CHANNELS   = {SCALE_NUMBER_CHANNELS_ON_INTERVENTION}")
    print(f"NUMBER_SCALE_FACTOR     = {NUMBER_CHANNEL_SCALE_FACTOR}")
    print(f"RESET_HIDDEN_CHANNELS   = {RESET_HIDDEN_CHANNELS_ON_INTERVENTION}")
    print(f"NUM_TRIALS_PER_PUZZLE   = {NUM_TRIALS_PER_PUZZLE}")
    print(f"SAVE_INITIAL_STATE      = {SAVE_INITIAL_STATE}")
    print(f"SAVE_EVERY_NCA_STEP     = {SAVE_EVERY_NCA_STEP}")
    print(f"TRACE_DTYPE             = {TRACE_DTYPE}")
    print(f"OUT_DIR                 = {OUT_DIR}")
    print()

    stacks, metadata, dataset_manifest = load_test_dataset(TEST_STACK_DIR)
    checkpoints = find_checkpoints(CHECKPOINT_ROOT)

    print(f"Loaded {len(stacks)} test stacks.")
    print(f"Found {len(checkpoints)} checkpoints.")
    print(f"First checkpoint: {checkpoints[0].name}")
    print(f"Last checkpoint:  {checkpoints[-1].name}")
    print()

    selected_zero_based: list[int] = []
    selected_entries: list[dict[str, Any]] = []

    for puzzle_idx in SELECTED_PUZZLE_INDICES:
        arr_idx = int(puzzle_idx) - 1
        if arr_idx < 0 or arr_idx >= len(stacks):
            raise IndexError(f"Selected puzzle index {puzzle_idx} is outside 1..{len(stacks)}")

        stack = stacks[arr_idx]
        meta = metadata[arr_idx]
        side, r0, c0 = active_window_from_stack(stack)

        selected_zero_based.append(arr_idx)
        selected_entries.append({
            "puzzle_idx_1based": int(puzzle_idx),
            "array_index_0based": int(arr_idx),
            "puzzle_id": meta.get("puzzle_id", f"{puzzle_idx:04d}"),
            "variant_id": meta.get("variant_id", ""),
            "side": int(meta.get("side", side)),
            "active_window": {"side": side, "row_offset": r0, "col_offset": c0},
            "source_file": meta.get("source_file", ""),
            "active_rule_names": meta.get("active_rule_names", []),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT_DIR / "trace_manifest.json",
        make_global_manifest(
            checkpoints=checkpoints,
            manifest=dataset_manifest,
            selected_entries=selected_entries,
        ),
    )
    write_json(OUT_DIR / "selected_puzzles.json", selected_entries)

    # Save static files once per puzzle.
    for puzzle_idx, arr_idx, entry in zip(SELECTED_PUZZLE_INDICES, selected_zero_based, selected_entries):
        puzzle_dir = OUT_DIR / f"puzzle_{int(puzzle_idx):04d}"
        save_static_puzzle_files(puzzle_dir, stacks[arr_idx], metadata[arr_idx])

    key = jax.random.PRNGKey(SEED)
    key, model_key = jax.random.split(key)

    # Model template from first selected puzzle.
    example_stack = jnp.asarray(stacks[selected_zero_based[0]])
    model_template = NCA(model_key, example_stack)

    rollout_trace_fn = make_rollout_trace_fn(NCA_STEPS, SAVE_INITIAL_STATE)

    all_metric_rows: list[dict[str, Any]] = []

    for checkpoint_number, ckpt_path in enumerate(checkpoints, start=1):
        ckpt_step = parse_step_from_checkpoint(ckpt_path)
        ckpt_label = f"step_{ckpt_step:06d}" if ckpt_step >= 0 else ckpt_path.name

        print("=" * 100)
        print(f"[{checkpoint_number}/{len(checkpoints)}] {ckpt_label}: {ckpt_path}")
        print("=" * 100)

        model, _, _, _, _ = load_checkpoint(
            ckpt_path,
            model_template=model_template,
            opt_state_template=None,
            pool_state_template=None,
            key_template=None,
        )

        for puzzle_idx, arr_idx, entry in zip(SELECTED_PUZZLE_INDICES, selected_zero_based, selected_entries):
            initial_stack_np = stacks[arr_idx]
            initial_stack = jnp.asarray(initial_stack_np)

            puzzle_dir = OUT_DIR / f"puzzle_{int(puzzle_idx):04d}"
            ckpt_dir = puzzle_dir / ckpt_label

            for trial_idx in range(1, NUM_TRIALS_PER_PUZZLE + 1):
                key, trial_key = jax.random.split(key)

                final_stack, trace = rollout_trace_fn(model, trial_key, initial_stack)
                final_stack_np = np.asarray(jax.device_get(final_stack))
                trace_np = np.asarray(jax.device_get(trace))

                metrics = evaluate_output(final_stack_np, initial_stack_np)

                if NUM_TRIALS_PER_PUZZLE == 1:
                    trace_name = "trace.npy"
                    metrics_name = "metrics.json"
                    pred_name = "final_prediction_9x9.npy"
                    pred_active_name = "final_prediction_active.npy"
                else:
                    trace_name = f"trace_trial_{trial_idx:02d}.npy"
                    metrics_name = f"metrics_trial_{trial_idx:02d}.json"
                    pred_name = f"final_prediction_9x9_trial_{trial_idx:02d}.npy"
                    pred_active_name = f"final_prediction_active_trial_{trial_idx:02d}.npy"

                save_trace(trace=trace_np, out_path=ckpt_dir / trace_name)

                pred_9x9 = decode_prediction_grid(final_stack_np)
                np.save(ckpt_dir / pred_name, pred_9x9)
                np.save(ckpt_dir / pred_active_name, crop_to_active(pred_9x9, initial_stack_np))

                metrics_payload = {
                    **metrics,
                    "checkpoint_step": ckpt_step,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_label": ckpt_label,
                    "puzzle_idx": int(puzzle_idx),
                    "array_index": int(arr_idx),
                    "puzzle_id": entry["puzzle_id"],
                    "variant_id": entry["variant_id"],
                    "side": entry["side"],
                    "trial_idx": trial_idx,
                    "nca_steps": NCA_STEPS,
                    "apply_combined_intervention_during_rollout": APPLY_COMBINED_INTERVENTION_DURING_ROLLOUT,
                    "intervention_after_steps": INTERVENTION_AFTER_STEPS,
                    "scale_number_channels_on_intervention": SCALE_NUMBER_CHANNELS_ON_INTERVENTION,
                    "number_channel_scale_factor": NUMBER_CHANNEL_SCALE_FACTOR,
                    "reset_hidden_channels_on_intervention": RESET_HIDDEN_CHANNELS_ON_INTERVENTION,
                    "trace_file": str((ckpt_dir / trace_name).relative_to(OUT_DIR)),
                }
                write_json(ckpt_dir / metrics_name, metrics_payload)

                all_metric_rows.append(metrics_payload)

                print(
                    f"Puzzle {int(puzzle_idx):04d} | trial {trial_idx:02d} | "
                    f"loss={metrics['loss']:.6f} | "
                    f"board_wrong={metrics['board_wrong']} | "
                    f"predict_wrong={metrics['predict_wrong']} | "
                    f"solved={metrics['solved']}"
                )

    write_csv(OUT_DIR / "all_metrics.csv", all_metric_rows)

    # Also write one checkpoints.csv per puzzle for easy local inspection.
    for puzzle_idx in SELECTED_PUZZLE_INDICES:
        puzzle_rows = [r for r in all_metric_rows if int(r["puzzle_idx"]) == int(puzzle_idx)]
        write_csv(OUT_DIR / f"puzzle_{int(puzzle_idx):04d}" / "checkpoints.csv", puzzle_rows)

    print()
    print("Done.")
    print(f"Trace output directory: {OUT_DIR.resolve()}")
    print(f"Global metrics CSV:     {(OUT_DIR / 'all_metrics.csv').resolve()}")
    print(f"Trace manifest:         {(OUT_DIR / 'trace_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
