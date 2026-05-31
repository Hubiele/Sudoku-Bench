from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
PROJECT_ROOT = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench")
TEST_STACK_DIR = PROJECT_ROOT / "StackGenerator" / "stack_test"
CHECKPOINT_PATH = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench/Train/checkpoints/step_097000")

NUM_TRIALS_PER_PUZZLE = 1
NCA_STEPS = 30
SEED = 0

# Name of the CSV file saved for each test run.
CSV_FILENAME = "step_097000_step30.csv"

# Set to None to run all test puzzles.
MAX_PUZZLES: int | None = None


# ------------------------------------------------------------
# Import project code after PROJECT_ROOT is known.
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
def load_test_dataset(test_stack_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    # The test dataset consists of stacks, metadata, and a manifest.
    stacks_path = test_stack_dir / "test_stacks.npy"
    metadata_path = test_stack_dir / "test_metadata.json"
    manifest_path = test_stack_dir / "manifest.json"

    if not stacks_path.exists():
        raise FileNotFoundError(f"Could not find test_stacks.npy: {stacks_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find test_metadata.json: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest.json: {manifest_path}")

    stacks = np.load(stacks_path, mmap_mode=None)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if len(stacks) != len(metadata):
        raise ValueError(
            f"Mismatch between number of stacks ({len(stacks)}) and metadata entries ({len(metadata)})."
        )

    return stacks, metadata, manifest


# ------------------------------------------------------------
# Channel helpers
# ------------------------------------------------------------
def get_channel_slices() -> dict[str, slice]:
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
        "given_digits": slice(given_digit_start, given_digit_end),
        "given_mask": slice(given_mask_start, given_mask_end),
        "board_mask": slice(board_mask_start, board_mask_end),
        "rule": slice(rule_start, rule_end),
        "hidden": slice(hidden_start, hidden_end),
        "target": slice(target_start, target_end),
    }


# ------------------------------------------------------------
# Evaluation / rollout
# ------------------------------------------------------------
def make_rollout_fn(nca_steps: int):
    nca_steps = int(nca_steps)

    @eqx.filter_jit
    def rollout(model, key, stack):
        def one_step(grid, step_key):
            return model(step_key, grid), None

        step_keys = jax.random.split(key, nca_steps)
        grid, _ = jax.lax.scan(one_step, stack, step_keys)
        return grid

    return rollout


def masked_cross_entropy_like_validation(full_stack_hwc: jnp.ndarray) -> jnp.ndarray:
    slices = get_channel_slices()

    # Compute loss only on cells that should be predicted.
    logits = full_stack_hwc[..., slices["given_digits"]]
    target_onehot = full_stack_hwc[..., slices["target"]]
    given_mask = full_stack_hwc[..., slices["given_mask"]]
    board_mask = full_stack_hwc[..., slices["board_mask"]]
    predict_mask = board_mask * (1.0 - given_mask)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce_per_cell = -jnp.sum(target_onehot * log_probs, axis=-1, keepdims=True)
    masked = ce_per_cell * predict_mask
    denom = jnp.maximum(jnp.sum(predict_mask), 1.0)
    return jnp.sum(masked) / denom


# ------------------------------------------------------------
# Decoding / metrics
# ------------------------------------------------------------
def decode_prediction_digits(full_stack_hwc: np.ndarray) -> np.ndarray:
    logits = full_stack_hwc[..., :NUMBER_CHANNELS]
    pred_idx = np.argmax(logits, axis=-1)
    return pred_idx + 1


def target_digits_from_stack(full_stack_hwc: np.ndarray) -> np.ndarray:
    slices = get_channel_slices()
    target_onehot = full_stack_hwc[..., slices["target"]]
    target_idx = np.argmax(target_onehot, axis=-1)
    return target_idx + 1


def masks_from_stack(full_stack_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slices = get_channel_slices()
    given_mask = full_stack_hwc[..., slices["given_mask"]][..., 0] > 0.5
    board_mask = full_stack_hwc[..., slices["board_mask"]][..., 0] > 0.5
    predict_mask = board_mask & (~given_mask)
    return given_mask, board_mask, predict_mask


def evaluate_trial(output_stack_hwc: np.ndarray, reference_stack_hwc: np.ndarray) -> dict[str, Any]:
    pred_digits = decode_prediction_digits(output_stack_hwc)
    target_digits = target_digits_from_stack(reference_stack_hwc)
    given_mask, board_mask, predict_mask = masks_from_stack(reference_stack_hwc)

    # Board metrics include all active cells. Prediction metrics exclude givens.
    board_correct_mask = (pred_digits == target_digits) & board_mask
    predict_correct_mask = (pred_digits == target_digits) & predict_mask

    board_total = int(board_mask.sum())
    predict_total = int(predict_mask.sum())

    board_correct = int(board_correct_mask.sum())
    board_wrong = board_total - board_correct

    predict_correct = int(predict_correct_mask.sum())
    predict_wrong = predict_total - predict_correct

    solved = bool(board_wrong == 0)

    loss_value = float(masked_cross_entropy_like_validation(jnp.asarray(output_stack_hwc)))

    return {
        "loss": loss_value,
        "board_total": board_total,
        "board_correct": board_correct,
        "board_wrong": board_wrong,
        "predict_total": predict_total,
        "predict_correct": predict_correct,
        "predict_wrong": predict_wrong,
        "solved": solved,
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.array([float(t["loss"]) for t in trials], dtype=np.float64)
    board_correct = np.array([int(t["board_correct"]) for t in trials], dtype=np.int32)
    board_wrong = np.array([int(t["board_wrong"]) for t in trials], dtype=np.int32)
    predict_correct = np.array([int(t["predict_correct"]) for t in trials], dtype=np.int32)
    predict_wrong = np.array([int(t["predict_wrong"]) for t in trials], dtype=np.int32)
    solved = np.array([bool(t["solved"]) for t in trials], dtype=bool)

    summary = {
        "runs": len(trials),
        "loss_mean": float(losses.mean()),
        "loss_min": float(losses.min()),
        "loss_max": float(losses.max()),
        "board_correct_mean": float(board_correct.mean()),
        "board_wrong_mean": float(board_wrong.mean()),
        "board_correct_best": int(board_correct.max()),
        "board_wrong_best": int(board_wrong.min()),
        "predict_correct_mean": float(predict_correct.mean()),
        "predict_wrong_mean": float(predict_wrong.mean()),
        "predict_correct_best": int(predict_correct.max()),
        "predict_wrong_best": int(predict_wrong.min()),
        "solved_count": int(solved.sum()),
        "solve_rate": float(solved.mean()),
    }
    return summary


def format_puzzle_id_list(puzzle_ids: list[str]) -> str:
    if not puzzle_ids:
        return "None"
    return ", ".join(puzzle_ids)


def make_csv_row(
    *,
    puzzle_idx: int,
    num_puzzles: int,
    puzzle_id: str,
    variant_id: Any,
    side: int,
    puzzle_trials: list[dict[str, Any]],
    summary: dict[str, Any],
    puzzle_solved: bool,
    manifest: dict[str, Any],
    ckpt_meta: Any,
) -> dict[str, Any]:
    """Create one CSV row for one puzzle.

    The row contains the same relevant information as the terminal output for
    the puzzle. Trial results are also saved as JSON, so no information is lost
    when NUM_TRIALS_PER_PUZZLE > 1.
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "test_stack_dir": str(TEST_STACK_DIR),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "num_trials_per_puzzle": NUM_TRIALS_PER_PUZZLE,
        "nca_steps": NCA_STEPS,
        "seed": SEED,
        "max_puzzles": MAX_PUZZLES if MAX_PUZZLES is not None else "None",
        "manifest_total_channels": manifest.get("total_channels"),
        "checkpoint_meta": json.dumps(ckpt_meta, ensure_ascii=False),
        "puzzle_idx": puzzle_idx,
        "num_puzzles": num_puzzles,
        "puzzle_id": puzzle_id,
        "variant_id": variant_id,
        "side": side,
        "trials_json": json.dumps(puzzle_trials, ensure_ascii=False),
        "runs": summary["runs"],
        "loss_mean": summary["loss_mean"],
        "loss_min": summary["loss_min"],
        "loss_max": summary["loss_max"],
        "board_correct_mean": summary["board_correct_mean"],
        "board_wrong_mean": summary["board_wrong_mean"],
        "board_correct_best": summary["board_correct_best"],
        "board_wrong_best": summary["board_wrong_best"],
        "predict_correct_mean": summary["predict_correct_mean"],
        "predict_wrong_mean": summary["predict_wrong_mean"],
        "predict_correct_best": summary["predict_correct_best"],
        "predict_wrong_best": summary["predict_wrong_best"],
        "solved_count": summary["solved_count"],
        "solve_rate": summary["solve_rate"],
        "puzzle_solved_any_trial": puzzle_solved,
    }


def write_csv_results(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    print(f"PROJECT_ROOT         = {PROJECT_ROOT}")
    print(f"TEST_STACK_DIR       = {TEST_STACK_DIR}")
    print(f"CHECKPOINT_PATH      = {CHECKPOINT_PATH}")
    print(f"NUM_TRIALS_PER_PUZZLE= {NUM_TRIALS_PER_PUZZLE}")
    print(f"NCA_STEPS            = {NCA_STEPS}")
    print(f"SEED                 = {SEED}")
    print(f"CSV_FILENAME         = {CSV_FILENAME}")
    print()

    stacks, metadata, manifest = load_test_dataset(TEST_STACK_DIR)

    print("Test dataset loaded.")
    print(f"Number of test puzzles: {len(stacks)}")
    print(f"Manifest total_channels     : {manifest.get('total_channels')}")
    print()

    if MAX_PUZZLES is not None:
        stacks = stacks[:MAX_PUZZLES]
        metadata = metadata[:MAX_PUZZLES]
        print(f"MAX_PUZZLES active: running only the first {len(stacks)} puzzles.")
        print()

    key = jax.random.PRNGKey(SEED)
    key, model_key = jax.random.split(key)

    first_stack = jnp.asarray(stacks[0])
    model_template = NCA(model_key, first_stack)

    model, _, _, _, ckpt_meta = load_checkpoint(
        CHECKPOINT_PATH,
        model_template=model_template,
        opt_state_template=None,
        pool_state_template=None,
        key_template=None,
    )

    print("Checkpoint loaded.")
    print("Checkpoint metadata:")
    print(json.dumps(ckpt_meta, indent=2, ensure_ascii=False))
    print()

    # Compile the rollout function for the selected number of NCA steps.
    rollout_fn = make_rollout_fn(NCA_STEPS)

    grand_trials: list[dict[str, Any]] = []
    trials_by_side: dict[int, list[dict[str, Any]]] = {
        4: [],
        6: [],
        9: [],
    }
    puzzle_results_by_side: dict[int, list[dict[str, Any]]] = {
        4: [],
        6: [],
        9: [],
    }
    csv_rows: list[dict[str, Any]] = []

    for puzzle_idx, (stack_hwc, meta) in enumerate(zip(stacks, metadata), start=1):
        puzzle_id = str(meta.get("puzzle_id", f"{puzzle_idx:04d}"))
        variant_id = meta.get("variant_id", "")
        side = int(meta.get("side", 4))

        print("=" * 100)
        print(f"PUZZLE {puzzle_idx}/{len(stacks)} | puzzle_id={puzzle_id} | variant_id={variant_id}")
        print(f"Side: {side}")
        print("-" * 100)

        puzzle_trials: list[dict[str, Any]] = []
        stack_jax = jnp.asarray(stack_hwc)

        for trial_idx in range(1, NUM_TRIALS_PER_PUZZLE + 1):
            key, trial_key = jax.random.split(key)
            out_stack = rollout_fn(model, trial_key, stack_jax)
            out_stack_np = np.asarray(out_stack)

            trial_metrics = evaluate_trial(out_stack_np, stack_hwc)
            puzzle_trials.append(trial_metrics)
            grand_trials.append(trial_metrics)
            trials_by_side.setdefault(side, []).append(trial_metrics)

            print(
                f"Trial {trial_idx:02d}/{NUM_TRIALS_PER_PUZZLE} | "
                f"loss={trial_metrics['loss']:.6f} | "
                f"board_correct={trial_metrics['board_correct']}/{trial_metrics['board_total']} | "
                f"board_wrong={trial_metrics['board_wrong']} | "
                f"predict_correct={trial_metrics['predict_correct']}/{trial_metrics['predict_total']} | "
                f"predict_wrong={trial_metrics['predict_wrong']} | "
                f"solved={trial_metrics['solved']}"
            )

        summary = summarize_trials(puzzle_trials)
        puzzle_solved = any(bool(t["solved"]) for t in puzzle_trials)

        puzzle_results_by_side.setdefault(side, []).append({
            "puzzle_id": puzzle_id,
            "solved": puzzle_solved,
        })
        csv_rows.append(
            make_csv_row(
                puzzle_idx=puzzle_idx,
                num_puzzles=len(stacks),
                puzzle_id=puzzle_id,
                variant_id=variant_id,
                side=side,
                puzzle_trials=puzzle_trials,
                summary=summary,
                puzzle_solved=puzzle_solved,
                manifest=manifest,
                ckpt_meta=ckpt_meta,
            )
        )

        print("-" * 100)
        print("Puzzle summary:")
        print(
            f"loss_mean={summary['loss_mean']:.6f} | "
            f"loss_min={summary['loss_min']:.6f} | "
            f"loss_max={summary['loss_max']:.6f}"
        )
        print(
            f"board_correct_mean={summary['board_correct_mean']:.2f} | "
            f"board_wrong_mean={summary['board_wrong_mean']:.2f} | "
            f"board_correct_best={summary['board_correct_best']} | "
            f"board_wrong_best={summary['board_wrong_best']}"
        )
        print(
            f"predict_correct_mean={summary['predict_correct_mean']:.2f} | "
            f"predict_wrong_mean={summary['predict_wrong_mean']:.2f} | "
            f"predict_correct_best={summary['predict_correct_best']} | "
            f"predict_wrong_best={summary['predict_wrong_best']}"
        )
        print(
            f"solved_count={summary['solved_count']}/{summary['runs']} | "
            f"solve_rate={summary['solve_rate']:.4f}"
        )
        print(
            f"puzzle_solved_any_trial={puzzle_solved}"
        )
        print()

    if grand_trials:
        overall = summarize_trials(grand_trials)
        print("=" * 100)
        print("TOTAL SUMMARY FOR THE FULL TEST SET")
        print("=" * 100)
        print(
            f"loss_mean={overall['loss_mean']:.6f} | "
            f"loss_min={overall['loss_min']:.6f} | "
            f"loss_max={overall['loss_max']:.6f}"
        )
        print(
            f"board_correct_mean={overall['board_correct_mean']:.2f} | "
            f"board_wrong_mean={overall['board_wrong_mean']:.2f}"
        )
        print(
            f"predict_correct_mean={overall['predict_correct_mean']:.2f} | "
            f"predict_wrong_mean={overall['predict_wrong_mean']:.2f}"
        )
        print(
            f"solved_count={overall['solved_count']}/{overall['runs']} | "
            f"solve_rate={overall['solve_rate']:.4f}"
        )

        print()
        print("=" * 100)
        print("SUMMARY BY GRID SIZE")
        print("=" * 100)

        for grid_side in (4, 6, 9):
            side_trials = trials_by_side.get(grid_side, [])
            side_puzzles = puzzle_results_by_side.get(grid_side, [])

            if not side_trials:
                print(f"{grid_side}x{grid_side}: no puzzles found.")
                print("-" * 100)
                continue

            side_summary = summarize_trials(side_trials)

            solved_ids = sorted(
                [entry["puzzle_id"] for entry in side_puzzles if entry["solved"]]
            )
            unsolved_ids = sorted(
                [entry["puzzle_id"] for entry in side_puzzles if not entry["solved"]]
            )

            print(f"{grid_side}x{grid_side}")
            print(
                f"loss_mean={side_summary['loss_mean']:.6f} | "
                f"loss_min={side_summary['loss_min']:.6f} | "
                f"loss_max={side_summary['loss_max']:.6f}"
            )
            print(
                f"board_correct_mean={side_summary['board_correct_mean']:.2f} | "
                f"board_wrong_mean={side_summary['board_wrong_mean']:.2f}"
            )
            print(
                f"predict_correct_mean={side_summary['predict_correct_mean']:.2f} | "
                f"predict_wrong_mean={side_summary['predict_wrong_mean']:.2f}"
            )
            print(
                f"solved_count={side_summary['solved_count']}/{side_summary['runs']} | "
                f"solve_rate={side_summary['solve_rate']:.4f}"
            )
            print(
                f"Solved puzzles ({len(solved_ids)}): {format_puzzle_id_list(solved_ids)}"
            )
            print(
                f"Unsolved puzzles ({len(unsolved_ids)}): {format_puzzle_id_list(unsolved_ids)}"
            )
            print("-" * 100)

    csv_path = Path(CSV_FILENAME)
    write_csv_results(csv_path, csv_rows)
    print()
    print(f"CSV file saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()