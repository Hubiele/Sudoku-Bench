#!/usr/bin/env python3
"""
Analyze NCA checkpoint traces saved by record_selected_puzzle_checkpoint_traces.py.

The script reads a folder like:

selected_puzzle_checkpoint_traces/
    trace_manifest.json
    selected_puzzles.json
    all_metrics.csv
    puzzle_0051/
        puzzle_metadata.json
        givens_9x9.npy
        target_9x9.npy
        board_mask_9x9.npy
        predict_mask_9x9.npy
        step_001000/trace.npy
        step_002000/trace.npy
        ...

It creates:
    trace_analysis/
        trace_analysis_all.csv
        combined/*.png
        puzzle_0051/*.png
        puzzle_0055/*.png
        ...

Typical usage:
    python analyze_checkpoint_traces.py
    python analyze_checkpoint_traces.py --trace-root /path/to/selected_puzzle_checkpoint_traces
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# =============================================================================
# USER CONFIG
# =============================================================================

TRACE_ROOT_DIR = Path(__file__).resolve().parent / "selected_puzzle_checkpoint_traces"
OUT_DIR_NAME = "trace_analysis"

# Set to e.g. 2, 5, or 10 to analyze fewer checkpoints. None analyzes all.
CHECKPOINT_STRIDE: int | None = None

# Used for prediction snapshot figures.
SNAPSHOT_NCA_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 64, 128]

EPS = 1e-12


# =============================================================================
# Basic helpers
# =============================================================================

def step_number_from_name(name: str) -> int:
    match = re.search(r"step[_-]?(\d+)", name)
    return int(match.group(1)) if match else -1


def safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_manifest(trace_root: Path) -> dict[str, Any]:
    path = trace_root / "trace_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find trace_manifest.json in {trace_root}")
    return safe_json_load(path)


def slice_from_manifest(manifest: dict[str, Any], name: str) -> slice:
    bounds = manifest.get("channel_slices", {}).get(name)
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise KeyError(f"Missing channel slice {name!r} in trace_manifest.json")
    return slice(int(bounds[0]), int(bounds[1]))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), EPS)


def active_crop(arr_9x9: np.ndarray, board_mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(board_mask)
    if len(coords) == 0:
        return arr_9x9
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr_9x9[r0:r1 + 1, c0:c1 + 1]


def puzzle_index_from_dir(puzzle_dir: Path) -> int:
    match = re.search(r"puzzle_(\d+)", puzzle_dir.name)
    return int(match.group(1)) if match else -1


def grid_size_from_board_mask(board_mask: np.ndarray) -> int:
    active_cells = int(np.sum(board_mask))
    n = int(round(math.sqrt(active_cells)))
    if n * n == active_cells:
        return n

    coords = np.argwhere(board_mask)
    if len(coords) == 0:
        return -1
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (r1 - r0) == (c1 - c0):
        return int(r1 - r0 + 1)
    return -1


def find_puzzle_dirs(trace_root: Path) -> list[Path]:
    return sorted(
        [p for p in trace_root.glob("puzzle_*") if p.is_dir()],
        key=lambda p: puzzle_index_from_dir(p),
    )


def find_checkpoint_dirs(puzzle_dir: Path) -> list[Path]:
    dirs = [
        p for p in puzzle_dir.glob("step_*")
        if p.is_dir() and (p / "trace.npy").exists()
    ]
    dirs = sorted(dirs, key=lambda p: (step_number_from_name(p.name), p.name))
    if CHECKPOINT_STRIDE is not None and CHECKPOINT_STRIDE > 1:
        dirs = dirs[::CHECKPOINT_STRIDE]
    return dirs


def nca_steps_for_trace(trace: np.ndarray, manifest: dict[str, Any]) -> list[int]:
    saved = manifest.get("saved_step_indices", [])
    if isinstance(saved, list) and len(saved) >= trace.shape[0]:
        return [int(x) for x in saved[:trace.shape[0]]]

    save_every = int(manifest.get("save_every_nca_step", 1))
    save_initial = bool(manifest.get("save_initial_state", True))
    if save_initial:
        return [i * save_every for i in range(trace.shape[0])]
    return [(i + 1) * save_every for i in range(trace.shape[0])]


def target_digits_from_trace(trace0: np.ndarray, target_slice: slice) -> np.ndarray:
    target_onehot = trace0[..., target_slice]
    return np.argmax(target_onehot, axis=-1) + 1


def target_probability(probs: np.ndarray, target_digits: np.ndarray) -> np.ndarray:
    target_idx = np.clip(target_digits - 1, 0, probs.shape[-1] - 1)
    return np.take_along_axis(probs, target_idx[None, ..., None], axis=-1)[..., 0]


def mean_masked(arr: np.ndarray, mask: np.ndarray) -> float:
    values = arr[mask]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def sum_masked_bool(arr: np.ndarray, mask: np.ndarray) -> int:
    return int(np.sum(arr & mask))


# =============================================================================
# Trace metric extraction
# =============================================================================

def analyze_trace(
    *,
    trace: np.ndarray,
    manifest: dict[str, Any],
    puzzle_idx: int,
    checkpoint_step: int,
    checkpoint_label: str,
    board_mask: np.ndarray,
    predict_mask: np.ndarray,
    grid_size: int,
) -> pd.DataFrame:
    number_slice = slice_from_manifest(manifest, "number_channels")
    hidden_slice = slice_from_manifest(manifest, "hidden_channels")
    target_slice = slice_from_manifest(manifest, "target_channels")

    trace = np.asarray(trace, dtype=np.float32)
    nca_steps = nca_steps_for_trace(trace, manifest)

    logits = trace[..., number_slice]
    hidden = trace[..., hidden_slice] if hidden_slice.stop > hidden_slice.start else None

    probs = softmax(logits, axis=-1)
    pred = np.argmax(logits, axis=-1) + 1
    target = target_digits_from_trace(trace[0], target_slice)

    entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=-1)
    confidence = np.max(probs, axis=-1)
    target_prob = target_probability(probs, target)

    sorted_probs = np.sort(probs, axis=-1)
    margin = sorted_probs[..., -1] - sorted_probs[..., -2]

    rows: list[dict[str, Any]] = []
    prev_pred = None
    prev_correct = None

    for t_idx, nca_step in enumerate(nca_steps):
        pred_t = pred[t_idx]
        correct_t = pred_t == target

        board_wrong = int(np.sum((~correct_t) & board_mask))
        predict_wrong = int(np.sum((~correct_t) & predict_mask))
        solved = bool(board_wrong == 0)

        if prev_pred is None:
            prediction_changes = 0
            corrected_cells = 0
            broken_cells = 0
        else:
            changed = pred_t != prev_pred
            prediction_changes = sum_masked_bool(changed, predict_mask)

            corrected = (~prev_correct) & correct_t
            broken = prev_correct & (~correct_t)
            corrected_cells = sum_masked_bool(corrected, predict_mask)
            broken_cells = sum_masked_bool(broken, predict_mask)

        row = {
            "puzzle_idx": puzzle_idx,
            "grid_size": int(grid_size),
            "checkpoint_step": checkpoint_step,
            "checkpoint_label": checkpoint_label,
            "trace_index": t_idx,
            "nca_step": int(nca_step),
            "board_wrong": board_wrong,
            "predict_wrong": predict_wrong,
            "solved": solved,
            "prediction_changes": prediction_changes,
            "corrected_cells": corrected_cells,
            "broken_cells": broken_cells,
            "mean_entropy_board": mean_masked(entropy[t_idx], board_mask),
            "mean_entropy_predict": mean_masked(entropy[t_idx], predict_mask),
            "mean_confidence_board": mean_masked(confidence[t_idx], board_mask),
            "mean_confidence_predict": mean_masked(confidence[t_idx], predict_mask),
            "mean_target_probability_board": mean_masked(target_prob[t_idx], board_mask),
            "mean_target_probability_predict": mean_masked(target_prob[t_idx], predict_mask),
            "mean_margin_board": mean_masked(margin[t_idx], board_mask),
            "mean_margin_predict": mean_masked(margin[t_idx], predict_mask),
            "mean_abs_digit_logit": mean_masked(np.mean(np.abs(logits[t_idx]), axis=-1), board_mask),
            "max_abs_digit_logit": float(np.max(np.abs(logits[t_idx][board_mask]))) if np.any(board_mask) else float("nan"),
        }

        if hidden is not None and hidden.shape[-1] > 0:
            hidden_abs = np.mean(np.abs(hidden[t_idx]), axis=-1)
            row["mean_abs_hidden"] = mean_masked(hidden_abs, board_mask)
            row["max_abs_hidden"] = float(np.max(np.abs(hidden[t_idx][board_mask]))) if np.any(board_mask) else float("nan")
        else:
            row["mean_abs_hidden"] = float("nan")
            row["max_abs_hidden"] = float("nan")

        rows.append(row)
        prev_pred = pred_t
        prev_correct = correct_t

    return pd.DataFrame(rows)


def analyze_puzzle(puzzle_dir: Path, manifest: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    puzzle_idx = puzzle_index_from_dir(puzzle_dir)
    checkpoint_dirs = find_checkpoint_dirs(puzzle_dir)

    if not checkpoint_dirs:
        print(f"[WARN] No checkpoint trace folders found for {puzzle_dir.name}")
        return pd.DataFrame()

    board_mask = np.load(puzzle_dir / "board_mask_9x9.npy").astype(bool)
    predict_mask = np.load(puzzle_dir / "predict_mask_9x9.npy").astype(bool)
    grid_size = grid_size_from_board_mask(board_mask)

    frames = []
    for ckpt_dir in checkpoint_dirs:
        trace_path = ckpt_dir / "trace.npy"
        checkpoint_step = step_number_from_name(ckpt_dir.name)
        checkpoint_label = ckpt_dir.name

        print(f"[analyze] {puzzle_dir.name} / {checkpoint_label}")
        trace = np.load(trace_path, mmap_mode="r")
        df = analyze_trace(
            trace=trace,
            manifest=manifest,
            puzzle_idx=puzzle_idx,
            checkpoint_step=checkpoint_step,
            checkpoint_label=checkpoint_label,
            board_mask=board_mask,
            predict_mask=predict_mask,
            grid_size=grid_size,
        )
        frames.append(df)

    puzzle_df = pd.concat(frames, ignore_index=True)

    p_out = out_dir / puzzle_dir.name
    p_out.mkdir(parents=True, exist_ok=True)
    puzzle_df.to_csv(p_out / "trace_analysis.csv", index=False)

    return puzzle_df


# =============================================================================
# Plot helpers
# =============================================================================

def save_heatmap(
    data_2d: np.ndarray,
    x_labels: list[Any],
    y_labels: list[Any],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    annotate: bool = False,
    integer_values: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.18), max(5, len(y_labels) * 0.22)))
    im = ax.imshow(data_2d, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_tick_count = min(10, len(x_labels))
    y_tick_count = min(12, len(y_labels))

    if len(x_labels) > 0:
        x_idx = np.linspace(0, len(x_labels) - 1, x_tick_count, dtype=int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(x_labels[i]) for i in x_idx], rotation=45, ha="right")

    if len(y_labels) > 0:
        y_idx = np.linspace(0, len(y_labels) - 1, y_tick_count, dtype=int)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([str(y_labels[i]) for i in y_idx])

    if annotate and data_2d.size <= 300:
        for r in range(data_2d.shape[0]):
            for c in range(data_2d.shape[1]):
                val = data_2d[r, c]
                label = f"{int(val)}" if integer_values else f"{val:.2f}"
                ax.text(c, r, label, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_binary_heatmap(
    data_2d: np.ndarray,
    x_labels: list[Any],
    y_labels: list[Any],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    false_label: str = "Not solved",
    true_label: str = "Solved",
    annotate: bool = False,
) -> None:
    """Save a true binary heatmap with a discrete colorbar/legend."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_2d = np.asarray(data_2d, dtype=float)
    cmap = ListedColormap(["#2b004f", "#ffe51f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.18), max(5, len(y_labels) * 0.22)))
    im = ax.imshow(data_2d, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels([false_label, true_label])
    cbar.set_label("Status")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_tick_count = min(10, len(x_labels))
    y_tick_count = min(12, len(y_labels))

    if len(x_labels) > 0:
        x_idx = np.linspace(0, len(x_labels) - 1, x_tick_count, dtype=int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([str(x_labels[i]) for i in x_idx], rotation=45, ha="right")

    if len(y_labels) > 0:
        y_idx = np.linspace(0, len(y_labels) - 1, y_tick_count, dtype=int)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([str(y_labels[i]) for i in y_idx])

    if annotate and data_2d.size <= 300:
        for r in range(data_2d.shape[0]):
            for c in range(data_2d.shape[1]):
                ax.text(c, r, str(int(data_2d[r, c])), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_sudoku_grid_lines(ax, n: int) -> None:
    """Add light grid lines around cells in a small Sudoku snapshot."""
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)


def pivot_metric(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[int], list[int]]:
    pivot = df.pivot_table(
        index="checkpoint_step",
        columns="nca_step",
        values=metric,
        aggfunc="mean",
    ).sort_index()

    data = pivot.to_numpy()
    y_labels = [int(x) for x in pivot.index.tolist()]
    x_labels = [int(x) for x in pivot.columns.tolist()]
    return data, x_labels, y_labels


def plot_puzzle_heatmaps(puzzle_df: pd.DataFrame, puzzle_out: Path, puzzle_idx: int) -> None:
    metrics = [
        ("board_wrong", "Wrong cells", "wrong_cells_heatmap.png", True),
        ("mean_entropy_predict", "Mean digit entropy on predicted cells", "entropy_heatmap.png", False),
        ("mean_confidence_predict", "Mean digit confidence on predicted cells", "confidence_heatmap.png", False),
        ("mean_target_probability_predict", "Mean target probability on predicted cells", "target_probability_heatmap.png", False),
        ("prediction_changes", "Prediction changes since previous NCA step", "prediction_changes_heatmap.png", True),
        ("mean_abs_digit_logit", "Mean absolute digit logit", "mean_abs_digit_logit_heatmap.png", False),
    ]

    for metric, title, filename, integer_values in metrics:
        data, x_labels, y_labels = pivot_metric(puzzle_df, metric)
        save_heatmap(
            data,
            x_labels=x_labels,
            y_labels=y_labels,
            out_path=puzzle_out / filename,
            title=f"Puzzle {puzzle_idx:04d}: {title}",
            xlabel="NCA step",
            ylabel="Checkpoint step",
            cbar_label=title,
            annotate=False,
            integer_values=integer_values,
        )

    df = puzzle_df.copy()
    df["corrected_minus_broken"] = df["corrected_cells"] - df["broken_cells"]
    data, x_labels, y_labels = pivot_metric(df, "corrected_minus_broken")
    save_heatmap(
        data,
        x_labels=x_labels,
        y_labels=y_labels,
        out_path=puzzle_out / "corrected_minus_broken_heatmap.png",
        title=f"Puzzle {puzzle_idx:04d}: corrected cells minus broken cells",
        xlabel="NCA step",
        ylabel="Checkpoint step",
        cbar_label="Corrected - broken",
        annotate=False,
        integer_values=True,
    )


def plot_final_wrong_by_checkpoint(puzzle_df: pd.DataFrame, puzzle_out: Path, puzzle_idx: int) -> None:
    final_df = (
        puzzle_df.sort_values(["checkpoint_step", "nca_step"])
        .groupby("checkpoint_step", as_index=False)
        .tail(1)
        .sort_values("checkpoint_step")
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(final_df["checkpoint_step"], final_df["board_wrong"], marker="o", linewidth=1.5)
    ax.set_title(f"Puzzle {puzzle_idx:04d}: final wrong cells by checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Wrong cells after final NCA step")
    ax.grid(True, linestyle="--", alpha=0.3)

    solved = final_df[final_df["board_wrong"] == 0]
    if not solved.empty:
        ax.scatter(solved["checkpoint_step"], solved["board_wrong"], s=60, zorder=3, label="Solved")
        ax.legend()

    fig.tight_layout()
    fig.savefig(puzzle_out / "final_wrong_cells_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_representative_checkpoints(puzzle_df: pd.DataFrame) -> list[int]:
    checkpoints = sorted(puzzle_df["checkpoint_step"].unique().tolist())
    if not checkpoints:
        return []

    selected = {checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]}

    final_df = (
        puzzle_df.sort_values(["checkpoint_step", "nca_step"])
        .groupby("checkpoint_step", as_index=False)
        .tail(1)
        .sort_values("checkpoint_step")
    )
    solved = final_df[final_df["board_wrong"] == 0]
    if not solved.empty:
        selected.add(int(solved.iloc[0]["checkpoint_step"]))

    return sorted(selected)


def plot_selected_checkpoint_dynamics(puzzle_df: pd.DataFrame, puzzle_out: Path, puzzle_idx: int) -> None:
    selected_checkpoints = choose_representative_checkpoints(puzzle_df)
    if not selected_checkpoints:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    for ckpt in selected_checkpoints:
        d = puzzle_df[puzzle_df["checkpoint_step"] == ckpt].sort_values("nca_step")
        ax1.plot(d["nca_step"], d["board_wrong"], marker="o", markersize=3, linewidth=1.4, label=f"wrong @ {ckpt}")

    ax1.set_title(f"Puzzle {puzzle_idx:04d}: selected checkpoint dynamics")
    ax1.set_xlabel("NCA step")
    ax1.set_ylabel("Wrong cells")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    for ckpt in selected_checkpoints:
        d = puzzle_df[puzzle_df["checkpoint_step"] == ckpt].sort_values("nca_step")
        ax2.plot(d["nca_step"], d["mean_entropy_predict"], linestyle="--", linewidth=1.0, alpha=0.7, label=f"entropy @ {ckpt}")

    ax2.set_ylabel("Mean entropy on predicted cells")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(puzzle_out / "selected_checkpoint_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_snapshots(
    puzzle_dir: Path,
    puzzle_df: pd.DataFrame,
    manifest: dict[str, Any],
    puzzle_out: Path,
    puzzle_idx: int,
) -> None:
    final_df = (
        puzzle_df.sort_values(["checkpoint_step", "nca_step"])
        .groupby("checkpoint_step", as_index=False)
        .tail(1)
        .sort_values("checkpoint_step")
    )

    checkpoint_steps = [int(final_df.iloc[-1]["checkpoint_step"])]
    solved = final_df[final_df["board_wrong"] == 0]
    if not solved.empty:
        first_solved = int(solved.iloc[0]["checkpoint_step"])
        if first_solved not in checkpoint_steps:
            checkpoint_steps.insert(0, first_solved)

    number_slice = slice_from_manifest(manifest, "number_channels")
    target_slice = slice_from_manifest(manifest, "target_channels")
    board_mask = np.load(puzzle_dir / "board_mask_9x9.npy").astype(bool)
    givens = np.load(puzzle_dir / "givens_9x9.npy")

    snapshot_dir = puzzle_out / "prediction_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_step in checkpoint_steps:
        ckpt_dir = puzzle_dir / f"step_{ckpt_step:06d}"
        trace_path = ckpt_dir / "trace.npy"
        if not trace_path.exists():
            continue

        trace = np.load(trace_path, mmap_mode="r")
        saved_steps = nca_steps_for_trace(trace, manifest)
        target = target_digits_from_trace(np.asarray(trace[0]), target_slice)

        selected_indices = []
        for requested in SNAPSHOT_NCA_STEPS:
            nearest = int(np.argmin(np.abs(np.asarray(saved_steps) - requested)))
            if nearest not in selected_indices:
                selected_indices.append(nearest)

        n = len(selected_indices)
        cols = min(5, n)
        rows = int(math.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")

        for ax, t_idx in zip(axes.flat, selected_indices):
            state = np.asarray(trace[t_idx], dtype=np.float32)
            pred = np.argmax(state[..., number_slice], axis=-1) + 1
            correct = (pred == target) & board_mask

            # 0 outside, 1 wrong, 2 correct, 3 given.
            # Outside cells are not visible after active_crop for 4x4/6x6 puzzles,
            # but keeping the category makes the mapping explicit.
            image = np.zeros(pred.shape, dtype=np.float32)
            image[board_mask] = 1.0
            image[correct] = 2.0
            image[(givens > 0) & board_mask] = 3.0

            crop_img = active_crop(image, board_mask)
            crop_pred = active_crop(pred, board_mask)
            crop_givens = active_crop(givens, board_mask)

            snapshot_cmap = ListedColormap(["#f2f2f2", "#2f6f95", "#35b779", "#f2c94c"])
            snapshot_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], snapshot_cmap.N)

            ax.imshow(crop_img, interpolation="nearest", cmap=snapshot_cmap, norm=snapshot_norm)
            ax.set_title(f"NCA step {saved_steps[t_idx]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            add_sudoku_grid_lines(ax, crop_img.shape[0])

            for r in range(crop_pred.shape[0]):
                for c in range(crop_pred.shape[1]):
                    value = int(crop_pred[r, c])
                    suffix = "*" if int(crop_givens[r, c]) > 0 else ""
                    ax.text(c, r, f"{value}{suffix}", ha="center", va="center", fontsize=10, color="black")

        legend_handles = [
            Patch(facecolor="#f2c94c", edgecolor="black", label="Given cell"),
            Patch(facecolor="#35b779", edgecolor="black", label="Correct prediction"),
            Patch(facecolor="#2f6f95", edgecolor="black", label="Incorrect prediction"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.suptitle(
            f"Puzzle {puzzle_idx:04d}: predictions at checkpoint {ckpt_step}\n"
            "Numbers marked with * are givens",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        fig.savefig(snapshot_dir / f"puzzle_{puzzle_idx:04d}_step_{ckpt_step:06d}_predictions.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Combined plots
# =============================================================================

def final_step_df(all_df: pd.DataFrame) -> pd.DataFrame:
    return (
        all_df.sort_values(["puzzle_idx", "checkpoint_step", "nca_step"])
        .groupby(["puzzle_idx", "checkpoint_step"], as_index=False)
        .tail(1)
        .sort_values(["puzzle_idx", "checkpoint_step"])
    )


def plot_combined_final_wrong(all_df: pd.DataFrame, out_dir: Path) -> None:
    final_df = final_step_df(all_df)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for puzzle_idx, group in final_df.groupby("puzzle_idx"):
        ax.plot(group["checkpoint_step"], group["board_wrong"], marker="o", linewidth=1.4, label=f"Puzzle {int(puzzle_idx):04d}")

    ax.set_title("Final wrong cells by checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Wrong cells after final NCA step")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "final_wrong_cells_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_wrong_cells_heatmap(all_df: pd.DataFrame, out_dir: Path) -> None:
    final_df = final_step_df(all_df)
    pivot = final_df.pivot_table(index="puzzle_idx", columns="checkpoint_step", values="board_wrong", aggfunc="mean")
    pivot = pivot.sort_index()
    data = pivot.to_numpy(dtype=float)

    save_heatmap(
        data,
        x_labels=[int(x) for x in pivot.columns.tolist()],
        y_labels=[f"{int(x):04d}" for x in pivot.index.tolist()],
        out_path=out_dir / "wrong_cells_heatmap.png",
        title="Final wrong cells across checkpoints",
        xlabel="Checkpoint step",
        ylabel="Puzzle",
        cbar_label="Wrong cells",
        annotate=data.size <= 300,
        integer_values=True,
    )


def plot_combined_solved_heatmap(all_df: pd.DataFrame, out_dir: Path) -> None:
    final_df = final_step_df(all_df)
    pivot = final_df.pivot_table(index="puzzle_idx", columns="checkpoint_step", values="solved", aggfunc="max")
    pivot = pivot.sort_index()
    data = pivot.to_numpy(dtype=float)

    save_binary_heatmap(
        data,
        x_labels=[int(x) for x in pivot.columns.tolist()],
        y_labels=[f"{int(x):04d}" for x in pivot.index.tolist()],
        out_path=out_dir / "solved_heatmap.png",
        title="Solved puzzles across checkpoints",
        xlabel="Checkpoint step",
        ylabel="Puzzle",
        false_label="Not solved",
        true_label="Solved",
        annotate=data.size <= 300,
    )


def plot_first_solved_checkpoint(all_df: pd.DataFrame, out_dir: Path) -> None:
    final_df = final_step_df(all_df)

    rows = []
    for puzzle_idx, group in final_df.groupby("puzzle_idx"):
        solved = group[group["board_wrong"] == 0]
        if solved.empty:
            first = np.nan
        else:
            first = int(solved.iloc[0]["checkpoint_step"])
        rows.append({"puzzle_idx": int(puzzle_idx), "first_solved_checkpoint": first})

    out = pd.DataFrame(rows).sort_values("puzzle_idx")
    out.to_csv(out_dir / "first_solved_checkpoint.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [f"{int(x):04d}" for x in out["puzzle_idx"]]
    values = out["first_solved_checkpoint"].to_numpy(dtype=float)
    plot_values = np.nan_to_num(values, nan=0.0)
    bars = ax.bar(labels, plot_values)

    ax.set_title("First checkpoint where each puzzle is solved")
    ax.set_xlabel("Puzzle")
    ax.set_ylabel("Checkpoint step")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        if np.isnan(value):
            label = "not solved"
            y = 0
        else:
            label = f"{int(value)}"
            y = value
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va="bottom", fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(out_dir / "first_solved_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_metric_final(all_df: pd.DataFrame, out_dir: Path, metric: str, filename: str, ylabel: str, title: str) -> None:
    final_df = final_step_df(all_df)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for puzzle_idx, group in final_df.groupby("puzzle_idx"):
        ax.plot(group["checkpoint_step"], group[metric], marker="o", linewidth=1.4, label=f"Puzzle {int(puzzle_idx):04d}")

    ax.set_title(title)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_prediction_changes(all_df: pd.DataFrame, out_dir: Path) -> None:
    grouped = (
        all_df.groupby(["puzzle_idx", "checkpoint_step"], as_index=False)["prediction_changes"]
        .mean()
        .sort_values(["puzzle_idx", "checkpoint_step"])
    )

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for puzzle_idx, group in grouped.groupby("puzzle_idx"):
        ax.plot(group["checkpoint_step"], group["prediction_changes"], marker="o", linewidth=1.4, label=f"Puzzle {int(puzzle_idx):04d}")

    ax.set_title("Mean prediction changes per NCA step")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Mean changed cells per NCA step")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "mean_prediction_changes_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_mean_metric_by_rollout_step(
    all_df: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str,
    mean_col_name: str,
    std_col_name: str,
    csv_filename: str,
    plot_filename_template: str,
    title_template: str,
    ylabel: str,
    max_nca_step: int = 32,
    grid_sizes: tuple[int, ...] = (4, 9),
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Plot a mean per-NCA-step metric for each grid size.

    The mean is computed over all selected puzzles and all checkpoints that
    match the grid size. The plot is restricted to NCA steps up to
    max_nca_step, which is useful for studying dynamics inside the original
    training horizon.
    """
    if "grid_size" not in all_df.columns:
        print(f"[WARN] Cannot plot {metric}: missing grid_size column.")
        return pd.DataFrame()

    if metric not in all_df.columns:
        print(f"[WARN] Cannot plot {metric}: missing metric column.")
        return pd.DataFrame()

    d = all_df[all_df["nca_step"] <= max_nca_step].copy()
    if d.empty:
        print(f"[WARN] Cannot plot {metric}: no rows within requested NCA-step range.")
        return pd.DataFrame()

    summary = (
        d.groupby(["grid_size", "nca_step"], as_index=False)
        .agg(
            **{
                mean_col_name: (metric, "mean"),
                std_col_name: (metric, "std"),
                "count": (metric, "size"),
            }
        )
        .sort_values(["grid_size", "nca_step"])
    )
    summary.to_csv(out_dir / csv_filename, index=False)

    for grid_size in grid_sizes:
        g = summary[summary["grid_size"] == grid_size].sort_values("nca_step")
        if g.empty:
            print(f"[WARN] No rows found for {grid_size}x{grid_size}; skipping {metric} rollout plot.")
            continue

        x = g["nca_step"].to_numpy(dtype=float)
        y = g[mean_col_name].to_numpy(dtype=float)
        std = g[std_col_name].fillna(0.0).to_numpy(dtype=float)
        n = np.maximum(g["count"].to_numpy(dtype=float), 1.0)
        sem = std / np.sqrt(n)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(x, y, marker="o", linewidth=1.8, label=ylabel)
        ax.fill_between(x, y - sem, y + sem, alpha=0.2, label="±1 standard error")

        ax.set_title(title_template.format(grid_size=grid_size))
        ax.set_xlabel("NCA step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max_nca_step)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            out_dir / plot_filename_template.format(grid_size=grid_size),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    return summary


def plot_mean_wrong_cells_by_rollout_step(
    all_df: pd.DataFrame,
    out_dir: Path,
    max_nca_step: int = 32,
    grid_sizes: tuple[int, ...] = (4, 9),
) -> pd.DataFrame:
    return plot_mean_metric_by_rollout_step(
        all_df,
        out_dir,
        metric="board_wrong",
        mean_col_name="mean_board_wrong",
        std_col_name="std_board_wrong",
        csv_filename="mean_wrong_cells_by_rollout_step_to_32.csv",
        plot_filename_template="mean_wrong_cells_by_rollout_step_{grid_size}x{grid_size}.png",
        title_template="{grid_size}x{grid_size}: mean wrong cells by NCA step",
        ylabel="Mean wrong cells",
        max_nca_step=max_nca_step,
        grid_sizes=grid_sizes,
    )


def plot_mean_confidence_by_rollout_step(
    all_df: pd.DataFrame,
    out_dir: Path,
    max_nca_step: int = 32,
    grid_sizes: tuple[int, ...] = (4, 9),
) -> pd.DataFrame:
    return plot_mean_metric_by_rollout_step(
        all_df,
        out_dir,
        metric="mean_confidence_predict",
        mean_col_name="mean_confidence_predict",
        std_col_name="std_confidence_predict",
        csv_filename="mean_confidence_by_rollout_step_to_32.csv",
        plot_filename_template="mean_confidence_by_rollout_step_{grid_size}x{grid_size}.png",
        title_template="{grid_size}x{grid_size}: mean confidence by NCA step",
        ylabel="Mean confidence on predicted cells",
        max_nca_step=max_nca_step,
        grid_sizes=grid_sizes,
        ylim=(0.0, 1.0),
    )


def plot_wrong_cells_and_confidence_by_rollout_step(
    all_df: pd.DataFrame,
    out_dir: Path,
    max_nca_step: int = 32,
    grid_sizes: tuple[int, ...] = (4, 9),
) -> None:
    """Plot wrong cells and confidence together using two y-axes.

    This makes it easier to visually inspect whether confidence increases while
    wrong-cell count decreases.
    """
    required = {"grid_size", "nca_step", "board_wrong", "mean_confidence_predict"}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[WARN] Cannot plot combined wrong/confidence rollout curves. Missing: {sorted(missing)}")
        return

    d = all_df[all_df["nca_step"] <= max_nca_step].copy()
    if d.empty:
        print("[WARN] Cannot plot combined wrong/confidence rollout curves: no rows within requested NCA-step range.")
        return

    summary = (
        d.groupby(["grid_size", "nca_step"], as_index=False)
        .agg(
            mean_board_wrong=("board_wrong", "mean"),
            mean_confidence_predict=("mean_confidence_predict", "mean"),
            count=("board_wrong", "size"),
        )
        .sort_values(["grid_size", "nca_step"])
    )
    summary.to_csv(out_dir / "mean_wrong_cells_and_confidence_by_rollout_step_to_32.csv", index=False)

    for grid_size in grid_sizes:
        g = summary[summary["grid_size"] == grid_size].sort_values("nca_step")
        if g.empty:
            print(f"[WARN] No rows found for {grid_size}x{grid_size}; skipping combined wrong/confidence plot.")
            continue

        x = g["nca_step"].to_numpy(dtype=float)
        wrong = g["mean_board_wrong"].to_numpy(dtype=float)
        conf = g["mean_confidence_predict"].to_numpy(dtype=float)

        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        line1 = ax1.plot(x, wrong, marker="o", linewidth=1.8, label="Mean wrong cells")
        ax1.set_xlabel("NCA step")
        ax1.set_ylabel("Mean wrong cells")
        ax1.set_xlim(0, max_nca_step)
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax2 = ax1.twinx()
        line2 = ax2.plot(x, conf, marker="s", linewidth=1.8, linestyle="--", label="Mean confidence")
        ax2.set_ylabel("Mean confidence on predicted cells")
        ax2.set_ylim(0.0, 1.0)

        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="center right")

        ax1.set_title(f"{grid_size}x{grid_size}: wrong cells and confidence by NCA step")

        fig.tight_layout()
        fig.savefig(
            out_dir / f"mean_wrong_cells_and_confidence_by_rollout_step_{grid_size}x{grid_size}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)



# =============================================================================
# Best checkpoint/NCA-step combination analysis
# =============================================================================

def _latex_escape(value: Any) -> str:
    """Small LaTeX escape helper that avoids pandas.to_latex/Jinja2 dependency."""
    if pd.isna(value):
        return ""

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old_char, new_text in replacements.items():
        text = text.replace(old_char, new_text)
    return text


def save_latex_table(df: pd.DataFrame, out_path: Path, *, caption: str, label: str) -> None:
    """Save a small LaTeX table without requiring pandas.to_latex/Jinja2.

    Pandas 2.x may require Jinja2 for DataFrame.to_latex(). To keep this script
    dependency-light, the LaTeX table is written manually.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(df.columns)
    col_spec = "l" * len(columns)

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    lines.append(rf"\label{{{_latex_escape(label)}}}")
    lines.append(rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join([rf"\textbf{{{_latex_escape(c)}}}" for c in columns]) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[c]) for c in columns) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_best_checkpoint_nca_step_combinations(all_df: pd.DataFrame, out_dir: Path) -> None:
    """Analyze which checkpoint/NCA-step combinations solve the most puzzles.

    This uses all NCA steps, not only the final NCA step.
    """
    required = {
        "puzzle_idx", "grid_size", "checkpoint_step", "checkpoint_label",
        "nca_step", "solved", "board_wrong", "predict_wrong",
    }
    missing = required - set(all_df.columns)
    if missing:
        print(f"[WARN] Cannot run best-combination analysis. Missing columns: {sorted(missing)}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    d = all_df.copy()
    d["solved_bool"] = d["solved"].astype(bool)
    d["solved_int"] = d["solved_bool"].astype(int)

    # 1. Best single checkpoint/NCA-step combination.
    combo = (
        d.groupby(["checkpoint_step", "checkpoint_label", "nca_step"], as_index=False)
        .agg(
            solved_puzzles=("solved_int", "sum"),
            num_puzzles=("puzzle_idx", "nunique"),
            mean_board_wrong=("board_wrong", "mean"),
            median_board_wrong=("board_wrong", "median"),
            mean_predict_wrong=("predict_wrong", "mean"),
            median_predict_wrong=("predict_wrong", "median"),
        )
        .sort_values(
            ["solved_puzzles", "mean_board_wrong", "median_board_wrong", "checkpoint_step", "nca_step"],
            ascending=[False, True, True, True, True],
        )
        .reset_index(drop=True)
    )
    combo["solve_rate"] = combo["solved_puzzles"] / combo["num_puzzles"]

    combo.to_csv(out_dir / "checkpoint_nca_step_solve_counts.csv", index=False)
    combo.head(25).to_csv(out_dir / "top_checkpoint_nca_step_combinations.csv", index=False)

    if not combo.empty:
        best_ckpt = int(combo.iloc[0]["checkpoint_step"])
        best_nca_step = int(combo.iloc[0]["nca_step"])

        best_rows = d[
            (d["checkpoint_step"] == best_ckpt)
            & (d["nca_step"] == best_nca_step)
        ].copy()

        best_solved_puzzles = (
            best_rows[best_rows["solved_bool"]]
            .sort_values(["grid_size", "puzzle_idx"])
            [["puzzle_idx", "grid_size", "checkpoint_step", "checkpoint_label", "nca_step", "board_wrong", "predict_wrong"]]
        )
        best_solved_puzzles.to_csv(out_dir / "best_combination_solved_puzzles.csv", index=False)

        best_unsolved_puzzles = (
            best_rows[~best_rows["solved_bool"]]
            .sort_values(["grid_size", "puzzle_idx"])
            [["puzzle_idx", "grid_size", "checkpoint_step", "checkpoint_label", "nca_step", "board_wrong", "predict_wrong"]]
        )
        best_unsolved_puzzles.to_csv(out_dir / "best_combination_unsolved_puzzles.csv", index=False)

    # 2. Best combinations per grid size.
    combo_by_grid = (
        d.groupby(["grid_size", "checkpoint_step", "checkpoint_label", "nca_step"], as_index=False)
        .agg(
            solved_puzzles=("solved_int", "sum"),
            num_puzzles=("puzzle_idx", "nunique"),
            mean_board_wrong=("board_wrong", "mean"),
            median_board_wrong=("board_wrong", "median"),
        )
        .sort_values(
            ["grid_size", "solved_puzzles", "mean_board_wrong", "checkpoint_step", "nca_step"],
            ascending=[True, False, True, True, True],
        )
        .reset_index(drop=True)
    )
    combo_by_grid["solve_rate"] = combo_by_grid["solved_puzzles"] / combo_by_grid["num_puzzles"]
    combo_by_grid.to_csv(out_dir / "checkpoint_nca_step_solve_counts_by_grid_size.csv", index=False)

    best_by_grid = (
        combo_by_grid
        .sort_values(
            ["grid_size", "solved_puzzles", "mean_board_wrong", "checkpoint_step", "nca_step"],
            ascending=[True, False, True, True, True],
        )
        .groupby("grid_size", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_grid.to_csv(out_dir / "best_checkpoint_nca_step_by_grid_size.csv", index=False)

    # 3. Puzzle-wise oracle result: solved anywhere.
    solved_anywhere = (
        d.groupby("puzzle_idx", as_index=False)
        .agg(
            solved_anywhere=("solved_bool", "max"),
            solved_count_across_all_rows=("solved_int", "sum"),
            grid_size=("grid_size", "first"),
            min_board_wrong=("board_wrong", "min"),
            min_predict_wrong=("predict_wrong", "min"),
        )
    )

    best_per_puzzle = (
        d.sort_values(["puzzle_idx", "board_wrong", "predict_wrong", "nca_step", "checkpoint_step"])
        .groupby("puzzle_idx", as_index=False)
        .head(1)
        .copy()
    )

    best_per_puzzle = best_per_puzzle.merge(
        solved_anywhere[
            ["puzzle_idx", "solved_anywhere", "solved_count_across_all_rows", "min_board_wrong", "min_predict_wrong"]
        ],
        on="puzzle_idx",
        how="left",
    )

    keep_cols = [
        "puzzle_idx", "grid_size", "solved_anywhere", "solved_count_across_all_rows",
        "min_board_wrong", "min_predict_wrong", "checkpoint_step", "checkpoint_label",
        "nca_step", "board_wrong", "predict_wrong", "mean_confidence_predict",
        "mean_entropy_predict",
    ]
    best_per_puzzle_out = best_per_puzzle[[c for c in keep_cols if c in best_per_puzzle.columns]]
    best_per_puzzle_out = best_per_puzzle_out.sort_values(["grid_size", "puzzle_idx"])
    best_per_puzzle_out.to_csv(out_dir / "best_result_per_puzzle_any_checkpoint_step.csv", index=False)

    solved_anywhere[solved_anywhere["solved_anywhere"]].sort_values(
        ["grid_size", "puzzle_idx"]
    ).to_csv(out_dir / "puzzles_solved_anywhere.csv", index=False)

    solved_anywhere[~solved_anywhere["solved_anywhere"]].sort_values(
        ["grid_size", "puzzle_idx"]
    ).to_csv(out_dir / "puzzles_never_solved.csv", index=False)

    summary_by_grid = (
        solved_anywhere
        .groupby("grid_size", as_index=False)
        .agg(
            total_puzzles=("puzzle_idx", "nunique"),
            solved_anywhere=("solved_anywhere", "sum"),
            mean_min_board_wrong=("min_board_wrong", "mean"),
            median_min_board_wrong=("min_board_wrong", "median"),
        )
        .sort_values("grid_size")
    )
    summary_by_grid["solve_rate_anywhere"] = summary_by_grid["solved_anywhere"] / summary_by_grid["total_puzzles"]
    summary_by_grid.to_csv(out_dir / "solved_anywhere_summary_by_grid_size.csv", index=False)

    total_summary = {
        "num_puzzles": int(solved_anywhere["puzzle_idx"].nunique()),
        "solved_anywhere": int(solved_anywhere["solved_anywhere"].sum()),
        "solve_rate_anywhere": float(solved_anywhere["solved_anywhere"].mean()),
        "best_single_combination": None if combo.empty else {
            "checkpoint_step": int(combo.iloc[0]["checkpoint_step"]),
            "checkpoint_label": str(combo.iloc[0]["checkpoint_label"]),
            "nca_step": int(combo.iloc[0]["nca_step"]),
            "solved_puzzles": int(combo.iloc[0]["solved_puzzles"]),
            "num_puzzles": int(combo.iloc[0]["num_puzzles"]),
            "solve_rate": float(combo.iloc[0]["solve_rate"]),
            "mean_board_wrong": float(combo.iloc[0]["mean_board_wrong"]),
            "median_board_wrong": float(combo.iloc[0]["median_board_wrong"]),
        },
        "by_grid_size": [
            {
                "grid_size": int(row["grid_size"]),
                "total_puzzles": int(row["total_puzzles"]),
                "solved_anywhere": int(row["solved_anywhere"]),
                "solve_rate_anywhere": float(row["solve_rate_anywhere"]),
                "mean_min_board_wrong": float(row["mean_min_board_wrong"]),
                "median_min_board_wrong": float(row["median_min_board_wrong"]),
            }
            for _, row in summary_by_grid.iterrows()
        ],
    }

    (out_dir / "solved_anywhere_summary.json").write_text(
        json.dumps(total_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 4. Heatmaps and plots.
    if not combo.empty:
        pivot_counts = combo.pivot_table(
            index="checkpoint_step",
            columns="nca_step",
            values="solved_puzzles",
            aggfunc="max",
        ).sort_index()

        save_heatmap(
            pivot_counts.to_numpy(dtype=float),
            x_labels=[int(x) for x in pivot_counts.columns.tolist()],
            y_labels=[int(x) for x in pivot_counts.index.tolist()],
            out_path=out_dir / "solved_count_by_checkpoint_and_nca_step_heatmap.png",
            title="Number of solved puzzles by checkpoint and NCA step",
            xlabel="NCA step",
            ylabel="Checkpoint step",
            cbar_label="Solved puzzles",
            annotate=pivot_counts.size <= 300,
            integer_values=True,
        )

        pivot_rate = combo.pivot_table(
            index="checkpoint_step",
            columns="nca_step",
            values="solve_rate",
            aggfunc="max",
        ).sort_index()

        save_heatmap(
            pivot_rate.to_numpy(dtype=float),
            x_labels=[int(x) for x in pivot_rate.columns.tolist()],
            y_labels=[int(x) for x in pivot_rate.index.tolist()],
            out_path=out_dir / "solve_rate_by_checkpoint_and_nca_step_heatmap.png",
            title="Solve rate by checkpoint and NCA step",
            xlabel="NCA step",
            ylabel="Checkpoint step",
            cbar_label="Solve rate",
            annotate=False,
            integer_values=False,
        )

        top_plot = combo.head(20).copy()
        top_plot["label"] = top_plot.apply(
            lambda r: f"ckpt {int(r['checkpoint_step'])}, step {int(r['nca_step'])}",
            axis=1,
        )

        fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top_plot))))
        ax.barh(top_plot["label"], top_plot["solved_puzzles"])
        ax.invert_yaxis()
        ax.set_title("Top checkpoint/NCA-step combinations by solved puzzles")
        ax.set_xlabel("Solved puzzles")
        ax.set_ylabel("Combination")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "top_checkpoint_nca_step_combinations.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        top_latex = combo.head(10)[
            ["checkpoint_step", "nca_step", "solved_puzzles", "num_puzzles", "solve_rate", "mean_board_wrong", "median_board_wrong"]
        ].copy()
        top_latex = top_latex.rename(
            columns={
                "checkpoint_step": "Checkpoint",
                "nca_step": "NCA step",
                "solved_puzzles": "Solved",
                "num_puzzles": "Puzzles",
                "solve_rate": "Solve rate",
                "mean_board_wrong": "Mean wrong",
                "median_board_wrong": "Median wrong",
            }
        )
        save_latex_table(
            top_latex,
            out_dir / "top_checkpoint_nca_step_combinations.tex",
            caption="Top checkpoint and NCA-step combinations ranked by the number of solved puzzles.",
            label="tab:top-checkpoint-nca-step-combinations",
        )

    summary_latex = summary_by_grid[
        ["grid_size", "total_puzzles", "solved_anywhere", "solve_rate_anywhere", "mean_min_board_wrong", "median_min_board_wrong"]
    ].copy()
    summary_latex = summary_latex.rename(
        columns={
            "grid_size": "Grid size",
            "total_puzzles": "Puzzles",
            "solved_anywhere": "Solved anywhere",
            "solve_rate_anywhere": "Solve rate anywhere",
            "mean_min_board_wrong": "Mean min. wrong",
            "median_min_board_wrong": "Median min. wrong",
        }
    )
    save_latex_table(
        summary_latex,
        out_dir / "solved_anywhere_summary_by_grid_size.tex",
        caption="Puzzles solved at least once across all checkpoints and NCA steps.",
        label="tab:solved-anywhere-summary-by-grid-size",
    )

    print()
    print("[best-combination analysis]")
    print(f"  Output dir: {out_dir}")
    if total_summary["best_single_combination"] is not None:
        b = total_summary["best_single_combination"]
        print(
            "  Best single combination: "
            f"checkpoint={b['checkpoint_step']}, nca_step={b['nca_step']}, "
            f"solved={b['solved_puzzles']}/{b['num_puzzles']}"
        )
    print(
        "  Solved anywhere: "
        f"{total_summary['solved_anywhere']}/{total_summary['num_puzzles']}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved NCA checkpoint traces.")
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=TRACE_ROOT_DIR,
        help="Path to selected_puzzle_checkpoint_traces.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <trace-root>/trace_analysis.",
    )
    args = parser.parse_args()

    trace_root = args.trace_root.resolve()
    out_dir = (trace_root / OUT_DIR_NAME) if args.out_dir is None else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(trace_root)
    puzzle_dirs = find_puzzle_dirs(trace_root)
    if not puzzle_dirs:
        raise SystemExit(f"No puzzle_* folders found in {trace_root}")

    print(f"Trace root: {trace_root}")
    print(f"Output dir: {out_dir}")
    print(f"Found {len(puzzle_dirs)} puzzle folders.")

    all_frames = []

    for puzzle_dir in puzzle_dirs:
        puzzle_idx = puzzle_index_from_dir(puzzle_dir)
        puzzle_out = out_dir / puzzle_dir.name
        puzzle_out.mkdir(parents=True, exist_ok=True)

        puzzle_df = analyze_puzzle(puzzle_dir, manifest, out_dir)
        if puzzle_df.empty:
            continue

        all_frames.append(puzzle_df)

        plot_puzzle_heatmaps(puzzle_df, puzzle_out, puzzle_idx)
        plot_final_wrong_by_checkpoint(puzzle_df, puzzle_out, puzzle_idx)
        plot_selected_checkpoint_dynamics(puzzle_df, puzzle_out, puzzle_idx)
        plot_prediction_snapshots(puzzle_dir, puzzle_df, manifest, puzzle_out, puzzle_idx)

    if not all_frames:
        raise SystemExit("No trace data was analyzed.")

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv(out_dir / "trace_analysis_all.csv", index=False)

    combined_dir = out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_final_wrong(all_df, combined_dir)
    plot_combined_wrong_cells_heatmap(all_df, combined_dir)
    plot_combined_solved_heatmap(all_df, combined_dir)
    plot_first_solved_checkpoint(all_df, combined_dir)
    plot_combined_metric_final(
        all_df,
        combined_dir,
        metric="mean_entropy_predict",
        filename="mean_entropy_by_checkpoint.png",
        ylabel="Mean entropy on predicted cells",
        title="Final-step entropy by checkpoint",
    )
    plot_combined_metric_final(
        all_df,
        combined_dir,
        metric="mean_target_probability_predict",
        filename="mean_target_probability_by_checkpoint.png",
        ylabel="Mean target probability on predicted cells",
        title="Final-step target probability by checkpoint",
    )
    plot_combined_prediction_changes(all_df, combined_dir)
    plot_mean_wrong_cells_by_rollout_step(all_df, combined_dir, max_nca_step=32, grid_sizes=(4, 9))
    plot_mean_confidence_by_rollout_step(all_df, combined_dir, max_nca_step=32, grid_sizes=(4, 9))
    plot_wrong_cells_and_confidence_by_rollout_step(all_df, combined_dir, max_nca_step=32, grid_sizes=(4, 9))

    best_combo_dir = combined_dir / "best_checkpoint_nca_step"
    analyze_best_checkpoint_nca_step_combinations(all_df, best_combo_dir)

    print()
    print("Done.")
    print(f"Main CSV:       {out_dir / 'trace_analysis_all.csv'}")
    print(f"Combined plots: {combined_dir}")
    print("Per-puzzle folders:")
    for puzzle_dir in puzzle_dirs:
        print(f"  - {out_dir / puzzle_dir.name}")


if __name__ == "__main__":
    main()