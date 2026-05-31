#!/usr/bin/env python3
"""
Analyze parameter-search training logs.

This script is intended for logs from the learning-rate/configuration search.
It parses log files such as:

    parameter_search-44245_0.log
    parameter_search-44245_1.log
    ...

and creates:
  - a ranked CSV table of runs by lowest validation loss
  - a LaTeX table with the best runs
  - one plot per run
  - combined plots across runs
  - a short text summary

The script can read either:
  1) a directory containing .log files, or
  2) a zip file containing .log files.

Typical usage:
    python analyze_lr_search_logs.py --input 20260513_logs.zip

or, if the script is placed next to a logs/ directory:
    python analyze_lr_search_logs.py --input logs

Dependencies:
    pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "lr_search_analysis"
DEFAULT_ROLLING_WINDOW = 15
DEFAULT_TOP_N = 8

# Log metric line format, for example:
# 100 1.9118148 train 1.8897132 lr 0.000102 gn 0.7349 upd 0.0203 mlp ...
METRIC_RE = re.compile(
    r"^\s*"
    r"(?P<step>\d+)\s+"
    r"(?P<val_loss>[-+0-9.eE]+)\s+"
    r"train\s+(?P<train_loss>[-+0-9.eE]+)\s+"
    r"lr\s+(?P<lr>[-+0-9.eE]+)\s+"
    r"gn\s+(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"upd\s+(?P<update_norm>[-+0-9.eE]+)\s+"
    r"mlp\s+(?P<mlp_grad_norm>[-+0-9.eE]+)\s+"
    r"rule\s+(?P<rule_grad_norm>[-+0-9.eE]+)\s+"
    r"size\s+(?P<size_grad_norm>[-+0-9.eE]+)"
)

HEADER_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]+)\s*=\s*(.*?)\s*$")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x: object) -> int | None:
    try:
        return int(float(x))
    except Exception:
        return None


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_") or "run"


def rolling_mean(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1, center=True).mean()


def maybe_number(s: str) -> object:
    """Convert simple numeric strings to int/float, otherwise keep string."""
    s = str(s).strip()
    if s == "":
        return s
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def read_log_sources(input_path: Path) -> dict[str, str]:
    """Return {relative_name: text} for all .log files under a dir or zip."""
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        logs: dict[str, str] = {}
        with zipfile.ZipFile(input_path) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith(".log"):
                    logs[name] = z.read(name).decode("utf-8", errors="replace")
                elif name.lower().endswith(".zip"):
                    # Support nested logs.zip inside the uploaded archive.
                    try:
                        from io import BytesIO
                        with zipfile.ZipFile(BytesIO(z.read(name))) as nested:
                            for n2 in sorted(nested.namelist()):
                                if n2.lower().endswith(".log"):
                                    logs[f"{name}!{n2}"] = nested.read(n2).decode("utf-8", errors="replace")
                    except Exception:
                        pass
        return logs

    if input_path.is_dir():
        logs = {}
        for p in sorted(input_path.rglob("*.log")):
            logs[str(p.relative_to(input_path))] = p.read_text(encoding="utf-8", errors="replace")
        return logs

    raise FileNotFoundError(f"Input must be a directory or .zip file: {input_path}")


def parse_final_json(text: str) -> dict[str, object]:
    """Try to parse the final JSON object printed at the end of the log."""
    # Find the last JSON-looking block. This is intentionally conservative.
    candidates = re.findall(r"\{\s*\"status\".*?\}", text, flags=re.DOTALL)
    if not candidates:
        return {}
    try:
        return json.loads(candidates[-1])
    except Exception:
        return {}


@dataclass
class ParsedLog:
    source_name: str
    config: dict[str, object]
    metrics: pd.DataFrame
    final_json: dict[str, object]


def parse_log(source_name: str, text: str) -> ParsedLog:
    config: dict[str, object] = {}
    rows: list[dict[str, object]] = []

    for line in text.splitlines():
        h = HEADER_RE.match(line)
        if h:
            key, value = h.group(1), h.group(2)
            config[key] = maybe_number(value)
            continue

        m = METRIC_RE.match(line)
        if m:
            rows.append({k: maybe_number(v) for k, v in m.groupdict().items()})

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        # Ensure numeric types.
        for col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
        metrics = metrics.sort_values("step").reset_index(drop=True)

    final_json = parse_final_json(text)
    # Final JSON run_name/status should override header only if present.
    for key, val in final_json.items():
        if key.upper() not in config:
            config[f"FINAL_{key.upper()}"] = val

    return ParsedLog(source_name=source_name, config=config, metrics=metrics, final_json=final_json)


def summarize_log(parsed: ParsedLog, rolling_window: int) -> dict[str, object]:
    cfg = parsed.config
    m = parsed.metrics

    run_name = str(cfg.get("RUN_NAME", cfg.get("FINAL_RUN_NAME", Path(parsed.source_name).stem)))
    task_id = cfg.get("SLURM_ARRAY_TASK_ID", "")
    job_id = cfg.get("SLURM_JOB_ID", "")

    row: dict[str, object] = {
        "run_name": run_name,
        "source_file": parsed.source_name,
        "slurm_job_id": job_id,
        "array_task_id": task_id,
        "status": cfg.get("FINAL_STATUS", cfg.get("status", "")),
        "lr_schedule": cfg.get("LR_SCHEDULE", ""),
        "learning_rate": cfg.get("LEARNING_RATE", np.nan),
        "weight_decay": cfg.get("WEIGHT_DECAY", np.nan),
        "init_value": cfg.get("INIT_VALUE", np.nan),
        "end_value": cfg.get("END_VALUE", np.nan),
        "warmup_steps": cfg.get("WARMUP_STEPS", np.nan),
        "clip_norm": cfg.get("CLIP_NORM", np.nan),
        "total_steps_config": cfg.get("TOTAL_STEPS", np.nan),
        "iterations_config": cfg.get("ITERATIONS", np.nan),
        "num_logged_points": len(m),
    }

    if m.empty:
        row.update(
            {
                "last_step": np.nan,
                "best_val_loss": np.nan,
                "best_val_step": np.nan,
                "best_train_loss_at_best_val": np.nan,
                "final_val_loss": np.nan,
                "final_train_loss": np.nan,
                "best_smoothed_val_loss": np.nan,
                "best_smoothed_val_step": np.nan,
                "final_smoothed_val_loss": np.nan,
                "final_lr": np.nan,
                "min_lr": np.nan,
                "max_lr": np.nan,
            }
        )
        return row

    smoothed = rolling_mean(m["val_loss"], rolling_window)
    best_idx = int(m["val_loss"].idxmin())
    best_smooth_idx = int(smoothed.idxmin())

    row.update(
        {
            "last_step": int(m["step"].iloc[-1]),
            "best_val_loss": float(m.loc[best_idx, "val_loss"]),
            "best_val_step": int(m.loc[best_idx, "step"]),
            "best_train_loss_at_best_val": float(m.loc[best_idx, "train_loss"]),
            "final_val_loss": float(m["val_loss"].iloc[-1]),
            "final_train_loss": float(m["train_loss"].iloc[-1]),
            "best_smoothed_val_loss": float(smoothed.iloc[best_smooth_idx]),
            "best_smoothed_val_step": int(m.loc[best_smooth_idx, "step"]),
            "final_smoothed_val_loss": float(smoothed.iloc[-1]),
            "final_lr": float(m["lr"].iloc[-1]) if "lr" in m else np.nan,
            "min_lr": float(m["lr"].min()) if "lr" in m else np.nan,
            "max_lr": float(m["lr"].max()) if "lr" in m else np.nan,
            "mean_grad_norm": float(m["grad_norm"].mean()) if "grad_norm" in m else np.nan,
            "max_grad_norm": float(m["grad_norm"].max()) if "grad_norm" in m else np.nan,
            "mean_update_norm": float(m["update_norm"].mean()) if "update_norm" in m else np.nan,
            "max_update_norm": float(m["update_norm"].max()) if "update_norm" in m else np.nan,
        }
    )
    return row


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_single_run(parsed: ParsedLog, summary_row: pd.Series, out_path: Path, rolling_window: int) -> None:
    m = parsed.metrics
    if m.empty:
        return

    run_name = summary_row["run_name"]
    rank = int(summary_row["rank_by_best_val_loss"]) if "rank_by_best_val_loss" in summary_row else None

    val_smooth = rolling_mean(m["val_loss"], rolling_window)
    train_smooth = rolling_mean(m["train_loss"], rolling_window)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(m["step"], m["val_loss"], alpha=0.25, linewidth=0.8, label="Validation loss (raw)")
    ax1.plot(m["step"], val_smooth, linewidth=2.0, label=f"Validation loss ({rolling_window}-point moving avg.)")
    ax1.plot(m["step"], m["train_loss"], alpha=0.18, linewidth=0.8, label="Train loss (raw)")
    ax1.plot(m["step"], train_smooth, linewidth=1.6, linestyle="--", label=f"Train loss ({rolling_window}-point moving avg.)")

    best_step = summary_row["best_val_step"]
    best_loss = summary_row["best_val_loss"]
    ax1.scatter([best_step], [best_loss], s=45, zorder=4, label=f"Best val: {best_loss:.4f} at {int(best_step)}")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss")
    title_prefix = f"Rank {rank}: " if rank is not None else ""
    ax1.set_title(f"{title_prefix}{run_name}")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # LR curve on secondary axis, visually muted.
    if "lr" in m.columns:
        ax2 = ax1.twinx()
        ax2.plot(m["step"], m["lr"], linewidth=1.0, alpha=0.45, linestyle=":", label="Learning rate")
        ax2.set_ylabel("Learning rate")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    else:
        ax1.legend(fontsize=8, loc="upper right")

    # Config annotation.
    config_text = (
        f"lr={summary_row.get('learning_rate', np.nan)} | "
        f"init={summary_row.get('init_value', np.nan)} | "
        f"end={summary_row.get('end_value', np.nan)} | "
        f"warmup={summary_row.get('warmup_steps', np.nan)} | "
        f"clip={summary_row.get('clip_norm', np.nan)}"
    )
    ax1.text(
        0.01,
        0.02,
        config_text,
        transform=ax1.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_loss(
    parsed_logs: list[ParsedLog],
    ranking: pd.DataFrame,
    out_path: Path,
    rolling_window: int,
    top_n: int | None = None,
    title: str = "Validation loss curves",
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))

    if top_n is not None:
        allowed = set(ranking.head(top_n)["run_name"].astype(str))
    else:
        allowed = set(ranking["run_name"].astype(str))

    rank_lookup = {
        str(row["run_name"]): int(row["rank_by_best_val_loss"])
        for _, row in ranking.iterrows()
    }

    # Plot in reverse rank so best curves are drawn on top.
    logs_by_rank = []
    for parsed in parsed_logs:
        run_name = str(parsed.config.get("RUN_NAME", parsed.config.get("FINAL_RUN_NAME", Path(parsed.source_name).stem)))
        if run_name in allowed and not parsed.metrics.empty:
            logs_by_rank.append((rank_lookup.get(run_name, 9999), run_name, parsed))
    logs_by_rank.sort(reverse=True)

    for rank, run_name, parsed in logs_by_rank:
        m = parsed.metrics
        smooth = rolling_mean(m["val_loss"], rolling_window)
        label = f"#{rank} {run_name}"
        alpha = 0.9 if rank <= 3 else 0.55
        lw = 2.4 if rank <= 3 else 1.3
        ax.plot(m["step"], smooth, linewidth=lw, alpha=alpha, label=label)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Validation loss ({rolling_window}-point moving avg.)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2 if len(logs_by_rank) > 8 else 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_loss_bar(ranking: pd.DataFrame, out_path: Path, top_n: int | None = None) -> None:
    data = ranking.copy()
    if top_n is not None:
        data = data.head(top_n)
    data = data.sort_values("best_val_loss", ascending=True)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(data) + 4), 5.5))
    labels = [f"#{int(r)} {name}" for r, name in zip(data["rank_by_best_val_loss"], data["run_name"])]
    x = np.arange(len(data))
    bars = ax.bar(x, data["best_val_loss"].to_numpy())

    ax.set_title("Runs ranked by lowest validation loss")
    ax.set_ylabel("Best validation loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, val in zip(bars, data["best_val_loss"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lr_vs_loss(ranking: pd.DataFrame, out_path: Path) -> None:
    data = ranking.dropna(subset=["learning_rate", "best_val_loss"]).copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    schedules = data["lr_schedule"].fillna("").astype(str).unique()
    for schedule in schedules:
        d = data[data["lr_schedule"].fillna("").astype(str) == schedule]
        ax.scatter(d["learning_rate"], d["best_val_loss"], s=70, alpha=0.85, label=schedule or "unknown")

        for _, row in d.iterrows():
            ax.annotate(
                str(row["run_name"]),
                (row["learning_rate"], row["best_val_loss"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.8,
            )

    if (data["learning_rate"] > 0).all() and data["learning_rate"].nunique() > 2:
        ax.set_xscale("log")

    ax.set_title("Learning rate vs. best validation loss")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best validation loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_vs_best(ranking: pd.DataFrame, out_path: Path) -> None:
    data = ranking.dropna(subset=["best_val_loss", "final_val_loss"]).copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(data["best_val_loss"], data["final_val_loss"], s=75, alpha=0.85)

    for _, row in data.iterrows():
        ax.annotate(
            str(row["run_name"]),
            (row["best_val_loss"], row["final_val_loss"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.8,
        )

    lim_min = min(data["best_val_loss"].min(), data["final_val_loss"].min())
    lim_max = max(data["best_val_loss"].max(), data["final_val_loss"].max())
    pad = (lim_max - lim_min) * 0.08 if lim_max > lim_min else 0.01
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], linestyle="--", alpha=0.5)

    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_title("Best validation loss vs. final validation loss")
    ax.set_xlabel("Best validation loss")
    ax.set_ylabel("Final validation loss")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Output tables/text
# ---------------------------------------------------------------------

def write_latex_ranking(ranking: pd.DataFrame, out_path: Path, top_n: int) -> None:
    data = ranking.head(top_n)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top learning-rate search configurations ranked by lowest validation loss.}")
    lines.append(r"\label{tab:lr-search-ranking}")
    lines.append(r"\begin{tabular}{@{}r l r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Rank} & \textbf{Run} & \textbf{LR} & \textbf{Warmup} & "
        r"\textbf{Best loss} & \textbf{Best step} & \textbf{Final loss} \\"
    )
    lines.append(r"\midrule")

    def fmt_lr(v: object) -> str:
        try:
            return f"{float(v):.0e}"
        except Exception:
            return str(v)

    def esc(s: str) -> str:
        return str(s).replace("_", r"\_")

    for _, row in data.iterrows():
        lines.append(
            f"{int(row['rank_by_best_val_loss'])} & "
            rf"\texttt{{{esc(row['run_name'])}}} & "
            f"{fmt_lr(row['learning_rate'])} & "
            f"{int(row['warmup_steps']) if not pd.isna(row['warmup_steps']) else ''} & "
            f"{row['best_val_loss']:.4f} & "
            f"{int(row['best_val_step']) if not pd.isna(row['best_val_step']) else ''} & "
            f"{row['final_val_loss']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_text(ranking: pd.DataFrame, out_path: Path, top_n: int) -> None:
    lines = []
    lines.append("Learning-rate/configuration search summary")
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"Number of parsed runs: {len(ranking)}")
    lines.append("")
    lines.append(f"Top {min(top_n, len(ranking))} by best validation loss:")
    lines.append("")

    cols = [
        "rank_by_best_val_loss",
        "run_name",
        "best_val_loss",
        "best_val_step",
        "final_val_loss",
        "learning_rate",
        "init_value",
        "end_value",
        "warmup_steps",
        "clip_norm",
    ]

    for _, row in ranking.head(top_n).iterrows():
        lines.append(
            f"#{int(row['rank_by_best_val_loss'])}: {row['run_name']} | "
            f"best={row['best_val_loss']:.6f} at step {int(row['best_val_step'])} | "
            f"final={row['final_val_loss']:.6f} | "
            f"lr={row['learning_rate']} | "
            f"init={row['init_value']} | "
            f"end={row['end_value']} | "
            f"warmup={row['warmup_steps']} | "
            f"clip={row['clip_norm']}"
        )

    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- best_val_loss is the lowest raw validation loss observed in the log.")
    lines.append("- best_smoothed_val_loss is computed from the moving average curve.")
    lines.append("- final_val_loss is the validation loss at the last logged step.")
    lines.append("- The best raw loss can be affected by noise, so inspect the curves as well as the ranking.")
    lines.append("- If a run has lower loss but worse final solve rate, prioritize the metric that matches the final evaluation objective.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze learning-rate search logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Directory or .zip file containing .log files. Default: current directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help=f"Moving-average window in logged points. Default: {DEFAULT_ROLLING_WINDOW}",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top runs to emphasize. Default: {DEFAULT_TOP_N}",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = read_log_sources(input_path)
    if not logs:
        raise SystemExit(f"No .log files found in: {input_path}")

    print(f"Found {len(logs)} log file(s).")

    parsed_logs: list[ParsedLog] = []
    summaries: list[dict[str, object]] = []

    for source_name, text in logs.items():
        parsed = parse_log(source_name, text)
        if parsed.metrics.empty:
            print(f"[WARN] No metric lines parsed from {source_name}")
        parsed_logs.append(parsed)
        summaries.append(summarize_log(parsed, args.rolling_window))

    ranking = pd.DataFrame(summaries)
    ranking = ranking.sort_values(
        ["best_val_loss", "best_smoothed_val_loss", "final_val_loss"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank_by_best_val_loss", np.arange(1, len(ranking) + 1))

    # Save full ranking.
    ranking_csv = out_dir / "lr_search_ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    # Save all metric histories in one CSV.
    history_rows = []
    rank_lookup = {row["run_name"]: row["rank_by_best_val_loss"] for _, row in ranking.iterrows()}
    for parsed in parsed_logs:
        run_name = str(parsed.config.get("RUN_NAME", parsed.config.get("FINAL_RUN_NAME", Path(parsed.source_name).stem)))
        if parsed.metrics.empty:
            continue
        m = parsed.metrics.copy()
        m.insert(0, "run_name", run_name)
        m.insert(1, "rank_by_best_val_loss", rank_lookup.get(run_name, np.nan))
        m.insert(2, "source_file", parsed.source_name)
        m["val_loss_smooth"] = rolling_mean(m["val_loss"], args.rolling_window)
        m["train_loss_smooth"] = rolling_mean(m["train_loss"], args.rolling_window)
        history_rows.append(m)

    if history_rows:
        history = pd.concat(history_rows, ignore_index=True)
        history.to_csv(out_dir / "lr_search_history.csv", index=False)

    # Output summary files.
    write_latex_ranking(ranking, out_dir / "lr_search_ranking_top.tex", args.top_n)
    write_summary_text(ranking, out_dir / "analysis_summary.txt", args.top_n)

    # Plots.
    combined_dir = out_dir / "combined_plots"
    combined_dir.mkdir(exist_ok=True)

    plot_combined_loss(
        parsed_logs,
        ranking,
        combined_dir / "combined_validation_loss_all_runs.png",
        args.rolling_window,
        top_n=None,
        title="Validation loss curves for all runs",
    )

    plot_combined_loss(
        parsed_logs,
        ranking,
        combined_dir / f"combined_validation_loss_top_{args.top_n}.png",
        args.rolling_window,
        top_n=args.top_n,
        title=f"Validation loss curves for top {args.top_n} runs",
    )

    plot_best_loss_bar(
        ranking,
        combined_dir / "best_validation_loss_ranking.png",
        top_n=None,
    )

    plot_best_loss_bar(
        ranking,
        combined_dir / f"best_validation_loss_ranking_top_{args.top_n}.png",
        top_n=args.top_n,
    )

    plot_lr_vs_loss(ranking, combined_dir / "learning_rate_vs_best_loss.png")
    plot_final_vs_best(ranking, combined_dir / "best_vs_final_validation_loss.png")

    # One plot per run.
    per_run_dir = out_dir / "per_run_plots"
    per_run_dir.mkdir(exist_ok=True)

    summary_by_name = {str(row["run_name"]): row for _, row in ranking.iterrows()}
    for parsed in parsed_logs:
        run_name = str(parsed.config.get("RUN_NAME", parsed.config.get("FINAL_RUN_NAME", Path(parsed.source_name).stem)))
        row = summary_by_name.get(run_name)
        if row is None:
            continue
        rank = int(row["rank_by_best_val_loss"])
        fname = f"{rank:02d}_{sanitize_filename(run_name)}.png"
        plot_single_run(parsed, row, per_run_dir / fname, args.rolling_window)

    print()
    print(f"Ranking CSV:       {ranking_csv}")
    print(f"History CSV:       {out_dir / 'lr_search_history.csv'}")
    print(f"LaTeX table:       {out_dir / 'lr_search_ranking_top.tex'}")
    print(f"Summary text:      {out_dir / 'analysis_summary.txt'}")
    print(f"Combined plots:    {combined_dir}")
    print(f"Per-run plots:     {per_run_dir}")
    print()
    print("Top runs:")
    for _, row in ranking.head(args.top_n).iterrows():
        print(
            f"  #{int(row['rank_by_best_val_loss'])} {row['run_name']}: "
            f"best={row['best_val_loss']:.6f} at step {int(row['best_val_step'])}, "
            f"final={row['final_val_loss']:.6f}"
        )


if __name__ == "__main__":
    main()