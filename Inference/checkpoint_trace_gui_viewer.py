#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set this to the full path of the folder created by
# record_selected_puzzle_checkpoint_traces.py.
#
# Example:
# TRACE_ROOT_DIR = Path("/home/daniel/.../selected_puzzle_checkpoint_traces")
#
TRACE_ROOT_DIR = Path(__file__).resolve().parent / "selected_puzzle_checkpoint_traces_new"


# ==============================================================================


def _step_number_from_name(name: str) -> int:
    match = re.search(r"step[_-]?(\d+)", name)
    return int(match.group(1)) if match else -1


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _slice_from_list(v: Any) -> slice:
    if isinstance(v, list) and len(v) == 2:
        return slice(int(v[0]), int(v[1]))
    return slice(0, 0)


class CheckpointTraceViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NCA Checkpoint Trace Viewer")
        self.root.geometry("1650x950")

        self.trace_root: Path | None = None
        self.trace_manifest: dict[str, Any] = {}

        self.puzzle_dirs: list[Path] = []
        self.checkpoint_dirs: list[Path] = []

        self.current_puzzle_idx = tk.IntVar(value=0)
        self.current_checkpoint_idx = tk.IntVar(value=0)
        self.current_nca_step_idx = tk.IntVar(value=0)
        self.current_view_label = tk.StringVar(value="")

        self.current_trace: np.ndarray | None = None
        self.current_metrics: dict[str, Any] = {}
        self.current_puzzle_metadata: dict[str, Any] = {}

        self.current_views: list[dict[str, Any]] = []
        self.channel_names: list[str] = []
        self.saved_step_indices: list[int] = []

        self._build_ui()
        self._try_load_default_root()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open trace root", command=self.choose_trace_root).pack(side=tk.LEFT)
        self.root_label = ttk.Label(top, text="No trace root selected")
        self.root_label.pack(side=tk.LEFT, padx=10)

        # Puzzle navigation
        puzzle_nav = ttk.LabelFrame(self.root, text="Puzzle", padding=8)
        puzzle_nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        ttk.Button(puzzle_nav, text="<< Previous puzzle", command=self.prev_puzzle).pack(side=tk.LEFT)
        ttk.Button(puzzle_nav, text="Next puzzle >>", command=self.next_puzzle).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(puzzle_nav, text="Select:").pack(side=tk.LEFT)
        self.puzzle_combo = ttk.Combobox(puzzle_nav, state="readonly", width=45, values=())
        self.puzzle_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.puzzle_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_puzzle_combo())

        self.puzzle_label = ttk.Label(puzzle_nav, text="Puzzle: -")
        self.puzzle_label.pack(side=tk.LEFT)

        # Checkpoint navigation
        checkpoint_nav = ttk.LabelFrame(self.root, text="Checkpoint", padding=8)
        checkpoint_nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        ttk.Button(checkpoint_nav, text="<< Previous checkpoint", command=self.prev_checkpoint).pack(side=tk.LEFT)
        ttk.Button(checkpoint_nav, text="Next checkpoint >>", command=self.next_checkpoint).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(checkpoint_nav, text="Select:").pack(side=tk.LEFT)
        self.checkpoint_combo = ttk.Combobox(checkpoint_nav, state="readonly", width=30, values=())
        self.checkpoint_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.checkpoint_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_checkpoint_combo())

        self.checkpoint_label = ttk.Label(checkpoint_nav, text="Checkpoint: -")
        self.checkpoint_label.pack(side=tk.LEFT)

        # NCA step navigation
        step_nav = ttk.LabelFrame(self.root, text="NCA step", padding=8)
        step_nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        ttk.Button(step_nav, text="<< Previous NCA step", command=self.prev_nca_step).pack(side=tk.LEFT)
        ttk.Button(step_nav, text="Next NCA step >>", command=self.next_nca_step).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(step_nav, text="Select:").pack(side=tk.LEFT)
        self.step_combo = ttk.Combobox(step_nav, state="readonly", width=20, values=())
        self.step_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.step_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_step_combo())

        self.step_label = ttk.Label(step_nav, text="NCA step: -")
        self.step_label.pack(side=tk.LEFT, padx=(0, 12))

        self.step_scale = ttk.Scale(step_nav, orient=tk.HORIZONTAL, command=self.on_step_scale)
        self.step_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Channel/view navigation
        channel_nav = ttk.LabelFrame(self.root, text="Channel / view", padding=8)
        channel_nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        ttk.Button(channel_nav, text="<< Previous channel", command=self.prev_view).pack(side=tk.LEFT)
        ttk.Button(channel_nav, text="Next channel >>", command=self.next_view).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(channel_nav, text="Select:").pack(side=tk.LEFT)
        self.view_combo = ttk.Combobox(
            channel_nav,
            textvariable=self.current_view_label,
            state="readonly",
            width=62,
            values=(),
        )
        self.view_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.view_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_view())

        self.channel_label = ttk.Label(channel_nav, text="Channel: -")
        self.channel_label.pack(side=tk.LEFT)

        # Info row
        info = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        info.pack(side=tk.TOP, fill=tk.X)

        self.info_label = ttk.Label(info, text="")
        self.info_label.pack(side=tk.LEFT)

        # Main content
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=2)
        main.add(right, weight=1)

        self.figure = Figure(figsize=(8.5, 8.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        text_top = ttk.LabelFrame(right, text="Matrix values", padding=6)
        text_top.pack(fill=tk.BOTH, expand=True)

        self.matrix_text = tk.Text(text_top, wrap=tk.NONE, font=("Courier New", 10))
        self.matrix_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(text_top, orient=tk.VERTICAL, command=self.matrix_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.matrix_text.configure(yscrollcommand=yscroll.set)

        meta_frame = ttk.LabelFrame(right, text="Metadata / metrics", padding=8)
        meta_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.meta_text = tk.Text(meta_frame, wrap=tk.WORD, height=20, font=("Courier New", 9))
        self.meta_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load_default_root(self) -> None:
        candidates = [
            Path(TRACE_ROOT_DIR),
            Path.cwd() / "selected_puzzle_checkpoint_traces",
            Path(__file__).resolve().parent / "selected_puzzle_checkpoint_traces",
        ]

        for candidate in candidates:
            if (candidate / "trace_manifest.json").exists():
                try:
                    self.load_trace_root(candidate)
                    return
                except Exception:
                    pass

    def choose_trace_root(self) -> None:
        selected = filedialog.askdirectory(title="Choose selected_puzzle_checkpoint_traces folder")
        if not selected:
            return
        self.load_trace_root(Path(selected))

    def load_trace_root(self, root: Path) -> None:
        root = root.resolve()
        manifest_path = root / "trace_manifest.json"

        if not manifest_path.exists():
            messagebox.showerror("Invalid trace root", f"Could not find trace_manifest.json in:\n{root}")
            return

        self.trace_root = root
        self.trace_manifest = _safe_json_load(manifest_path)

        self.puzzle_dirs = sorted(
            [p for p in root.glob("puzzle_*") if p.is_dir()],
            key=lambda p: p.name,
        )

        if not self.puzzle_dirs:
            messagebox.showerror("No puzzles", f"Could not find any puzzle_* folders in:\n{root}")
            return

        self.channel_names = self._build_channel_names()
        self.saved_step_indices = [int(x) for x in self.trace_manifest.get("saved_step_indices", [])]

        self.current_puzzle_idx.set(0)
        self.current_checkpoint_idx.set(0)
        self.current_nca_step_idx.set(0)

        self.root_label.config(text=f"{root}")
        self._refresh_puzzle_combo()
        self.load_current_puzzle()

    def _refresh_puzzle_combo(self) -> None:
        labels = [self._puzzle_label(i, p) for i, p in enumerate(self.puzzle_dirs)]
        self.puzzle_combo.configure(values=labels)
        if labels:
            self.puzzle_combo.current(self.current_puzzle_idx.get())

    def _refresh_checkpoint_combo(self) -> None:
        labels = [p.name for p in self.checkpoint_dirs]
        self.checkpoint_combo.configure(values=labels)
        if labels:
            self.checkpoint_combo.current(self.current_checkpoint_idx.get())

    def _refresh_step_combo(self) -> None:
        if self.current_trace is None:
            self.step_combo.configure(values=())
            return

        n = self.current_trace.shape[0]
        labels = []
        for i in range(n):
            actual_step = self._actual_nca_step_from_saved_index(i)
            labels.append(f"{i:03d} (NCA step {actual_step})")

        self.step_combo.configure(values=labels)
        idx = min(self.current_nca_step_idx.get(), max(0, n - 1))
        self.current_nca_step_idx.set(idx)
        if labels:
            self.step_combo.current(idx)

        self.step_scale.configure(from_=0, to=max(0, n - 1))
        self.step_scale.set(idx)

    def _puzzle_label(self, idx: int, path: Path) -> str:
        meta = _safe_json_load(path / "puzzle_metadata.json")
        puzzle_idx = self._extract_puzzle_idx_from_folder(path)
        side = meta.get("side", "?")
        variant = meta.get("variant_id", "")
        return f"{idx + 1}/{len(self.puzzle_dirs)} — puzzle_{puzzle_idx:04d} — {side}x{side} — {variant}"

    def _extract_puzzle_idx_from_folder(self, path: Path) -> int:
        match = re.search(r"puzzle_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    def load_current_puzzle(self) -> None:
        if not self.puzzle_dirs:
            self.clear_view()
            return

        pidx = max(0, min(self.current_puzzle_idx.get(), len(self.puzzle_dirs) - 1))
        self.current_puzzle_idx.set(pidx)

        puzzle_dir = self.puzzle_dirs[pidx]
        self.current_puzzle_metadata = _safe_json_load(puzzle_dir / "puzzle_metadata.json")

        self.checkpoint_dirs = sorted(
            [
                p for p in puzzle_dir.glob("step_*")
                if p.is_dir() and (p / "trace.npy").exists()
            ],
            key=lambda p: (_step_number_from_name(p.name), p.name),
        )

        if not self.checkpoint_dirs:
            messagebox.showerror("No checkpoints", f"No step_* folders with trace.npy found in:\n{puzzle_dir}")
            self.clear_view()
            return

        self.current_checkpoint_idx.set(0)
        self.current_nca_step_idx.set(0)

        self._refresh_puzzle_combo()
        self._refresh_checkpoint_combo()
        self.load_current_trace()

    def load_current_trace(self) -> None:
        if not self.checkpoint_dirs:
            self.clear_view()
            return

        cidx = max(0, min(self.current_checkpoint_idx.get(), len(self.checkpoint_dirs) - 1))
        self.current_checkpoint_idx.set(cidx)

        ckpt_dir = self.checkpoint_dirs[cidx]
        trace_path = ckpt_dir / "trace.npy"

        try:
            self.current_trace = np.load(trace_path, mmap_mode="r")
        except Exception as exc:
            messagebox.showerror("Could not load trace", f"{trace_path}\n\n{exc}")
            self.current_trace = None
            self.clear_view()
            return

        self.current_metrics = _safe_json_load(ckpt_dir / "metrics.json")
        self.current_nca_step_idx.set(0)

        self._refresh_checkpoint_combo()
        self._refresh_step_combo()

        self.current_views = self._build_views()
        labels = [v["label"] for v in self.current_views]
        self.view_combo.configure(values=labels)

        if labels:
            self.current_view_label.set(labels[0])
        else:
            self.current_view_label.set("")

        self.refresh_view()

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def on_puzzle_combo(self) -> None:
        idx = self.puzzle_combo.current()
        if idx >= 0:
            self.current_puzzle_idx.set(idx)
            self.load_current_puzzle()

    def on_checkpoint_combo(self) -> None:
        idx = self.checkpoint_combo.current()
        if idx >= 0:
            self.current_checkpoint_idx.set(idx)
            self.load_current_trace()

    def on_step_combo(self) -> None:
        idx = self.step_combo.current()
        if idx >= 0:
            self.current_nca_step_idx.set(idx)
            self.step_scale.set(idx)
            self.refresh_view()

    def on_step_scale(self, value: str) -> None:
        if self.current_trace is None:
            return
        idx = int(round(float(value)))
        idx = max(0, min(idx, self.current_trace.shape[0] - 1))
        if idx != self.current_nca_step_idx.get():
            self.current_nca_step_idx.set(idx)
            if self.step_combo["values"]:
                self.step_combo.current(idx)
            self.refresh_view()

    def prev_puzzle(self) -> None:
        if not self.puzzle_dirs:
            return
        self.current_puzzle_idx.set(max(0, self.current_puzzle_idx.get() - 1))
        self.load_current_puzzle()

    def next_puzzle(self) -> None:
        if not self.puzzle_dirs:
            return
        self.current_puzzle_idx.set(min(len(self.puzzle_dirs) - 1, self.current_puzzle_idx.get() + 1))
        self.load_current_puzzle()

    def prev_checkpoint(self) -> None:
        if not self.checkpoint_dirs:
            return
        self.current_checkpoint_idx.set(max(0, self.current_checkpoint_idx.get() - 1))
        self.load_current_trace()

    def next_checkpoint(self) -> None:
        if not self.checkpoint_dirs:
            return
        self.current_checkpoint_idx.set(min(len(self.checkpoint_dirs) - 1, self.current_checkpoint_idx.get() + 1))
        self.load_current_trace()

    def prev_nca_step(self) -> None:
        if self.current_trace is None:
            return
        idx = max(0, self.current_nca_step_idx.get() - 1)
        self.current_nca_step_idx.set(idx)
        self.step_scale.set(idx)
        if self.step_combo["values"]:
            self.step_combo.current(idx)
        self.refresh_view()

    def next_nca_step(self) -> None:
        if self.current_trace is None:
            return
        idx = min(self.current_trace.shape[0] - 1, self.current_nca_step_idx.get() + 1)
        self.current_nca_step_idx.set(idx)
        self.step_scale.set(idx)
        if self.step_combo["values"]:
            self.step_combo.current(idx)
        self.refresh_view()

    def prev_view(self) -> None:
        if not self.current_views:
            return
        labels = [v["label"] for v in self.current_views]
        try:
            idx = labels.index(self.current_view_label.get())
        except ValueError:
            idx = 0
        idx = max(0, idx - 1)
        self.current_view_label.set(labels[idx])
        self.refresh_view()

    def next_view(self) -> None:
        if not self.current_views:
            return
        labels = [v["label"] for v in self.current_views]
        try:
            idx = labels.index(self.current_view_label.get())
        except ValueError:
            idx = 0
        idx = min(len(labels) - 1, idx + 1)
        self.current_view_label.set(labels[idx])
        self.refresh_view()

    # ------------------------------------------------------------------
    # Channel names and views
    # ------------------------------------------------------------------

    def _build_channel_names(self) -> list[str]:
        channel_slices = self.trace_manifest.get("channel_slices", {})
        constants = self.trace_manifest.get("model_channel_constants", {})
        dataset_manifest = self.trace_manifest.get("dataset_manifest", {})

        n_number = int(constants.get("NUMBER_CHANNELS", 9))
        n_given = int(constants.get("GIVEN_MASK_CHANNELS", 1))
        n_board = int(constants.get("BOARD_MASK_CHANNELS", 1))
        n_rule = int(constants.get("RULE_CHANNELS", 0))
        n_hidden = int(constants.get("HIDDEN_CHANNELS", 0))
        n_target = int(constants.get("TARGET_CHANNELS", 9))

        total_channels = (
            n_number + n_given + n_board + n_rule + n_hidden + n_target
        )

        # If the dataset manifest already has full channel names, use them.
        full_names = dataset_manifest.get("channel_names")
        if isinstance(full_names, list) and len(full_names) == total_channels:
            return [str(x) for x in full_names]

        names: list[str] = []
        names.extend([f"number_digit_{i}" for i in range(1, n_number + 1)])
        names.extend(["given_mask" if n_given == 1 else f"given_mask_{i}" for i in range(n_given)])
        names.extend(["board_mask" if n_board == 1 else f"board_mask_{i}" for i in range(n_board)])

        rule_names = dataset_manifest.get("rule_channel_names")
        if isinstance(rule_names, list) and len(rule_names) == n_rule:
            names.extend([str(x) for x in rule_names])
        else:
            names.extend([f"rule_channel_{i:03d}" for i in range(n_rule)])

        names.extend([f"hidden_{i:03d}" for i in range(n_hidden)])
        names.extend([f"target_digit_{i}" for i in range(1, n_target + 1)])

        # Pad if needed.
        while len(names) < total_channels:
            names.append(f"channel_{len(names):03d}")

        return names

    def _build_views(self) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []

        if self.current_trace is None:
            return views

        c = int(self.current_trace.shape[-1])
        for i in range(c):
            name = self.channel_names[i] if i < len(self.channel_names) else f"channel_{i:03d}"
            group = self._channel_group(i)
            views.append(
                {
                    "label": f"{i:03d}: [{group}] {name}",
                    "type": "channel",
                    "channel_index": i,
                }
            )

        # Derived views from the current trace step.
        views.append({"label": "derived: predicted digits", "type": "derived_prediction"})
        views.append({"label": "derived: prediction confidence", "type": "derived_confidence"})
        views.append({"label": "derived: digit entropy", "type": "derived_entropy"})

        # Static views saved once per puzzle.
        views.append({"label": "static: givens active", "type": "static_file", "filename": "givens_active.npy"})
        views.append({"label": "static: target active", "type": "static_file", "filename": "target_active.npy"})
        views.append({"label": "static: board mask 9x9", "type": "static_file", "filename": "board_mask_9x9.npy"})
        views.append({"label": "static: predict mask 9x9", "type": "static_file", "filename": "predict_mask_9x9.npy"})

        return views

    def _channel_group(self, ch: int) -> str:
        channel_slices = self.trace_manifest.get("channel_slices", {})
        for name, bounds in channel_slices.items():
            s = _slice_from_list(bounds)
            if s.start <= ch < s.stop:
                return name
        return "unknown"

    # ------------------------------------------------------------------
    # Current data extraction
    # ------------------------------------------------------------------

    def _current_trace_step(self) -> np.ndarray:
        if self.current_trace is None:
            raise RuntimeError("No trace loaded.")
        idx = self.current_nca_step_idx.get()
        idx = max(0, min(idx, self.current_trace.shape[0] - 1))
        return np.asarray(self.current_trace[idx], dtype=np.float32)

    def _actual_nca_step_from_saved_index(self, saved_idx: int) -> int:
        if self.saved_step_indices and saved_idx < len(self.saved_step_indices):
            return int(self.saved_step_indices[saved_idx])

        save_every = int(self.trace_manifest.get("save_every_nca_step", 1))
        save_initial = bool(self.trace_manifest.get("save_initial_state", True))

        if save_initial:
            return saved_idx * save_every
        return (saved_idx + 1) * save_every

    def _get_active_side_offset(self) -> tuple[int, tuple[int, int]]:
        meta = self.current_puzzle_metadata or {}
        side = int(meta.get("side", 9))
        board_offset = meta.get("board_offset", [0, 0])
        if isinstance(board_offset, (list, tuple)) and len(board_offset) == 2:
            offset = (int(board_offset[0]), int(board_offset[1]))
        else:
            offset = (0, 0)
        return side, offset

    def _get_selected_matrix(self, view: dict[str, Any]) -> np.ndarray:
        view_type = view["type"]

        if view_type == "channel":
            trace_step = self._current_trace_step()
            return trace_step[..., int(view["channel_index"])]

        if view_type == "derived_prediction":
            trace_step = self._current_trace_step()
            number_slice = _slice_from_list(self.trace_manifest["channel_slices"]["number_channels"])
            logits = trace_step[..., number_slice]
            return (np.argmax(logits, axis=-1) + 1).astype(np.float32)

        if view_type == "derived_confidence":
            trace_step = self._current_trace_step()
            number_slice = _slice_from_list(self.trace_manifest["channel_slices"]["number_channels"])
            logits = trace_step[..., number_slice]
            probs = self._softmax(logits, axis=-1)
            return np.max(probs, axis=-1).astype(np.float32)

        if view_type == "derived_entropy":
            trace_step = self._current_trace_step()
            number_slice = _slice_from_list(self.trace_manifest["channel_slices"]["number_channels"])
            logits = trace_step[..., number_slice]
            probs = self._softmax(logits, axis=-1)
            entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)
            return entropy.astype(np.float32)

        if view_type == "static_file":
            puzzle_dir = self.puzzle_dirs[self.current_puzzle_idx.get()]
            path = puzzle_dir / view["filename"]
            if not path.exists():
                raise FileNotFoundError(f"Could not find static file: {path}")
            return np.asarray(np.load(path), dtype=np.float32)

        raise ValueError(f"Unknown view type: {view_type}")

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)

    # ------------------------------------------------------------------
    # Refresh view
    # ------------------------------------------------------------------

    def refresh_view(self) -> None:
        if self.current_trace is None or not self.current_views:
            self.clear_view()
            return

        label = self.current_view_label.get()
        view = next((v for v in self.current_views if v["label"] == label), None)
        if view is None:
            view = self.current_views[0]
            self.current_view_label.set(view["label"])

        try:
            matrix = self._get_selected_matrix(view)
        except Exception as exc:
            messagebox.showerror("Could not show view", str(exc))
            return

        side, offset = self._get_active_side_offset()
        self._draw_matrix(matrix, side, offset, self._make_title(view))

        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert(tk.END, self._matrix_to_text(matrix))

        self._update_labels(matrix, view)
        self._update_metadata_text()
        self._sync_comboboxes()

    def _sync_comboboxes(self) -> None:
        if self.puzzle_combo["values"]:
            self.puzzle_combo.current(self.current_puzzle_idx.get())
        if self.checkpoint_combo["values"]:
            self.checkpoint_combo.current(self.current_checkpoint_idx.get())
        if self.step_combo["values"]:
            self.step_combo.current(self.current_nca_step_idx.get())

    def _make_title(self, view: dict[str, Any]) -> str:
        puzzle_dir = self.puzzle_dirs[self.current_puzzle_idx.get()]
        ckpt_dir = self.checkpoint_dirs[self.current_checkpoint_idx.get()]
        step_idx = self.current_nca_step_idx.get()
        actual_step = self._actual_nca_step_from_saved_index(step_idx)

        puzzle_idx = self._extract_puzzle_idx_from_folder(puzzle_dir)
        return (
            f"Puzzle {puzzle_idx:04d} | {ckpt_dir.name} | "
            f"saved step {step_idx} / NCA step {actual_step}\n"
            f"{view['label']}"
        )

    def _update_labels(self, matrix: np.ndarray, view: dict[str, Any]) -> None:
        pidx = self.current_puzzle_idx.get()
        cidx = self.current_checkpoint_idx.get()
        sidx = self.current_nca_step_idx.get()

        puzzle_dir = self.puzzle_dirs[pidx]
        ckpt_dir = self.checkpoint_dirs[cidx]

        puzzle_idx = self._extract_puzzle_idx_from_folder(puzzle_dir)
        actual_step = self._actual_nca_step_from_saved_index(sidx)

        self.puzzle_label.config(text=f"Puzzle: {pidx + 1}/{len(self.puzzle_dirs)} — puzzle_{puzzle_idx:04d}")
        self.checkpoint_label.config(text=f"Checkpoint: {cidx + 1}/{len(self.checkpoint_dirs)} — {ckpt_dir.name}")
        self.step_label.config(text=f"NCA step: {sidx + 1}/{self.current_trace.shape[0]} — actual {actual_step}")
        self.channel_label.config(text=f"Channel/view: {view['label']}")

        min_val = float(np.min(matrix))
        max_val = float(np.max(matrix))
        nonzero = int(np.count_nonzero(np.abs(matrix) > 1e-8))
        solved = self.current_metrics.get("solved", None)
        board_wrong = self.current_metrics.get("board_wrong", None)
        loss = self.current_metrics.get("loss", None)

        self.info_label.config(
            text=(
                f"shape={matrix.shape}, min={min_val:.4g}, max={max_val:.4g}, "
                f"nonzero={nonzero} | final solved={solved}, "
                f"final board_wrong={board_wrong}, final loss={loss}"
            )
        )

    def _update_metadata_text(self) -> None:
        puzzle_dir = self.puzzle_dirs[self.current_puzzle_idx.get()]
        ckpt_dir = self.checkpoint_dirs[self.current_checkpoint_idx.get()]

        payload = {
            "trace_root": str(self.trace_root),
            "puzzle_folder": puzzle_dir.name,
            "checkpoint_folder": ckpt_dir.name,
            "puzzle_metadata": self.current_puzzle_metadata,
            "checkpoint_metrics": self.current_metrics,
            "trace_shape": list(self.current_trace.shape) if self.current_trace is not None else None,
            "channel_slices": self.trace_manifest.get("channel_slices", {}),
            "saved_step_indices": self.trace_manifest.get("saved_step_indices", []),
        }

        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, json.dumps(payload, indent=2, ensure_ascii=False))

    def clear_view(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("No data loaded")
        self.canvas.draw_idle()

        self.matrix_text.delete("1.0", tk.END)
        self.meta_text.delete("1.0", tk.END)
        self.info_label.config(text="")
        self.puzzle_label.config(text="Puzzle: -")
        self.checkpoint_label.config(text="Checkpoint: -")
        self.step_label.config(text="NCA step: -")
        self.channel_label.config(text="Channel: -")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_matrix(self, matrix: np.ndarray, side: int, offset: tuple[int, int], title: str) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        im = self.ax.imshow(matrix, interpolation="nearest")
        self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)

        self.ax.set_title(title, fontsize=10)
        self.ax.set_xticks(range(matrix.shape[1]))
        self.ax.set_yticks(range(matrix.shape[0]))

        # Write numbers for smaller matrices and for sparse/low-valued matrices.
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = float(matrix[r, c])
                if abs(val) > 1e-8:
                    text = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
                    self.ax.text(c, r, text, ha="center", va="center", fontsize=8)

        rr0, cc0 = offset
        rect = Rectangle((cc0 - 0.5, rr0 - 0.5), side, side, fill=False, edgecolor="red", linewidth=2.5)
        self.ax.add_patch(rect)

        self.ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        self.ax.set_ylim(matrix.shape[0] - 0.5, -0.5)

        self.canvas.draw_idle()

    def _matrix_to_text(self, matrix: np.ndarray) -> str:
        rows = []
        for row in matrix:
            vals = []
            for x in row:
                x = float(x)
                if x.is_integer():
                    vals.append(f"{int(x):>7}")
                else:
                    vals.append(f"{x:>7.3f}")
            rows.append(" ".join(vals))
        return "\n".join(rows)


def main() -> None:
    root = tk.Tk()
    CheckpointTraceViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
