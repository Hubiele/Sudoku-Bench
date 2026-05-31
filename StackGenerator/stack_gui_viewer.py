#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# ==============================================================================
# Configuration: set the default dataset folder here.
# ==============================================================================
# This can be any folder that contains the npy files.
# By default, this script uses the "stack_test" folder in the same directory.
# Example absolute path: DEFAULT_DATASET_DIR = r"C:\Users\Name\Documents\Sudoku\stack_test"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "stack_test"


# ==============================================================================


def _try_import_puzzles_module() -> dict[str, Any]:
    """Try to import puzzle constants from nearby project files."""
    candidates = [
        Path.cwd() / "puzzles.py",
        Path.cwd() / "StackGenerator" / "puzzles.py",
        Path(__file__).resolve().parent / "puzzles.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("puzzles_runtime", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return {
                "NUMBER_CHANNELS": int(getattr(mod, "NUMBER_CHANNELS", 9)),
                "GIVEN_MASK_CHANNELS": int(getattr(mod, "GIVEN_MASK_CHANNELS", 1)),
                "BOARD_MASK_CHANNELS": int(getattr(mod, "BOARD_MASK_CHANNELS", 1)),
                "RULE_CHANNELS": int(getattr(mod, "RULE_CHANNELS", 0)),
                "HIDDEN_CHANNELS": int(getattr(mod, "HIDDEN_CHANNELS", 0)),
                "TARGET_CHANNELS": int(getattr(mod, "TARGET_CHANNELS", 9)),
                "STACK_SIDE": int(getattr(mod, "STACK_SIDE", getattr(mod, "EMBED_SIDE", 9))),
            }
        except Exception:
            continue
    return {
        "NUMBER_CHANNELS": 9,
        "GIVEN_MASK_CHANNELS": 1,
        "BOARD_MASK_CHANNELS": 1,
        "RULE_CHANNELS": 0,
        "HIDDEN_CHANNELS": 0,
        "TARGET_CHANNELS": 9,
        "STACK_SIDE": 9,
    }


PUZZLES_CONSTS = _try_import_puzzles_module()


class StackDatasetViewer:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sudoku Stack Viewer")
        self.root.geometry("1500x900")

        self.dataset_root: Path | None = None
        self.dataset_mode: str | None = None  # "npy" or "json"

        # New .npy dataset, including the test split.
        self.np_data_by_split: dict[str, np.ndarray | None] = {"train": None, "validation": None, "test": None}
        self.metadata_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.manifest: dict[str, Any] = {}

        # Old JSON dataset fallback
        self.files_by_split: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}

        self.current_split = tk.StringVar(value="train")
        self.current_index = tk.IntVar(value=0)
        self.current_view_label = tk.StringVar(value="")

        self.current_record: dict[str, Any] | None = None
        self.current_views: list[dict[str, Any]] = []
        self.channel_names: list[str] = []

        self._build_ui()
        self._refresh_controls_state()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open dataset folder", command=self.choose_dataset_root).pack(side=tk.LEFT)
        self.root_label = ttk.Label(top, text="No folder selected")
        self.root_label.pack(side=tk.LEFT, padx=10)

        nav = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        nav.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(nav, text="Split:").pack(side=tk.LEFT)
        self.split_combo = ttk.Combobox(
            nav,
            textvariable=self.current_split,
            state="readonly",
            width=12,
            values=("train", "validation", "test"),  # Include the test split.
        )
        self.split_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.split_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_split_changed())

        ttk.Button(nav, text="Previous stack", command=self.prev_item).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next stack", command=self.next_item).pack(side=tk.LEFT, padx=(4, 12))

        self.file_label = ttk.Label(nav, text="No files")
        self.file_label.pack(side=tk.LEFT)

        ctrl = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Tensor/layer:").pack(side=tk.LEFT)
        self.view_combo = ttk.Combobox(
            ctrl,
            textvariable=self.current_view_label,
            state="readonly",
            width=52,
            values=(),
        )
        self.view_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.view_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_view())

        ttk.Button(ctrl, text="Previous layer", command=self.prev_view).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Next layer", command=self.next_view).pack(side=tk.LEFT, padx=(4, 12))

        self.layer_label = ttk.Label(ctrl, text="Layer: -")
        self.layer_label.pack(side=tk.LEFT)

        info = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        info.pack(side=tk.TOP, fill=tk.X)
        self.info_label = ttk.Label(info, text="")
        self.info_label.pack(side=tk.LEFT)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=2)
        main.add(right, weight=1)

        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        text_top = ttk.Frame(right)
        text_top.pack(fill=tk.BOTH, expand=True)

        self.matrix_text = tk.Text(text_top, wrap=tk.NONE, font=("Courier New", 10))
        self.matrix_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(text_top, orient=tk.VERTICAL, command=self.matrix_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.matrix_text.configure(yscrollcommand=yscroll.set)

        meta_frame = ttk.LabelFrame(right, text="Metadata", padding=8)
        meta_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.meta_text = tk.Text(meta_frame, wrap=tk.WORD, height=16, font=("Courier New", 9))
        self.meta_text.pack(fill=tk.BOTH, expand=True)

    def choose_dataset_root(self) -> None:
        selected = filedialog.askdirectory(title="Select dataset root")
        if not selected:
            return
        self.load_dataset_root(Path(selected))

    def load_dataset_root(self, root: Path) -> None:
        # Prefer the new .npy format, including test_stacks.npy.
        train_npy = root / "train_stacks.npy"
        val_npy = root / "validation_stacks.npy"
        test_npy = root / "test_stacks.npy"
        if train_npy.exists() or val_npy.exists() or test_npy.exists():
            self._load_npy_dataset(root)
            return

        # Fallback old JSON split directories
        train_files = sorted((root / "train").glob("*.json")) if (root / "train").exists() else []
        val_files = sorted((root / "validation").glob("*.json")) if (root / "validation").exists() else []
        test_files = sorted((root / "test").glob("*.json")) if (root / "test").exists() else []
        if not train_files and not val_files and not test_files:
            messagebox.showerror(
                "No data",
                f"Could not find either an .npy dataset or JSON split in {root}",
            )
            return

        self.dataset_mode = "json"
        self.dataset_root = root
        self.files_by_split = {"train": train_files, "validation": val_files, "test": test_files}
        self.np_data_by_split = {"train": None, "validation": None, "test": None}
        self.metadata_by_split = {"train": [], "validation": [], "test": []}
        self.manifest = {}
        self.root_label.config(text=f"{root}  [json]")

        # Select the first available split.
        if train_files:
            self.current_split.set("train")
        elif val_files:
            self.current_split.set("validation")
        else:
            self.current_split.set("test")

        self.current_index.set(0)
        self.load_current_record()

    def _load_npy_dataset(self, root: Path) -> None:
        self.dataset_mode = "npy"
        self.dataset_root = root
        self.files_by_split = {"train": [], "validation": []}
        self.manifest = {}
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                self.manifest = json.load(f)

        # Load each split if the corresponding files exist.
        for split in ("train", "validation", "test"):
            stack_path = root / f"{split}_stacks.npy"
            meta_path = root / f"{split}_metadata.json"
            self.np_data_by_split[split] = np.load(stack_path, mmap_mode="r") if stack_path.exists() else None
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    self.metadata_by_split[split] = json.load(f)
            else:
                self.metadata_by_split[split] = []

        if self.np_data_by_split["train"] is None and self.np_data_by_split["validation"] is None and \
                self.np_data_by_split["test"] is None:
            raise FileNotFoundError(f"Could not find any *_stacks.npy in {root}")

        self.root_label.config(text=f"{root}  [npy]")

        if self.np_data_by_split["train"] is not None:
            self.current_split.set("train")
        elif self.np_data_by_split["validation"] is not None:
            self.current_split.set("validation")
        else:
            self.current_split.set("test")

        self.current_index.set(0)
        self.load_current_record()

    def _refresh_controls_state(self) -> None:
        split = self.current_split.get()
        count = self._split_count(split)
        if count == 0:
            self.file_label.config(text=f"{split}: 0 files")
            self.layer_label.config(text="Layer: -")
            return

        idx = max(0, min(self.current_index.get(), count - 1))
        self.current_index.set(idx)
        if self.dataset_mode == "npy":
            self.file_label.config(text=f"{split}: {idx + 1}/{count} — stack_{idx:06d}")
        else:
            files = self.files_by_split.get(split, [])
            self.file_label.config(text=f"{split}: {idx + 1}/{count} — {files[idx].name}")

    def _split_count(self, split: str) -> int:
        if self.dataset_mode == "npy":
            arr = self.np_data_by_split.get(split)
            return 0 if arr is None else int(arr.shape[0])
        return len(self.files_by_split.get(split, []))

    def on_split_changed(self) -> None:
        self.current_index.set(0)
        self.load_current_record()

    def prev_item(self) -> None:
        if self._split_count(self.current_split.get()) == 0:
            return
        self.current_index.set(max(0, self.current_index.get() - 1))
        self.load_current_record()

    def next_item(self) -> None:
        count = self._split_count(self.current_split.get())
        if count == 0:
            return
        self.current_index.set(min(count - 1, self.current_index.get() + 1))
        self.load_current_record()

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

    def load_current_record(self) -> None:
        self._refresh_controls_state()
        split = self.current_split.get()
        idx = self.current_index.get()

        if self._split_count(split) == 0:
            self.current_record = None
            self.current_views = []
            self.channel_names = []
            self.clear_view()
            return

        if self.dataset_mode == "npy":
            self.current_record = self._load_npy_record(split, idx)
        else:
            file_path = self.files_by_split[split][idx]
            with file_path.open("r", encoding="utf-8") as f:
                self.current_record = json.load(f)

        self.channel_names = self._extract_channel_names(self.current_record)
        self.current_views = self._build_views(self.current_record)
        labels = [v["label"] for v in self.current_views]
        self.view_combo.configure(values=labels)
        self.current_view_label.set(labels[0] if labels else "")
        self.refresh_view()

    def _load_npy_record(self, split: str, idx: int) -> dict[str, Any]:
        # Read one stack from the selected split.
        stack_arr = self.np_data_by_split[split]
        assert stack_arr is not None
        stack = np.asarray(stack_arr[idx], dtype=np.float32)  # (H,W,C)
        stack_chw = np.transpose(stack, (2, 0, 1)).tolist()
        metadata = {}
        meta_list = self.metadata_by_split.get(split, [])
        if idx < len(meta_list):
            metadata = meta_list[idx]

        consts = {
            **PUZZLES_CONSTS,
            "NUMBER_CHANNELS": int(metadata.get("number_channels", PUZZLES_CONSTS["NUMBER_CHANNELS"])),
            "GIVEN_MASK_CHANNELS": int(metadata.get("given_mask_channels", PUZZLES_CONSTS["GIVEN_MASK_CHANNELS"])),
            "BOARD_MASK_CHANNELS": int(metadata.get("board_mask_channels", PUZZLES_CONSTS["BOARD_MASK_CHANNELS"])),
            "RULE_CHANNELS": int(metadata.get("rule_channels", PUZZLES_CONSTS["RULE_CHANNELS"])),
            "HIDDEN_CHANNELS": int(metadata.get("hidden_channels", PUZZLES_CONSTS["HIDDEN_CHANNELS"])),
            "TARGET_CHANNELS": int(metadata.get("target_channels", PUZZLES_CONSTS["TARGET_CHANNELS"])),
        }
        n_num = consts["NUMBER_CHANNELS"]
        n_given = consts["GIVEN_MASK_CHANNELS"]
        n_board = consts["BOARD_MASK_CHANNELS"]
        n_rule = consts["RULE_CHANNELS"]
        n_hidden = consts["HIDDEN_CHANNELS"]
        n_target = consts["TARGET_CHANNELS"]

        given_mask_start = n_num
        board_mask_start = n_num + n_given
        target_start = stack.shape[-1] - n_target

        given_mask = stack[..., given_mask_start]
        board_mask = stack[..., board_mask_start]
        solution_one_hot = stack[..., target_start:]
        target = np.argmax(solution_one_hot, axis=-1).astype(np.int32)
        predict_mask = board_mask * (1.0 - given_mask)
        input_stack = stack[..., :target_start]
        input_stack_chw = np.transpose(input_stack, (2, 0, 1)).tolist()

        channel_names = metadata.get("channel_names") or self.manifest.get(
            "channel_names") or self._build_default_channel_names(consts)
        metadata = dict(metadata)
        metadata.setdefault("channel_names", channel_names)
        metadata.setdefault("number_channels", n_num)
        metadata.setdefault("given_mask_channels", n_given)
        metadata.setdefault("board_mask_channels", n_board)
        metadata.setdefault("rule_channels", n_rule)
        metadata.setdefault("hidden_channels", n_hidden)
        metadata.setdefault("target_channels", n_target)

        return {
            "stack": stack_chw,
            "input_stack": input_stack_chw,
            "target": target.tolist(),
            "solution_one_hot": np.transpose(solution_one_hot, (2, 0, 1)).tolist(),
            "board_mask": board_mask.tolist(),
            "given_mask": given_mask.tolist(),
            "predict_mask": predict_mask.tolist(),
            "metadata": metadata,
            "channel_names": channel_names,
        }

    def _build_default_channel_names(self, consts: dict[str, int]) -> list[str]:
        names: list[str] = []
        names.extend([f"given_digit_{d}" for d in range(1, consts["NUMBER_CHANNELS"] + 1)])
        names.extend(["given_mask"] * consts["GIVEN_MASK_CHANNELS"])
        names.extend(["board_mask"] * consts["BOARD_MASK_CHANNELS"])
        rule_channel_names = self.manifest.get("rule_channel_names")
        if isinstance(rule_channel_names, list) and len(rule_channel_names) == consts["RULE_CHANNELS"]:
            names.extend([str(x) for x in rule_channel_names])
        else:
            names.extend([f"rule_channel_{i:03d}" for i in range(consts["RULE_CHANNELS"])])
        names.extend([f"hidden_{i:03d}" for i in range(consts["HIDDEN_CHANNELS"])])
        names.extend([f"solution_digit_{d}" for d in range(1, consts["TARGET_CHANNELS"] + 1)])
        return names

    def _extract_channel_names(self, record: dict[str, Any]) -> list[str]:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            channel_names = metadata.get("channel_names")
            if isinstance(channel_names, list) and channel_names:
                return [str(x) for x in channel_names]
        channel_names = record.get("channel_names")
        if isinstance(channel_names, list) and channel_names:
            return [str(x) for x in channel_names]
        return self._build_default_channel_names(PUZZLES_CONSTS)

    def _build_views(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        # Prefer full stack channels when they are available.
        stack = record.get("stack")
        if isinstance(stack, list):
            arr = np.asarray(stack, dtype=np.float32)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    ch_name = self.channel_names[i] if i < len(self.channel_names) else f"channel_{i}"
                    views.append({
                        "label": f"{i:03d}: {ch_name}",
                        "tensor_name": "stack",
                        "layer_index": i,
                    })
                return views

        input_stack = record.get("input_stack")
        if isinstance(input_stack, list):
            arr = np.asarray(input_stack, dtype=np.float32)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    ch_name = self.channel_names[i] if i < len(self.channel_names) else f"channel_{i}"
                    views.append({
                        "label": f"{i:03d}: {ch_name}",
                        "tensor_name": "input_stack",
                        "layer_index": i,
                    })
        for tensor_name in ("target", "board_mask", "given_mask", "predict_mask"):
            value = record.get(tensor_name)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 2:
                views.append({"label": tensor_name, "tensor_name": tensor_name, "layer_index": None})
        return views

    def clear_view(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("No data")
        self.canvas.draw_idle()
        self.matrix_text.delete("1.0", tk.END)
        self.meta_text.delete("1.0", tk.END)
        self.info_label.config(text="")

    def _get_side_and_offset(self, record: dict[str, Any]) -> tuple[int, tuple[int, int]]:
        metadata = record.get("metadata", {})
        side = 9
        offset = (0, 0)
        if isinstance(metadata, dict):
            side = int(metadata.get("side", metadata.get("grid_side", 9)))
            board_offset = metadata.get("board_offset", [0, 0])
            if isinstance(board_offset, (list, tuple)) and len(board_offset) == 2:
                offset = (int(board_offset[0]), int(board_offset[1]))
        return side, offset

    def _tensor_to_array(self, record: dict[str, Any], tensor_name: str) -> np.ndarray:
        value = record.get(tensor_name)
        if value is None:
            raise KeyError(f"Could not find tensor {tensor_name!r} in the stack file")
        return np.asarray(value, dtype=np.float32)

    def _matrix_to_text(self, matrix: np.ndarray) -> str:
        rows = []
        for row in matrix:
            vals = []
            for x in row:
                if float(x).is_integer():
                    vals.append(f"{int(x):>5}")
                else:
                    vals.append(f"{x:>5.3f}")
            rows.append(" ".join(vals))
        return "\n".join(rows)

    def refresh_view(self) -> None:
        if self.current_record is None or not self.current_views:
            self.clear_view()
            return

        label = self.current_view_label.get()
        selected = next((v for v in self.current_views if v["label"] == label), None)
        if selected is None:
            selected = self.current_views[0]
            self.current_view_label.set(selected["label"])

        tensor_name = selected["tensor_name"]
        layer_index = selected["layer_index"]
        arr = self._tensor_to_array(self.current_record, tensor_name)
        side, offset = self._get_side_and_offset(self.current_record)

        if layer_index is None:
            matrix = arr
            self.layer_label.config(text=f"Layer: {tensor_name}")
        else:
            matrix = arr[layer_index]
            self.layer_label.config(text=f"Layer: {selected['label']}")

        self._draw_matrix(matrix, side, offset, selected["label"])
        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert(tk.END, self._matrix_to_text(matrix))

        min_val = float(np.min(matrix))
        max_val = float(np.max(matrix))
        nonzero = int(np.count_nonzero(matrix))
        self.info_label.config(text=f"shape={matrix.shape}, min={min_val:.4g}, max={max_val:.4g}, nonzero={nonzero}")

        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, json.dumps(self.current_record.get("metadata", {}), indent=2, ensure_ascii=False))
        self._refresh_controls_state()

    def _draw_matrix(self, matrix: np.ndarray, side: int, offset: tuple[int, int], title: str) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(matrix, interpolation="nearest")
        self.ax.set_title(title)
        self.ax.set_xticks(range(matrix.shape[1]))
        self.ax.set_yticks(range(matrix.shape[0]))

        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                if abs(float(val)) > 1e-8:
                    text = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
                    self.ax.text(c, r, text, ha="center", va="center", fontsize=8)

                    rr0, cc0 = offset
                    rect = Rectangle((cc0 - 0.5, rr0 - 0.5), side, side, fill=False, edgecolor="red", linewidth=2.5)
                    self.ax.add_patch(rect)
                    self.ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
                    self.ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
                    self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    viewer = StackDatasetViewer(root)

    # Put the configured folder first in the list of automatic search locations.
    default_roots = [
        Path(DEFAULT_DATASET_DIR),
        Path.cwd() / "stack_dataset",
        Path.cwd() / "StackGenerator" / "stack_dataset",
        Path.cwd(),
    ]
    for candidate in default_roots:
        try:
            if (candidate / "train_stacks.npy").exists() or \
                    (candidate / "validation_stacks.npy").exists() or \
                    (candidate / "test_stacks.npy").exists() or \
                    (candidate / "train").exists() or \
                    (candidate / "validation").exists() or \
                    (candidate / "test").exists():
                viewer.load_dataset_root(candidate)
                break
        except Exception:
            pass

    root.mainloop()


if __name__ == "__main__":
    main()