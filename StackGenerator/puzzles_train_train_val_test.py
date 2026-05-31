from __future__ import annotations

import json
import random
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

NUMBER_CHANNELS = 9
GIVEN_MASK_CHANNELS = 1
BOARD_MASK_CHANNELS = 1
RULE_CHANNELS = 36
HIDDEN_CHANNELS = 96
TARGET_CHANNELS = 9

STACK_SIDE = 9
DATASET_INPUT_DIR = "/home/daniel/Documents/Skole/Masteroppgave/Kode/Datasets/train_generated_puzzle_4x4_9x9/"
SCHEMA_FILE_NAME = "shared_rule_channel_schema_4x4_new_arch_only.json"

DEFAULT_VALIDATION_COUNT = 256
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 0
DEFAULT_LOG_EVERY = 25


# ---------------------------------------------------------------------------
# Paths / schema
# ---------------------------------------------------------------------------

def _default_dataset_dir(dataset_dir: str | Path | None = None) -> Path:
    return DATASET_INPUT_DIR if dataset_dir is None else Path(dataset_dir)


def _schema_path(dataset_dir: str | Path | None = None, schema_path: str | Path | None = None) -> Path:
    if schema_path is not None:
        path = Path(schema_path)
        if not path.exists():
            raise FileNotFoundError(f"Fant ikke schemafil: {path}")
        return path

    dataset_root = _default_dataset_dir(dataset_dir)
    direct = dataset_root / SCHEMA_FILE_NAME
    if direct.exists():
        return direct

    candidates = sorted(dataset_root.glob("shared_rule_channel_schema*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Fant ingen schemafil i {dataset_root}. Forventet minst {SCHEMA_FILE_NAME!r}."
        )
    return candidates[0]


def _load_schema(dataset_dir: str | Path | None = None, schema_path: str | Path | None = None) -> dict[str, Any]:
    path = _schema_path(dataset_dir, schema_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalize_grid(grid: list[list[int]]) -> list[list[int]]:
    return [[int(x) for x in row] for row in grid]


def _board_offset(record: dict[str, Any]) -> tuple[int, int]:
    offset = record.get("board_offset", (0, 0))
    return int(offset[0]), int(offset[1])


def _embed_coord(local_coord: tuple[int, int], board_offset: tuple[int, int]) -> tuple[int, int]:
    return local_coord[0] + board_offset[0], local_coord[1] + board_offset[1]


def _normalize_rule_name(rule_name: str, schema_rules: dict[str, Any]) -> str:
    if rule_name in schema_rules:
        return rule_name
    if rule_name.startswith("standard_sudoku") and "standard_sudoku" in schema_rules:
        return "standard_sudoku"
    return rule_name


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(max(0, num_bytes))
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def _default_dataset_file_names() -> list[str]:
    return [
        "manifest.json",
        "train_stacks.npy",
        "validation_stacks.npy",
        "train_targets.npy",
        "validation_targets.npy",
        "train_metadata.json",
        "validation_metadata.json",
    ]


def _default_archive_path(dataset_dir: str | Path) -> Path:
    dataset_dir = Path(dataset_dir)
    return dataset_dir.parent / f"{dataset_dir.name}.tar.gz"


def _dataset_required_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return [dataset_dir / name for name in _default_dataset_file_names()]

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    names = {"manifest.json"}
    for key in (
        "train_stacks_file",
        "validation_stacks_file",
        "train_targets_file",
        "validation_targets_file",
        "train_metadata_file",
        "validation_metadata_file",
    ):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            names.add(value)
    return [dataset_dir / name for name in sorted(names)]


def dataset_is_complete(dataset_dir: str | Path) -> bool:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False
    return all(path.exists() for path in _dataset_required_files(dataset_dir))


def compress_dataset_directory(
    dataset_dir: str | Path,
    archive_path: str | Path | None = None,
    *,
    compression: str = "gz",
    overwrite: bool = True,
    log: bool = True,
) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Fant ikke dataset-mappe: {dataset_dir}")

    archive_path = _default_archive_path(dataset_dir) if archive_path is None else Path(archive_path).resolve()
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Arkiv finnes allerede: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    members = sorted([p for p in dataset_dir.rglob("*") if p.is_file()])
    if not members:
        raise ValueError(f"Fant ingen filer å komprimere i {dataset_dir}")

    total_bytes = sum(p.stat().st_size for p in members)
    if log:
        print(
            f"[compress] Starter komprimering av {len(members)} filer fra {dataset_dir} "
            f"til {archive_path} | total={_format_bytes(total_bytes)}"
        )

    mode = f"w:{compression}" if compression else "w"
    t0 = time.time()
    done_bytes = 0
    with tarfile.open(archive_path, mode) as tar:
        for i, member in enumerate(members, start=1):
            arcname = member.relative_to(dataset_dir.parent)
            tar.add(member, arcname=str(arcname), recursive=False)
            size = member.stat().st_size
            done_bytes += size
            if log:
                elapsed = time.time() - t0
                avg = elapsed / i
                remaining = avg * (len(members) - i)
                print(
                    f"[compress] {i}/{len(members)} | "
                    f"{_format_bytes(done_bytes)}/{_format_bytes(total_bytes)} | "
                    f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(remaining)} | "
                    f"siste={member.name}"
                )

    archive_size = archive_path.stat().st_size
    ratio = archive_size / total_bytes if total_bytes else 0.0
    if log:
        print(
            f"[compress] Ferdig | archive={archive_path} | size={_format_bytes(archive_size)} | "
            f"ratio={ratio:.3f}"
        )
    return archive_path


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, output_root: Path) -> None:
    output_root_resolved = output_root.resolve()
    target_path = (output_root / member.name).resolve()
    if output_root_resolved not in target_path.parents and target_path != output_root_resolved:
        raise ValueError(f"Usikker sti i arkivmedlem: {member.name}")
    tar.extract(member, path=output_root)


def extract_dataset_archive(
    archive_path: str | Path,
    output_root: str | Path | None = None,
    *,
    overwrite: bool = False,
    log: bool = True,
) -> Path:
    archive_path = Path(archive_path).resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Fant ikke arkiv: {archive_path}")

    output_root = archive_path.parent if output_root is None else Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as tar:
        members = tar.getmembers()
        top_levels = sorted({member.name.split("/", 1)[0] for member in members if member.name})
        if len(top_levels) != 1:
            raise ValueError(f"Forventer ett toppnivå i dataset-arkivet, fant: {top_levels}")
        dataset_dir = output_root / top_levels[0]

        if dataset_dir.exists() and overwrite:
            for path in sorted(dataset_dir.rglob('*'), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if dataset_dir.exists():
                dataset_dir.rmdir()

        file_members = [m for m in members if m.isfile()]
        total_bytes = sum(int(m.size) for m in file_members)
        if log:
            print(
                f"[extract] Starter utpakking av {archive_path} til {output_root} | "
                f"filer={len(file_members)} | total={_format_bytes(total_bytes)}"
            )

        t0 = time.time()
        done_files = 0
        done_bytes = 0
        for member in members:
            _safe_extract_member(tar, member, output_root)
            if member.isfile():
                done_files += 1
                done_bytes += int(member.size)
                if log:
                    elapsed = time.time() - t0
                    avg = elapsed / max(1, done_files)
                    remaining = avg * max(0, len(file_members) - done_files)
                    print(
                        f"[extract] {done_files}/{len(file_members)} | "
                        f"{_format_bytes(done_bytes)}/{_format_bytes(total_bytes)} | "
                        f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(remaining)} | "
                        f"siste={member.name}"
                    )

    if log:
        print(f"[extract] Ferdig | dataset_dir={dataset_dir}")
    return dataset_dir


def ensure_dataset_ready(
    dataset_dir: str | Path,
    archive_path: str | Path | None = None,
    *,
    force_extract: bool = False,
    log: bool = True,
) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    archive_path = _default_archive_path(dataset_dir) if archive_path is None else Path(archive_path).resolve()

    if dataset_is_complete(dataset_dir) and not force_extract:
        if log:
            print(f"[dataset] Fant ferdig utpakket dataset i {dataset_dir}")
        return dataset_dir

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Fant verken komplett dataset-mappe ({dataset_dir}) eller arkiv ({archive_path})."
        )

    return extract_dataset_archive(archive_path, output_root=dataset_dir.parent, overwrite=force_extract, log=log)


# ---------------------------------------------------------------------------
# Schema / channels
# ---------------------------------------------------------------------------

def _schema_rule_channel_count(schema: dict[str, Any]) -> int:
    max_idx = -1
    for rule_spec in schema.get("rules", {}).values():
        for channel in rule_spec.get("channels", []):
            max_idx = max(max_idx, int(channel["index_hint"]))
    resolved = max_idx + 1
    if RULE_CHANNELS is None:
        return resolved
    return max(RULE_CHANNELS, resolved)


def _schema_channel_map(schema: dict[str, Any], rule_channel_count: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for rule_spec in schema.get("rules", {}).values():
        for channel in rule_spec.get("channels", []):
            name = str(channel["name"])
            idx = int(channel["index_hint"])
            if idx < 0 or idx >= rule_channel_count:
                raise ValueError(
                    f"Schema channel {name!r} has index_hint={idx}, but rule_channel_count={rule_channel_count}."
                )
            if name in out and out[name] != idx:
                raise ValueError(
                    f"Channel {name!r} has conflicting index_hint values: {out[name]} vs {idx}"
                )
            out[name] = idx
    return out


def _schema_rule_channel_names(schema: dict[str, Any], rule_channel_count: int) -> list[str]:
    names = [f"rule_channel_{i:03d}" for i in range(rule_channel_count)]
    for rule_name, rule_spec in schema.get("rules", {}).items():
        for channel in rule_spec.get("channels", []):
            idx = int(channel["index_hint"])
            name = str(channel["name"])
            if rule_name == "standard_sudoku" and name == "given_mask":
                name = "standard_sudoku_active"
            names[idx] = name
    return names


def _all_channel_names(schema: dict[str, Any], rule_channel_count: int) -> list[str]:
    names: list[str] = []
    names.extend([f"given_digit_{d}" for d in range(1, NUMBER_CHANNELS + 1)])
    names.extend(["given_mask"])
    names.extend(["board_mask"])
    names.extend(_schema_rule_channel_names(schema, rule_channel_count))
    names.extend([f"hidden_{i:03d}" for i in range(HIDDEN_CHANNELS)])
    names.extend([f"solution_digit_{d}" for d in range(1, TARGET_CHANNELS + 1)])
    return names


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_variant_files(dataset_dir: str | Path, schema_path: str | Path | None = None) -> list[Path]:
    dataset_root = _default_dataset_dir(dataset_dir)
    schema_file = _schema_path(dataset_root, schema_path).resolve()
    files: list[Path] = []
    for path in sorted(dataset_root.glob("*.json")):
        if path.resolve() == schema_file:
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Base tensors
# ---------------------------------------------------------------------------

def _board_mask(side: int, board_offset: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((STACK_SIDE, STACK_SIDE), dtype=np.float32)
    rr0, cc0 = board_offset
    mask[rr0:rr0 + side, cc0:cc0 + side] = 1.0
    return mask


def _encode_givens_digit_channels(
    givens: list[list[int]],
    board_offset: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    digit_channels = np.zeros((NUMBER_CHANNELS, STACK_SIDE, STACK_SIDE), dtype=np.float32)
    given_mask = np.zeros((STACK_SIDE, STACK_SIDE), dtype=np.float32)

    for r, row in enumerate(givens):
        for c, value in enumerate(row):
            value = int(value)
            if value <= 0:
                continue
            if value > NUMBER_CHANNELS:
                raise ValueError(f"Given value {value} exceeds NUMBER_CHANNELS={NUMBER_CHANNELS}")
            rr, cc = _embed_coord((r, c), board_offset)
            digit_channels[value - 1, rr, cc] = 1.0
            given_mask[rr, cc] = 1.0

    return digit_channels, given_mask


def _encode_target_indices(
    solution: list[list[int]],
    board_offset: tuple[int, int],
) -> np.ndarray:
    target_digits = np.zeros((STACK_SIDE, STACK_SIDE), dtype=np.int32)
    for r, row in enumerate(solution):
        for c, value in enumerate(row):
            if int(value) > TARGET_CHANNELS:
                raise ValueError(f"Solution value {value} exceeds TARGET_CHANNELS={TARGET_CHANNELS}")
            rr, cc = _embed_coord((r, c), board_offset)
            target_digits[rr, cc] = int(value) - 1
    return target_digits


def _encode_solution_one_hot(
    solution: list[list[int]],
    board_offset: tuple[int, int],
) -> np.ndarray:
    solution_channels = np.zeros((TARGET_CHANNELS, STACK_SIDE, STACK_SIDE), dtype=np.float32)
    for r, row in enumerate(solution):
        for c, value in enumerate(row):
            if int(value) > TARGET_CHANNELS:
                raise ValueError(f"Solution value {value} exceeds TARGET_CHANNELS={TARGET_CHANNELS}")
            rr, cc = _embed_coord((r, c), board_offset)
            solution_channels[int(value) - 1, rr, cc] = 1.0
    return solution_channels


def _empty_hidden_channels() -> np.ndarray:
    return np.zeros((HIDDEN_CHANNELS, STACK_SIDE, STACK_SIDE), dtype=np.float32)


# ---------------------------------------------------------------------------
# Generic rule encoding
# ---------------------------------------------------------------------------

def _empty_rule_channels(rule_channel_count: int) -> np.ndarray:
    return np.zeros((rule_channel_count, STACK_SIDE, STACK_SIDE), dtype=np.float32)


def _rule_planes_dict(record: dict[str, Any]) -> dict[str, Any]:
    planes = record.get("rule_planes_9x9")
    if isinstance(planes, dict):
        return planes
    if isinstance(planes, list):
        names = record.get("rule_channel_names", [])
        if len(names) != len(planes):
            raise ValueError("rule_planes_9x9 is a list, but rule_channel_names has a different length")
        return {str(name): plane for name, plane in zip(names, planes)}
    return {}


def _rule_payload(record: dict[str, Any], rule_name: str) -> Any:
    rule_instances = record.get("rule_instances")
    if isinstance(rule_instances, dict) and rule_name in rule_instances:
        return rule_instances[rule_name]
    return record.get(rule_name)


def _coerce_instances(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            return [x for x in payload if isinstance(x, dict)]
        # Support simple coordinate lists such as [[r, c], ...]
        if payload and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in payload):
            return [{"cells": payload}]
        return []
    if isinstance(payload, dict):
        if isinstance(payload.get("instances"), list):
            return [x for x in payload["instances"] if isinstance(x, dict)]
        if "cells" in payload:
            return [payload]
        if payload.get("active") is True:
            return [payload]
    return []


def _instance_cells(instance: dict[str, Any]) -> list[tuple[int, int]]:
    cells = instance.get("cells", [])
    out: list[tuple[int, int]] = []
    for cell in cells:
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            out.append((int(cell[0]), int(cell[1])))
    return out


def _instance_rule_value(instance: dict[str, Any]) -> float | None:
    for key in ("rule_value", "clue_sum", "sum", "value"):
        if key in instance:
            return float(instance[key])
    return None


def _instance_progress_values(instance: dict[str, Any], length: int) -> list[float]:
    cell_values = instance.get("cell_values")
    if isinstance(cell_values, list) and len(cell_values) == length:
        return [float(v) for v in cell_values]
    return [float((i + 1) / (length + 1)) for i in range(length)]


def _paint_constant(
    rule_channels: np.ndarray,
    channel_index: int,
    cells: list[tuple[int, int]],
    board_offset: tuple[int, int],
    value: float,
) -> np.ndarray:
    for cell in cells:
        rr, cc = _embed_coord(cell, board_offset)
        rule_channels[channel_index, rr, cc] = float(value)
    return rule_channels


def _paint_sequence(
    rule_channels: np.ndarray,
    channel_index: int,
    cells: list[tuple[int, int]],
    board_offset: tuple[int, int],
    values: list[float],
) -> np.ndarray:
    for cell, value in zip(cells, values):
        rr, cc = _embed_coord(cell, board_offset)
        rule_channels[channel_index, rr, cc] = float(value)
    return rule_channels


def _paint_global(
    rule_channels: np.ndarray,
    channel_index: int,
    side: int,
    board_offset: tuple[int, int],
    value: float = 1.0,
) -> np.ndarray:
    rr0, cc0 = board_offset
    rule_channels[channel_index, rr0:rr0 + side, cc0:cc0 + side] = float(value)
    return rule_channels


def _channel_mode(channel_name: str, num_rule_channels: int) -> str:
    name = channel_name.lower()
    if name.endswith("_mask") or "active" in name:
        return "active"
    if "progress" in name or "order" in name:
        return "progress"
    if "distance" in name or "segment_index" in name:
        return "sequence_value"
    if "anchor_row" in name:
        return "anchor_row"
    if "anchor_col" in name:
        return "anchor_col"
    if "clue_sum" in name or "rule_value" in name or name.endswith("_sum") or name.endswith("_value"):
        return "rule_value"
    if num_rule_channels == 1:
        return "active"
    return "unknown"


def _encode_rule_channels(
    record: dict[str, Any],
    schema: dict[str, Any],
    board_offset: tuple[int, int],
    side: int,
    rule_channel_count: int,
) -> np.ndarray:
    rule_channels = _empty_rule_channels(rule_channel_count)
    channel_map = _schema_channel_map(schema, rule_channel_count)
    direct_planes = _rule_planes_dict(record)
    filled_channels: set[str] = set()

    for channel_name, plane in direct_planes.items():
        if channel_name not in channel_map:
            continue
        plane_arr = np.asarray(plane, dtype=np.float32)
        if plane_arr.shape != (STACK_SIDE, STACK_SIDE):
            raise ValueError(
                f"Channel {channel_name!r} has shape {plane_arr.shape}, expected {(STACK_SIDE, STACK_SIDE)}"
            )
        rule_channels[channel_map[channel_name]] = plane_arr
        filled_channels.add(channel_name)

    schema_rules = schema.get("rules", {})
    active_rule_names = [
        _normalize_rule_name(str(name), schema_rules)
        for name in record.get("active_rule_names", [])
    ]

    for rule_name in active_rule_names:
        if rule_name not in schema_rules:
            continue

        rule_spec_channels = schema_rules[rule_name].get("channels", [])

        if rule_name == "standard_sudoku":
            for channel_spec in rule_spec_channels:
                channel_name = str(channel_spec["name"])
                if channel_name in filled_channels:
                    continue
                channel_index = int(channel_spec["index_hint"])
                rule_channels = _paint_global(rule_channels, channel_index, side, board_offset, 1.0)
                filled_channels.add(channel_name)
            continue

        payload = _rule_payload(record, rule_name)
        instances = _coerce_instances(payload)

        # Global active rules such as anti_knight / mean_baby_snake.
        if isinstance(payload, dict) and payload.get("active") is True and not instances:
            instances = [payload]

        if not instances:
            continue

        for channel_spec in rule_spec_channels:
            channel_name = str(channel_spec["name"])
            if channel_name in {"given_mask", "board_mask"}:
                continue
            if channel_name in filled_channels:
                continue

            channel_index = int(channel_spec["index_hint"])
            mode = _channel_mode(channel_name, len(rule_spec_channels))
            if mode == "unknown":
                continue

            if mode == "active" and any(inst.get("active") is True for inst in instances if isinstance(inst, dict)):
                rule_channels = _paint_global(rule_channels, channel_index, side, board_offset, 1.0)
                filled_channels.add(channel_name)
                continue

            for instance in instances:
                cells = _instance_cells(instance)
                if not cells:
                    continue

                if mode == "active":
                    rule_channels = _paint_constant(rule_channels, channel_index, cells, board_offset, 1.0)
                elif mode == "rule_value":
                    rule_value = _instance_rule_value(instance)
                    if rule_value is None:
                        continue
                    rule_channels = _paint_constant(rule_channels, channel_index, cells, board_offset, rule_value)
                elif mode == "progress":
                    values = _instance_progress_values(instance, len(cells))
                    rule_channels = _paint_sequence(rule_channels, channel_index, cells, board_offset, values)
                elif mode == "sequence_value":
                    values = _instance_progress_values(instance, len(cells))
                    rule_channels = _paint_sequence(rule_channels, channel_index, cells, board_offset, values)
                elif mode == "anchor_row":
                    anchor = instance.get("anchor")
                    if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                        rule_channels = _paint_constant(rule_channels, channel_index, cells, board_offset, (int(anchor[0]) + 1) / (side + 1))
                elif mode == "anchor_col":
                    anchor = instance.get("anchor")
                    if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                        rule_channels = _paint_constant(rule_channels, channel_index, cells, board_offset, (int(anchor[1]) + 1) / (side + 1))

            filled_channels.add(channel_name)

    return rule_channels


# ---------------------------------------------------------------------------
# Example creation
# ---------------------------------------------------------------------------

def _make_example_arrays(
    path: Path,
    record: dict[str, Any],
    schema: dict[str, Any],
    channel_names: list[str],
    rule_channel_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    side = int(record["side"])
    solution = _normalize_grid(record["solution"])
    givens = _normalize_grid(record["givens"])
    board_offset = _board_offset(record)

    if side + board_offset[0] > STACK_SIDE or side + board_offset[1] > STACK_SIDE:
        raise ValueError(
            f"Board with side={side} and offset={board_offset} does not fit into STACK_SIDE={STACK_SIDE}"
        )

    board_mask = _board_mask(side, board_offset)
    given_digits, given_mask = _encode_givens_digit_channels(givens, board_offset)
    rule_channels = _encode_rule_channels(record, schema, board_offset, side, rule_channel_count)
    hidden_channels = _empty_hidden_channels()
    solution_one_hot = _encode_solution_one_hot(solution, board_offset)
    target = _encode_target_indices(solution, board_offset)
    predict_mask = board_mask * (1.0 - given_mask)

    stack_chw = np.concatenate(
        [
            given_digits,
            given_mask[None, ...],
            board_mask[None, ...],
            rule_channels,
            hidden_channels,
            solution_one_hot,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    stack_hwc = np.transpose(stack_chw, (1, 2, 0))

    metadata = {
        "source_file": str(path),
        "variant_id": record.get("variant_id", path.stem),
        "active_rule_names": list(record.get("active_rule_names", [])),
        "board_offset": list(board_offset),
        "side": side,
        "channel_names": channel_names,
        "number_channels": NUMBER_CHANNELS,
        "given_mask_channels": GIVEN_MASK_CHANNELS,
        "board_mask_channels": BOARD_MASK_CHANNELS,
        "rule_channels": rule_channel_count,
        "hidden_channels": HIDDEN_CHANNELS,
        "target_channels": TARGET_CHANNELS,
        "stack_side": STACK_SIDE,
        "predict_mask": predict_mask.tolist(),
    }
    return stack_hwc, target, metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_puzzle_stack(
    dataset_dir: str | Path = DATASET_INPUT_DIR,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = _default_dataset_dir(dataset_dir)
    schema = _load_schema(dataset_root, schema_path)
    rule_channel_count = _schema_rule_channel_count(schema)
    channel_map = _schema_channel_map(schema, rule_channel_count)
    rule_channel_names = _schema_rule_channel_names(schema, rule_channel_count)
    channel_names = _all_channel_names(schema, rule_channel_count)

    variant_files = _discover_variant_files(dataset_root, schema_path)
    if not variant_files:
        raise ValueError(f"Fant ingen puzzle-JSON-filer i {dataset_root!r}")

    stacks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for path in variant_files:
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        stack_hwc, target, meta = _make_example_arrays(path, record, schema, channel_names, rule_channel_count)
        stacks.append(stack_hwc)
        targets.append(target)
        metadata.append(meta)

    return {
        "stacks": np.stack(stacks, axis=0),
        "targets": np.stack(targets, axis=0),
        "metadata": metadata,
        "channel_names": channel_names,
        "rule_channel_names": rule_channel_names,
        "rule_channel_map": channel_map,
        "number_channels": NUMBER_CHANNELS,
        "given_mask_channels": GIVEN_MASK_CHANNELS,
        "board_mask_channels": BOARD_MASK_CHANNELS,
        "rule_channels": rule_channel_count,
        "hidden_channels": HIDDEN_CHANNELS,
        "target_channels": TARGET_CHANNELS,
        "total_channels": len(channel_names),
        "stack_side": STACK_SIDE,
        "dataset_dir": str(dataset_root.resolve()),
        "schema_path": str(_schema_path(dataset_root, schema_path).resolve()),
    }


def build_and_save_stack_dataset(
    output_dir: str | Path,
    dataset_dir: str | Path = DATASET_INPUT_DIR,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    test_output_dir: str | Path | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    schema_path: str | Path | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
    create_archive: bool = False,
    archive_path: str | Path | None = None,
    keep_uncompressed: bool = True,
) -> dict[str, Any]:
    if validation_count < 0:
        raise ValueError("validation_count må være >= 0.")
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError("test_fraction må være >= 0.0 og < 1.0.")
    if log_every <= 0:
        raise ValueError("log_every må være >= 1.")

    dataset_root = _default_dataset_dir(dataset_dir)
    schema = _load_schema(dataset_root, schema_path)
    rule_channel_count = _schema_rule_channel_count(schema)
    channel_map = _schema_channel_map(schema, rule_channel_count)
    rule_channel_names = _schema_rule_channel_names(schema, rule_channel_count)
    channel_names = _all_channel_names(schema, rule_channel_count)

    variant_files = _discover_variant_files(dataset_root, schema_path)
    if not variant_files:
        raise ValueError(f"Fant ingen puzzle-JSON-filer i {dataset_root!r}")

    rng = random.Random(seed)
    rng.shuffle(variant_files)

    n_total = len(variant_files)
    n_test = min(int(round(n_total * float(test_fraction))), n_total)
    test_files = variant_files[:n_test]
    remaining_files = variant_files[n_test:]

    n_val = min(int(validation_count), len(remaining_files))
    val_files = remaining_files[:n_val]
    train_files = remaining_files[n_val:]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    test_output_root = (
        output_root.parent / "stack_test"
        if test_output_dir is None
        else Path(test_output_dir)
    )
    test_output_root.mkdir(parents=True, exist_ok=True)

    total_channels = len(channel_names)
    stack_shape_train = (len(train_files), STACK_SIDE, STACK_SIDE, total_channels)
    stack_shape_val = (len(val_files), STACK_SIDE, STACK_SIDE, total_channels)
    stack_shape_test = (len(test_files), STACK_SIDE, STACK_SIDE, total_channels)
    target_shape_train = (len(train_files), STACK_SIDE, STACK_SIDE)
    target_shape_val = (len(val_files), STACK_SIDE, STACK_SIDE)
    target_shape_test = (len(test_files), STACK_SIDE, STACK_SIDE)

    manifest = {
        "schema_name": schema.get("schema_name"),
        "schema_version": schema.get("version"),
        "dataset_dir": str(dataset_root.resolve()),
        "schema_path": str(_schema_path(dataset_root, schema_path).resolve()),
        "seed": int(seed),
        "validation_count": int(n_val),
        "requested_validation_count": int(validation_count),
        "test_fraction": float(test_fraction),
        "total_examples_before_split": n_total,
        "test_examples": len(test_files),
        "remaining_after_test_examples": len(remaining_files),
        "train_examples": len(train_files),
        "validation_examples": len(val_files),
        "test_output_dir": str(test_output_root.resolve()),
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
        "train_stacks_file": "train_stacks.npy",
        "validation_stacks_file": "validation_stacks.npy",
        "train_targets_file": "train_targets.npy",
        "validation_targets_file": "validation_targets.npy",
        "train_metadata_file": "train_metadata.json",
        "validation_metadata_file": "validation_metadata.json",
    }

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    test_manifest = {
        "schema_name": schema.get("schema_name"),
        "schema_version": schema.get("version"),
        "dataset_dir": str(dataset_root.resolve()),
        "schema_path": str(_schema_path(dataset_root, schema_path).resolve()),
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "total_examples_before_split": n_total,
        "test_examples": len(test_files),
        "excluded_from_training_dataset": True,
        "train_validation_output_dir": str(output_root.resolve()),
        "output_dir": str(test_output_root.resolve()),
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
        "test_stacks_file": "test_stacks.npy",
        "test_targets_file": "test_targets.npy",
        "test_metadata_file": "test_metadata.json",
    }
    with (test_output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2, ensure_ascii=False)

    def _save_split(
        split_name: str,
        split_files: list[Path],
        stack_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        stack_path: Path,
        target_path: Path,
        metadata_path: Path,
    ) -> None:
        total = len(split_files)
        if total == 0:
            print(f"[{split_name}] Ingen filer å lagre.")
            np.save(stack_path, np.empty(stack_shape, dtype=np.float32))
            np.save(target_path, np.empty(target_shape, dtype=np.int32))
            metadata_path.write_text("[]", encoding="utf-8")
            return

        stacks_mm = np.lib.format.open_memmap(stack_path, mode="w+", dtype=np.float32, shape=stack_shape)
        targets_mm = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.int32, shape=target_shape)
        metadata_list: list[dict[str, Any]] = []

        t0 = time.time()
        for i, path in enumerate(split_files, start=1):
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
            stack_hwc, target, meta = _make_example_arrays(path, record, schema, channel_names, rule_channel_count)
            stacks_mm[i - 1] = stack_hwc
            targets_mm[i - 1] = target
            metadata_list.append(meta)

            if i == 1 or i == total or i % log_every == 0:
                elapsed = time.time() - t0
                avg = elapsed / i
                remaining = avg * (total - i)
                print(
                    f"[{split_name}] {i}/{total} lagret | "
                    f"elapsed={_format_seconds(elapsed)} | "
                    f"eta={_format_seconds(remaining)} | "
                    f"siste={path.name}"
                )

        stacks_mm.flush()
        targets_mm.flush()
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"Leser dataset fra: {dataset_root.resolve()}")
    print(f"Schema: {_schema_path(dataset_root, schema_path).resolve()}")
    print(f"Lagrer train/validation-dataset til: {output_root.resolve()}")
    print(f"Lagrer test-dataset til: {test_output_root.resolve()}")
    print(f"Totalt antall variantfiler: {n_total}")
    print(
        f"Test: {len(test_files)} ({float(test_fraction):.1%}) | "
        f"Validation: {len(val_files)} | Train: {len(train_files)}"
    )
    print("Format: .npy (train_stacks.npy / validation_stacks.npy / test_stacks.npy)")

    _save_split(
        "test",
        test_files,
        stack_shape_test,
        target_shape_test,
        test_output_root / "test_stacks.npy",
        test_output_root / "test_targets.npy",
        test_output_root / "test_metadata.json",
    )
    _save_split(
        "validation",
        val_files,
        stack_shape_val,
        target_shape_val,
        output_root / "validation_stacks.npy",
        output_root / "validation_targets.npy",
        output_root / "validation_metadata.json",
    )
    _save_split(
        "train",
        train_files,
        stack_shape_train,
        target_shape_train,
        output_root / "train_stacks.npy",
        output_root / "train_targets.npy",
        output_root / "train_metadata.json",
    )

    archive_file: Path | None = None
    if create_archive:
        archive_file = compress_dataset_directory(
            output_root,
            archive_path=archive_path,
            compression="gz",
            overwrite=True,
            log=True,
        )
        manifest["archive_file"] = str(archive_file.resolve())
        with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        if not keep_uncompressed:
            print(f"[compress] Fjerner utpakket dataset etter vellykket komprimering: {output_root}")
            for path in sorted(output_root.rglob('*'), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if output_root.exists():
                output_root.rmdir()

    print(f"Ferdig. Train/validation-dataset lagret i: {output_root.resolve()}")
    print(f"Ferdig. Test-dataset lagret i: {test_output_root.resolve()}")
    if archive_file is not None:
        print(f"Arkiv lagret i: {archive_file.resolve()}")
    return manifest


if __name__ == "__main__":
    output_root = Path(__file__).resolve().parent / "stack_dataset"
    manifest = build_and_save_stack_dataset(
        output_dir=output_root,
        dataset_dir=DATASET_INPUT_DIR,
        validation_count=DEFAULT_VALIDATION_COUNT,
        seed=DEFAULT_SPLIT_SEED,
        log_every=DEFAULT_LOG_EVERY,
        create_archive=True,
        keep_uncompressed=True,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))