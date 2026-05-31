import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from NCA.NCA_model import TOTAL_STACK_CHANNELS

# =============================================================================
# Configuration
# =============================================================================
pool_size = 256
pool_update_fraction = 0.25
pool_update_number = int(round(pool_size * pool_update_fraction))

DEFAULT_DATASET_DIR = Path("StackGenerator") / "stack_dataset"
LEGACY_DATASET_DIR = Path("stack_dataset")

# =============================================================================
# Dataset loading mode
# =============================================================================
# TRAIN_SPLIT_MODE controls only the training split.
#
# "memmap":
#   Low-RAM mode. Opens train_stacks.npy with np.load(..., mmap_mode="r").
#   The full training dataset is not loaded into host RAM.
#   Only sampled rows are copied into RAM/device memory.
#
# "ram":
#   Old/preload mode. Loads the full train_stacks.npy into host RAM.
#   This is faster, but requires enough memory for the full training split.
# =============================================================================
TRAIN_SPLIT_MODE = "memmap"  # "memmap" or "ram"

# Validation is intentionally kept as before.
# The validation split is small and is loaded as a full array when
# get_validation_dataset() is called.
VALIDATION_SPLIT_MODE = "ram"

# =============================================================================
# Memmap sampling optimization
# =============================================================================
# When data is a memmap, random indexing can be slow because it reads from many
# random locations in a large .npy file. Sorting the sampled indices before
# reading makes the disk access more sequential. The sampled batch is then
# restored to the original random order before being returned.
#
# This mostly matters for TRAIN_SPLIT_MODE="memmap".
# =============================================================================
SORT_INDICES_BEFORE_MEMMAP_READ = True

# Module-level dataset handles.
# In "ram" mode these are normal NumPy arrays.
# In "memmap" mode these are NumPy memmap arrays backed by the .npy file on disk.
_TRAIN_DATA: Optional[np.ndarray] = None
_VAL_DATA: Optional[np.ndarray] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_root() -> Path:
    env_path = os.environ.get("STACK_DATASET_DIR")
    if env_path:
        return Path(env_path)

    root = _project_root()
    preferred = root / DEFAULT_DATASET_DIR
    if preferred.exists():
        return preferred

    legacy = root / LEGACY_DATASET_DIR
    if legacy.exists():
        return legacy

    return preferred


def _split_file(split: str) -> Path:
    return _dataset_root() / f"{split}_stacks.npy"


def _mode_for_split(split: str) -> str:
    if split == "train":
        return TRAIN_SPLIT_MODE
    if split == "validation":
        return VALIDATION_SPLIT_MODE
    raise ValueError(f"Unknown split: {split}")


def _is_memmap_array(data: np.ndarray) -> bool:
    return isinstance(data, np.memmap)


def _load_split(split: str) -> np.ndarray:
    split_file = _split_file(split)
    mode = _mode_for_split(split)

    print(f"[pool] Looking for dataset split in: {split_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Could not find dataset split file: {split_file}")

    if mode == "ram":
        # =========================================================================
        # RAM MODE
        # Loads the full split into host RAM.
        # =========================================================================
        print(f"[pool] Loading {split} split fully into host RAM")
        stacks = np.load(split_file)
        stacks = stacks.astype(np.float32, copy=False)

    elif mode == "memmap":
        # =========================================================================
        # LOW-RAM / MEMMAP MODE
        # Opens the .npy file without loading the full array into RAM.
        # Only indexed samples are read later in create_pool/update_pool.
        # =========================================================================
        print(f"[pool] Opening {split} split with mmap_mode='r'")
        stacks = np.load(split_file, mmap_mode="r")

    else:
        raise ValueError(
            f"Invalid mode for {split}: {mode!r}. "
            "Expected 'ram' or 'memmap'."
        )

    if stacks.ndim != 4:
        raise ValueError(f"Expected 4D stack array in {split_file}, got shape {stacks.shape}")
    if stacks.shape[-1] != TOTAL_STACK_CHANNELS:
        raise ValueError(
            f"Unexpected channel count {stacks.shape[-1]} in {split_file}. "
            f"Expected TOTAL_STACK_CHANNELS={TOTAL_STACK_CHANNELS}."
        )

    print(
        f"[pool] Ready: {stacks.shape[0]} {split} stacks "
        f"| shape={stacks.shape} | mode={mode}"
    )
    return stacks


def _data_for_split(split: str) -> np.ndarray:
    global _TRAIN_DATA, _VAL_DATA

    if split == "train":
        if _TRAIN_DATA is None:
            _TRAIN_DATA = _load_split("train")
        return _TRAIN_DATA

    if split == "validation":
        if _VAL_DATA is None:
            _VAL_DATA = _load_split("validation")
        return _VAL_DATA

    raise ValueError(f"Unknown split: {split}")


def initialize_dataset_handles(load_validation: bool = False) -> None:
    """
    Initializes dataset handles without necessarily loading the full training
    dataset into RAM.

    In low-RAM mode, this only opens train_stacks.npy as a memmap.
    If load_validation=True, validation is also loaded according to
    VALIDATION_SPLIT_MODE.
    """
    _data_for_split("train")
    if load_validation:
        _data_for_split("validation")


def preload_dataset() -> None:
    """
    Backward-compatible helper.

    In the old code this forced eager RAM-loading. In this version, the behavior
    depends on TRAIN_SPLIT_MODE and VALIDATION_SPLIT_MODE.

    With TRAIN_SPLIT_MODE='memmap', the training split is only opened as a
    memory-mapped file, not loaded into RAM.
    """
    _data_for_split("train")
    try:
        _data_for_split("validation")
    except Exception:
        pass


def _read_rows(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Read rows from either a normal ndarray or a memmap.

    For memmap arrays, optionally sort indices before reading to reduce random
    disk access. The result is restored to the original random order.

    This function returns a normal in-memory NumPy array with dtype float32.
    Only the selected rows are materialized.
    """
    indices = np.asarray(indices, dtype=np.int64)

    if SORT_INDICES_BEFORE_MEMMAP_READ and _is_memmap_array(data) and len(indices) > 1:
        # Sort indices so the memmap read is closer to sequential disk access.
        order = np.argsort(indices)
        sorted_indices = indices[order]

        sampled_sorted = np.asarray(data[sorted_indices], dtype=np.float32)

        # Restore original random order, so the training batch is unchanged
        # statistically and positionally.
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        sampled = sampled_sorted[inverse_order]
        return sampled

    # Normal RAM mode, validation mode, or sorting disabled.
    return np.asarray(data[indices], dtype=np.float32)


def _sample_indices(key, n: int, batch_size: int) -> np.ndarray:
    return np.asarray(
        jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n),
        dtype=np.int64,
    )


def _sample_from_data(key, data: np.ndarray, batch_size: int) -> jnp.ndarray:
    n = data.shape[0]
    idx = _sample_indices(key, n, batch_size)

    # Important:
    # - If data is a normal ndarray, this samples from RAM.
    # - If data is a memmap, _read_rows can sort the indices before reading.
    sampled = _read_rows(data, idx)

    # Move only the sampled batch to device.
    return jnp.asarray(sampled)


def create_pool(key, split: str = "train"):
    data = _data_for_split(split)
    return _sample_from_data(key, data, pool_size)


def update_pool(key, pool, split: str = "train"):
    data = _data_for_split(split)
    n_data = data.shape[0]

    replace_indices = np.asarray(
        jax.random.randint(key, shape=(pool_update_number,), minval=0, maxval=pool.shape[0])
    )

    key, sample_key = jax.random.split(key)
    sample_indices = _sample_indices(sample_key, n_data, pool_update_number)

    # In low-RAM mode, this reads only pool_update_number rows from disk.
    # The rows are sorted before reading if SORT_INDICES_BEFORE_MEMMAP_READ=True.
    replacements = jnp.asarray(_read_rows(data, sample_indices))

    pool = pool.at[replace_indices].set(replacements)
    return pool


def get_validation_dataset() -> jnp.ndarray:
    """
    Returns the full validation split as a device array.

    This is intentionally kept unchanged in behavior. Since the validation split
    is small, it is still loaded as one fixed validation pool.
    """
    return jnp.asarray(np.asarray(_data_for_split("validation"), dtype=np.float32))