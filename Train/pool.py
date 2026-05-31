import json
import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from NCA.NCA_model import TOTAL_STACK_CHANNELS

# --- Configuration ---
pool_size = 256
pool_update_fraction = 0.25
pool_update_number = int(round(pool_size * pool_update_fraction))
DEFAULT_DATASET_DIR = Path("StackGenerator") / "stack_dataset"
LEGACY_DATASET_DIR = Path("stack_dataset")

# Module-level RAM cache (NumPy -> stays in host RAM)
_TRAIN_CACHE: Optional[np.ndarray] = None  # (N,H,W,C)
_VAL_CACHE: Optional[np.ndarray] = None  # (N,H,W,C)


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


def _load_split_to_ram(split: str) -> np.ndarray:
    split_file = _split_file(split)
    print(f"[pool] Looking for dataset split in: {split_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Could not find dataset split file: {split_file}")

    stacks = np.load(split_file)
    if stacks.ndim != 4:
        raise ValueError(f"Expected 4D stack array in {split_file}, got shape {stacks.shape}")
    if stacks.shape[-1] != TOTAL_STACK_CHANNELS:
        raise ValueError(
            f"Unexpected channel count {stacks.shape[-1]} in {split_file}. "
            f"Expected TOTAL_STACK_CHANNELS={TOTAL_STACK_CHANNELS}."
        )

    print(f"[pool] Loaded {stacks.shape[0]} {split} stacks from: {split_file}")
    return stacks.astype(np.float32, copy=False)


def _cache_for_split(split: str) -> np.ndarray:
    global _TRAIN_CACHE, _VAL_CACHE
    if split == "train":
        if _TRAIN_CACHE is None:
            _TRAIN_CACHE = _load_split_to_ram("train")
        return _TRAIN_CACHE
    if split == "validation":
        if _VAL_CACHE is None:
            _VAL_CACHE = _load_split_to_ram("validation")
        return _VAL_CACHE
    raise ValueError(f"Unknown split: {split}")


def preload_dataset() -> None:
    """Optional helper to force eager RAM-loading before training starts."""
    _cache_for_split("train")
    try:
        _cache_for_split("validation")
    except Exception:
        pass


def _sample_from_cache(key, cache: np.ndarray, batch_size: int) -> jnp.ndarray:
    n = cache.shape[0]
    idx = np.asarray(jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n))
    sampled = cache[idx]  # still NumPy in RAM
    return jnp.asarray(sampled)  # moved to device only for the sampled batch


def create_pool(key, split: str = "train"):
    cache = _cache_for_split(split)
    return _sample_from_cache(key, cache, pool_size)


def update_pool(key, pool, split: str = "train"):
    cache = _cache_for_split(split)
    n_cache = cache.shape[0]

    replace_indices = np.asarray(
        jax.random.randint(key, shape=(pool_update_number,), minval=0, maxval=pool.shape[0])
    )
    key, sample_key = jax.random.split(key)
    sample_indices = np.asarray(
        jax.random.randint(sample_key, shape=(pool_update_number,), minval=0, maxval=n_cache)
    )

    replacements = jnp.asarray(cache[sample_indices])
    pool = pool.at[replace_indices].set(replacements)
    return pool


def get_validation_dataset() -> jnp.ndarray:
    """Returns the full validation split on demand as a device array."""
    return jnp.asarray(_cache_for_split("validation"))