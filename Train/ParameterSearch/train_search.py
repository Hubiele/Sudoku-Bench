import json
import os
import time
from pathlib import Path

import jax

from Train.ParameterSearch.optimizer_search import optimizer, OPTIMIZER_CONFIG
from Train import pool as pool_mod
from Train.train_step import make_train_step, split_trainable
from NCA.NCA_model import NCA
from Train.checkpoints import save_checkpoint
from StackGenerator.puzzles_train import ensure_dataset_ready


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


# --- Config ---
RUN_NAME = os.environ.get("RUN_NAME", "parameter_search_default")
SEED = env_int("SEED", 0)

MAX_NCA_STEPS = env_int("MAX_NCA_STEPS", 32)
POOL_REFRESH_EVERY = env_int("POOL_REFRESH_EVERY", 1)
VALIDATION_NCA_STEPS = int((MAX_NCA_STEPS / pool_mod.pool_update_fraction) * POOL_REFRESH_EVERY)

ITERATIONS = env_int("ITERATIONS", OPTIMIZER_CONFIG["TOTAL_STEPS"])
VALIDATE_EVERY = env_int("VALIDATE_EVERY", 100)

# Checkpoint config
DEFAULT_CKPT_DIR = Path("runs") / "parameter_search" / "checkpoints" / RUN_NAME
CKPT_DIR = env_path("CKPT_DIR", DEFAULT_CKPT_DIR)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = env_int("SAVE_EVERY", 1000)

# Dataset config
# This file is located in Train/ParameterSearch, so parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "StackGenerator" / "stack_dataset"
DATASET_ARCHIVE = PROJECT_ROOT / "StackGenerator" / "stack_dataset.tar.gz"


def build_run_config(validation_size=None):
    config = {
        "RUN_NAME": RUN_NAME,
        "SEED": int(SEED),
        "CKPT_DIR": str(CKPT_DIR),
        "MAX_NCA_STEPS": int(MAX_NCA_STEPS),
        "VALIDATION_NCA_STEPS": int(VALIDATION_NCA_STEPS),
        "ITERATIONS": int(ITERATIONS),
        "POOL_REFRESH_EVERY": int(POOL_REFRESH_EVERY),
        "VALIDATE_EVERY": int(VALIDATE_EVERY),
        "SAVE_EVERY": int(SAVE_EVERY),
        "POOL_SIZE": int(pool_mod.pool_size),
        "POOL_UPDATE_FRACTION": float(pool_mod.pool_update_fraction),
        "OPTIMIZER": OPTIMIZER_CONFIG,
    }

    if validation_size is not None:
        config["VALIDATION_SIZE"] = int(validation_size)

    return config


print(json.dumps({
    "status": "starting",
    "config": build_run_config(),
}, indent=2))

# Store the run config before training starts.
with open(CKPT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(build_run_config(), f, indent=2)

# --- Init ---
key = jax.random.PRNGKey(SEED)

# Ensure dataset is available locally in extracted form before pool.py reads it.
ensure_dataset_ready(DATASET_DIR, DATASET_ARCHIVE, log=True)

# Load training/validation dataset into host RAM once.
print(f"[preload] Starter lasting av dataset til host RAM fra {DATASET_DIR}")
preload_t0 = time.time()
pool_mod.preload_dataset()
print(f"[preload] Ferdig | elapsed={time.time() - preload_t0:.2f}s")

key, subkey = jax.random.split(key)
pool_state = pool_mod.create_pool(subkey, split="train")  # (B,H,W,C)

# Validation split is kept fixed and also sourced from RAM cache.
validation_pool = pool_mod.get_validation_dataset()

# Update config after validation size is known.
with open(CKPT_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(build_run_config(validation_pool.shape[0]), f, indent=2)

key, subkey = jax.random.split(key)
model = NCA(subkey, pool_state[0])
trainable_model, _ = split_trainable(model)
opt_state = optimizer.init(trainable_model)
train_step, eval_step = make_train_step(MAX_NCA_STEPS, VALIDATION_NCA_STEPS)

# --- Train loop ---
for it in range(ITERATIONS):
    key, subkey = jax.random.split(key)
    (
        model,
        opt_state,
        pool_state,
        train_loss_value,
        grad_norm,
        mlp_gn,
        rule_gn,
        size_gn,
        update_norm,
    ) = train_step(model, opt_state, subkey, pool_state)

    if (it + 1) % POOL_REFRESH_EVERY == 0:
        key, subkey = jax.random.split(key)
        pool_state = pool_mod.update_pool(subkey, pool_state, split="train")

    if (it + 1) % VALIDATE_EVERY == 0:
        key, subkey = jax.random.split(key)
        val_loss_value = eval_step(model, subkey, validation_pool)
        print(
            it + 1,
            float(val_loss_value),
            "gn", float(grad_norm),
            "upd", float(update_norm),
            "mlp", float(mlp_gn),
            "rule", float(rule_gn),
            "size", float(size_gn),
        )

    if (it + 1) % SAVE_EVERY == 0:
        ckpt_path = CKPT_DIR / f"step_{it + 1:06d}"
        save_checkpoint(
            ckpt_path,
            step=it + 1,
            model=model,
            opt_state=opt_state,
            pool_state=pool_state,
            key=key,
            config=build_run_config(validation_pool.shape[0]),
        )

print(json.dumps({
    "status": "finished",
    "run_name": RUN_NAME,
    "iterations": ITERATIONS,
    "checkpoint_dir": str(CKPT_DIR),
}, indent=2))
