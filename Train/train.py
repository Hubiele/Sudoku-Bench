import json
import time
from pathlib import Path

import equinox as eqx
import jax

from Train.optimizer import optimizer
from Train import pool as pool_mod
from Train.train_step import make_train_step, split_trainable
from NCA.NCA_model import NCA
from Train.checkpoints import save_checkpoint
from StackGenerator.puzzles_train import ensure_dataset_ready

# --- Config ---
MAX_NCA_STEPS = 32
POOL_REFRESH_EVERY = 1
VALIDATION_NCA_STEPS = int((MAX_NCA_STEPS / pool_mod.pool_update_fraction) * POOL_REFRESH_EVERY)
ITERATIONS = 100000
VALIDATE_EVERY = 100

# Checkpoint config
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 1000

# Dataset config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "StackGenerator" / "stack_dataset"
DATASET_ARCHIVE = PROJECT_ROOT / "StackGenerator" / "stack_dataset.tar.gz"

# --- Init ---
key = jax.random.PRNGKey(0)

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
        config = {
            "MAX_NCA_STEPS": int(MAX_NCA_STEPS),
            "VALIDATION_NCA_STEPS": int(VALIDATION_NCA_STEPS),
            "ITERATIONS": int(ITERATIONS),
            "POOL_REFRESH_EVERY": int(POOL_REFRESH_EVERY),
            "VALIDATE_EVERY": int(VALIDATE_EVERY),
            "POOL_SIZE": int(pool_mod.pool_size),
            "VALIDATION_SIZE": int(validation_pool.shape[0]),
        }

        ckpt_path = CKPT_DIR / f"step_{it + 1:06d}"
        save_checkpoint(
            ckpt_path,
            step=it + 1,
            model=model,
            opt_state=opt_state,
            pool_state=pool_state,
            key=key,
            config=config,
        )

print(json.dumps({
    "status": "finished",
    "iterations": ITERATIONS,
}, indent=2))