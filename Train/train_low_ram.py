import json
from pathlib import Path

import equinox as eqx
import jax

from Train.optimizer import optimizer
from Train import pool_low_ram as pool_mod
from Train.train_step import make_train_step, split_trainable
from NCA.NCA_model import NCA
from Train.checkpoints import save_checkpoint
from StackGenerator.puzzles_train import ensure_dataset_ready

print(jax.default_backend())

print(jax.devices())

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

# =============================================================================
# DATASET INITIALIZATION
# =============================================================================
# The old version called:
#
#     pool_mod.preload_dataset()
#
# and pool.py loaded the full training split into host RAM.
#
# In the updated pool.py, TRAIN_SPLIT_MODE controls the behavior:
#
#     TRAIN_SPLIT_MODE = "memmap"  -> low-RAM mode for training data
#     TRAIN_SPLIT_MODE = "ram"     -> old behavior, full training split in RAM
#
# Here we only initialize the training dataset handle. If TRAIN_SPLIT_MODE is
# "memmap", this opens train_stacks.npy with mmap_mode='r' and does not load the
# full training dataset into RAM.
#
# Validation is intentionally kept unchanged below: get_validation_dataset()
# still returns the full validation split as a fixed validation pool.
# =============================================================================
print("[dataset] Initializing dataset handles")
pool_mod.initialize_dataset_handles(load_validation=False)

# =============================================================================
# OPTIONAL RAM PRELOAD BLOCK
# =============================================================================
# Enable this only if you want to force eager loading according to the modes in
# pool.py. If TRAIN_SPLIT_MODE='ram', this loads the full training dataset into
# RAM. Leave it commented out for low-RAM training.
#
# pool_mod.preload_dataset()
# =============================================================================

key, subkey = jax.random.split(key)
pool_state = pool_mod.create_pool(subkey, split="train")  # (B,H,W,C)

# Validation split is kept fixed and unchanged.
# The validation dataset is small, so it is still loaded as one full validation pool.
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
            "train", float(train_loss_value),
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
            "TRAIN_SPLIT_MODE": str(pool_mod.TRAIN_SPLIT_MODE),
            "VALIDATION_SPLIT_MODE": str(pool_mod.VALIDATION_SPLIT_MODE),
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
    "TRAIN_SPLIT_MODE": str(pool_mod.TRAIN_SPLIT_MODE),
    "VALIDATION_SPLIT_MODE": str(pool_mod.VALIDATION_SPLIT_MODE),
}, indent=2))
