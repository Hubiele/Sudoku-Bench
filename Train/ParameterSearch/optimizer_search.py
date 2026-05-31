import os

import optax


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --- Configuration from environment variables ---
# These defaults match the current single-run setup.
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "warmup_cosine")

TOTAL_STEPS = env_int("TOTAL_STEPS", 100000)
WARMUP_STEPS = env_int("WARMUP_STEPS", 10000)

LEARNING_RATE = env_float("LEARNING_RATE", 3e-4)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 5e-3)

INIT_VALUE = env_float("INIT_VALUE", 1e-4)
END_VALUE = env_float("END_VALUE", 1e-5)
CLIP_NORM = env_float("CLIP_NORM", 1.0)


def make_schedule():
    if LR_SCHEDULE == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=INIT_VALUE,
            peak_value=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            decay_steps=TOTAL_STEPS,
            end_value=END_VALUE,
        )

    if LR_SCHEDULE == "constant":
        return optax.constant_schedule(LEARNING_RATE)

    if LR_SCHEDULE == "warmup_linear":
        warmup_schedule = optax.linear_schedule(
            init_value=INIT_VALUE,
            end_value=LEARNING_RATE,
            transition_steps=max(WARMUP_STEPS, 1),
        )

        decay_schedule = optax.linear_schedule(
            init_value=LEARNING_RATE,
            end_value=END_VALUE,
            transition_steps=max(TOTAL_STEPS - WARMUP_STEPS, 1),
        )

        return optax.join_schedules(
            schedules=[warmup_schedule, decay_schedule],
            boundaries=[WARMUP_STEPS],
        )

    raise ValueError(
        f"Unknown LR_SCHEDULE={LR_SCHEDULE!r}. "
        "Supported values: warmup_cosine, warmup_linear, constant."
    )


schedule = make_schedule()

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(CLIP_NORM),
    optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
)


OPTIMIZER_CONFIG = {
    "LR_SCHEDULE": LR_SCHEDULE,
    "TOTAL_STEPS": int(TOTAL_STEPS),
    "WARMUP_STEPS": int(WARMUP_STEPS),
    "LEARNING_RATE": float(LEARNING_RATE),
    "WEIGHT_DECAY": float(WEIGHT_DECAY),
    "INIT_VALUE": float(INIT_VALUE),
    "END_VALUE": float(END_VALUE),
    "CLIP_NORM": float(CLIP_NORM),
}
