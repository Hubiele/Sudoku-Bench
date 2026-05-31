import optax

# --- Configuration ---
TOTAL_STEPS = 100000
WARMUP_STEPS = 5000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# Learning Rate Schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=5e-5,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=TOTAL_STEPS,
    end_value=1e-5
)

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),

    optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
)