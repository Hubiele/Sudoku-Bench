import jax
import jax.numpy as jnp
import equinox as eqx
from Train.optimizer import optimizer
from NCA.NCA_model import (
    NUMBER_CHANNELS,
    TARGET_CHANNELS,
    GIVEN_MASK_START,
    GIVEN_MASK_END,
    BOARD_MASK_START,
    BOARD_MASK_END,
)


def _masked_cross_entropy(full_stack):
    logits = full_stack[..., :NUMBER_CHANNELS]
    target_onehot = full_stack[..., -TARGET_CHANNELS:]
    given_mask = full_stack[..., GIVEN_MASK_START:GIVEN_MASK_END]
    board_mask = full_stack[..., BOARD_MASK_START:BOARD_MASK_END]
    predict_mask = board_mask * (1.0 - given_mask)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce_per_cell = -jnp.sum(target_onehot * log_probs, axis=-1, keepdims=True)
    masked = ce_per_cell * predict_mask
    denom = jnp.maximum(jnp.sum(predict_mask), 1.0)
    return jnp.sum(masked) / denom


def split_trainable(model):
    spec = jax.tree_util.tree_map(eqx.is_array, model)
    spec = eqx.tree_at(
        lambda m: (m.filter_bank, m.update_channels, m.perceive),
        spec,
        replace=(False, False, False),
    )
    return eqx.partition(model, spec)


def make_train_step(train_nca_steps: int, eval_nca_steps: int | None = None):
    if eval_nca_steps is None:
        eval_nca_steps = train_nca_steps

    def _l2_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.array(0.0, dtype=jnp.float32)
        sq = sum([jnp.sum(x * x) for x in leaves])
        return jnp.sqrt(sq)

    def _norm_of(maybe_tree):
        if maybe_tree is None:
            return jnp.array(0.0, dtype=jnp.float32)
        return _l2_norm(eqx.filter(maybe_tree, eqx.is_array))

    def _rollout(m, rollout_key, pool0, steps: int):
        B = pool0.shape[0]

        def one_step(pool_t, step_key):
            step_keys = jax.random.split(step_key, B)
            pool_next = jax.vmap(m)(step_keys, pool_t)
            return pool_next, None

        step_keys = jax.random.split(rollout_key, steps)
        poolT, _ = jax.lax.scan(one_step, pool0, step_keys)
        return poolT

    @eqx.filter_jit
    def train_step(model, opt_state, key, pool):
        trainable_model, frozen_model = split_trainable(model)

        def loss_fn(trainable_part):
            full_model = eqx.combine(trainable_part, frozen_model)
            poolT = _rollout(full_model, key, pool, train_nca_steps)
            loss_value = _masked_cross_entropy(poolT)
            return loss_value, poolT

        (loss_value, poolT), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable_model)

        grad_norm = _l2_norm(eqx.filter(grads, eqx.is_array))
        mlp_grad_norm = _norm_of(getattr(grads, "network", None))
        rule_emb_grad_norm = _norm_of(getattr(grads, "rule_embeddings", None))
        size_emb_grad_norm = _norm_of(getattr(grads, "grid_size_embeddings", None))

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params=trainable_model,
        )
        update_norm = _l2_norm(eqx.filter(updates, eqx.is_array))

        trainable_model = eqx.apply_updates(trainable_model, updates)
        model = eqx.combine(trainable_model, frozen_model)

        return (
            model,
            opt_state,
            poolT,
            loss_value,
            grad_norm,
            mlp_grad_norm,
            rule_emb_grad_norm,
            size_emb_grad_norm,
            update_norm,
        )

    @eqx.filter_jit
    def eval_step(model, key, pool):
        poolT = _rollout(model, key, pool, eval_nca_steps)
        loss_value = _masked_cross_entropy(poolT)
        return loss_value

    return train_step, eval_step