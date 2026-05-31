import os, json
from pathlib import Path
import equinox as eqx
import jax


def save_checkpoint(path: Path, *, step: int, model, opt_state, pool_state, key, config: dict):
    path.mkdir(parents=True, exist_ok=True)

    # 1) The model
    eqx.tree_serialise_leaves(path / "model.eqx", model)

    # 2) Trainer-state (opt_state/pool/key)
    eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)
    eqx.tree_serialise_leaves(path / "pool_state.eqx", pool_state)
    eqx.tree_serialise_leaves(path / "key.eqx", key)

    # 3) Metadata
    meta = {
        "step": int(step),
        "config": config,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(path: Path, *, model_template, opt_state_template=None, pool_state_template=None, key_template=None):
    model = eqx.tree_deserialise_leaves(path / "model.eqx", model_template)

    opt_state = None
    if opt_state_template is not None:
        opt_state = eqx.tree_deserialise_leaves(path / "opt_state.eqx", opt_state_template)

    pool_state = None
    if pool_state_template is not None:
        pool_state = eqx.tree_deserialise_leaves(path / "pool_state.eqx", pool_state_template)

    key = None
    if key_template is not None:
        key = eqx.tree_deserialise_leaves(path / "key.eqx", key_template)

    meta = json.loads((path / "meta.json").read_text())
    return model, opt_state, pool_state, key, meta
