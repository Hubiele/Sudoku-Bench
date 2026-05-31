import jax.numpy as jnp


# Identity filter
def get_identity():
    return jnp.zeros((3, 3), dtype=jnp.float32).at[1, 1].set(1.0)


# 8-neighborhood Laplace
def get_laplace8():
    return jnp.array([
        [0.5, 1.0, 0.5],
        [1.0, -6.0, 1.0],
        [0.5, 1.0, 0.5],
    ], dtype=jnp.float32) / 6.0


# 4-neighborhood Laplace (plus)
def get_laplace4():
    return jnp.array([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=jnp.float32)


# Uniform box-avg
def get_unibox():
    return jnp.ones((3, 3), dtype=jnp.float32) / 9.0


# Sobel x
def get_sobelx():
    return jnp.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ], dtype=jnp.float32)


# Sobel y
def get_sobely():
    return jnp.array([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ], dtype=jnp.float32)


# --- REGISTRY: Koblingen mellom tekst og funksjon ---

FILTER_REGISTRY = {
    "identity": get_identity,
    "laplace8": get_laplace8,
    "laplace4": get_laplace4,
    "unibox": get_unibox,
    "sobelx": get_sobelx,
    "sobely": get_sobely,
}


def create_filter_bank(filter_names):
    """
    Creates a filter bank based on a list of names

    Args:
        Filter names: a list of names, e.g. ['identity', 'sobelx'],

    Returns
        jnp.ndarray: a stacked JAX-array of n filters with shape: (n, 3, 3)
    """
    filters = []
    for name in filter_names:
        if name in FILTER_REGISTRY:
            # Call the function to get the array
            filters.append(FILTER_REGISTRY[name]())
        else:
            raise ValueError(f"Ukjent filter: {name}")

    return jnp.stack(filters)
